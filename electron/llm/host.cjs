const path = require('path');

// electron/llm/host.cjs
// The main process's half of the local-LLM worker: spawn it, correlate replies, kill it when idle.
//
// Same shape as electron/analysis/host.cjs, and for the same reasons: node-llama-cpp holds hundreds
// of megabytes of weights, and killing a child process returns every byte in a way dropping a handle
// inside the main process does not. It also keeps a native crash from taking the window down with it.
//
// One deliberate difference from the analysis worker: the model path is NOT argv at fork time. The
// user points this at any .gguf on disk and may change that setting while the app runs, so the path
// rides on every request and the worker reloads only when it sees a different one (see worker.cjs).

/** electron-store key holding the user-selected .gguf path. Read at request time, never cached here. */
const LOCAL_LLM_MODEL_PATH_KEY = 'LOCAL_LLM_MODEL_PATH';

/**
 * How long the worker is kept with nothing to do.
 *
 * Five minutes rather than the analysis worker's two: a GGUF reload is the whole cost of using this
 * feature (seconds of model load before the first token), and auto theme generation fires on song
 * change - a few minutes apart. The trade is the same one the analysis host documents: the idle kill
 * exists so a configured-but-unused local LLM does not sit on a gigabyte for the rest of the session.
 */
const IDLE_MS = 300_000;

/**
 * Hard ceiling on one request, past which the worker is killed rather than waited on.
 *
 * `complete` covers the slowest sanctioned case: a 4B-class model on pure CPU generating a full
 * DualTheme JSON. `test` is one short prompt and exists so the settings page cannot hang forever on a
 * model that loads but never produces a token. A deadline kill here costs only a reload - the worker
 * holds no state worth saving.
 */
const DEADLINE_MS = { complete: 300_000, test: 120_000 };

/** Answer for a request whose worker died under it. Never leaves this file - see `ask`. */
const CRASHED = Symbol('crashed');

/** Untrusted-renderer rejection for the handlers registered here, matching main.cjs's wording. */
const untrustedSenderError = () => new Error('Untrusted renderer attempted to use the local LLM IPC.');

let instance = null;

const createLlmHost = ({ app, ipcMain, dialog, store, isTrustedSender }) => {
    let child = null;
    let idleTimer = null;
    let quitting = false;
    let nextId = 0;
    const pending = new Map();

    const stop = () => {
        if (idleTimer) { clearTimeout(idleTimer); idleTimer = null; }
        const dying = child;
        child = null;
        for (const settle of pending.values()) settle(CRASHED);
        pending.clear();
        if (dying) dying.kill();
    };

    const touch = () => {
        if (idleTimer) clearTimeout(idleTimer);
        idleTimer = setTimeout(() => {
            if (pending.size) { touch(); return; }
            console.log('[llm] worker idle, stopping');
            stop();
        }, IDLE_MS);
        if (idleTimer.unref) idleTimer.unref();
    };

    const spawn = () => {
        if (child) return child;
        // Required here rather than at the top: this module is loaded during main's own startup,
        // and utilityProcess is only meaningful once the app is running.
        const { utilityProcess, app } = require('electron');
        // Packaged builds load node-llama-cpp from the real module tree afterPack laid down in
        // resources/llm-runtime (see build/afterPack.cjs): the runtime is an ESM package whose own
        // bare imports cannot resolve through Electron's asar layouts. Dev runs resolve normally.
        const runtimeDir = app?.isPackaged
            ? path.join(process.resourcesPath, 'llm-runtime', 'node_modules')
            : '';
        const started = utilityProcess.fork(path.join(__dirname, 'worker.cjs'), [], {
            // Piped and forwarded by hand, for the same measured reason as the analysis worker:
            // `inherit` silently drops the child's output on Windows.
            stdio: 'pipe',
            serviceName: 'folia-local-llm',
            env: { ...process.env, ...(runtimeDir ? { FOLIA_LLM_RUNTIME_DIR: runtimeDir } : {}) },
        });
        child = started;

        const forward = (stream) => stream?.on('data', (chunk) => {
            process.stdout.write(`[llm] ${String(chunk)}`);
        });
        forward(started.stdout);
        forward(started.stderr);

        started.on('message', ({ id, result }) => {
            const settle = pending.get(id);
            if (settle) settle(result ?? CRASHED);
        });
        started.on('exit', (code) => {
            if (child !== started || quitting) return;
            console.warn(`[llm] worker exited on its own (${code})`);
            stop();
        });
        return started;
    };

    const send = (kind, payload) => new Promise((resolve) => {
        const id = nextId += 1;
        const budget = DEADLINE_MS[kind];
        let timer = null;
        const settle = (value) => {
            if (!pending.has(id)) return;
            pending.delete(id);
            if (timer) clearTimeout(timer);
            resolve(value);
        };
        pending.set(id, settle);
        if (budget) {
            timer = setTimeout(() => {
                if (!pending.has(id)) return;
                console.warn(`[llm] ${kind} exceeded ${budget / 1000}s with no reply - killing the worker`);
                stop();
            }, budget);
            timer.unref?.();
        }
        touch();
        spawn().postMessage({ id, kind, payload });
    });

    /**
     * One retry on a crashed worker, mirroring the analysis host: the crash may be the idle kill
     * racing an in-flight request, and a re-fork costs a reload, not a lost answer. The second crash
     * is the runtime, and that is surfaced as an error instead of retried forever.
     */
    const ask = async (kind, payload) => {
        let answer = await send(kind, payload);
        if (answer === CRASHED) answer = await send(kind, payload);
        if (answer === CRASHED) {
            throw new Error('本地模型推理进程崩溃或超时，请检查模型文件是否完整，或换一个更小的 GGUF 模型。');
        }
        if (answer && answer.ok) return answer.text;
        throw new Error((answer && answer.error) || 'Local LLM returned no answer.');
    };

    /** The .gguf path the user configured, or a rejection naming the setting that is missing. */
    const requireModelPath = () => {
        const modelPath = store.get(LOCAL_LLM_MODEL_PATH_KEY);
        if (typeof modelPath === 'string' && modelPath.trim()) {
            return modelPath.trim();
        }
        throw new Error('尚未选择本地 GGUF 模型：请在 设置 → 桌面端 中选择 .gguf 模型文件。');
    };

    ipcMain.handle('choose-local-llm-model', async (event) => {
        if (!isTrustedSender?.(event.sender)) {
            throw untrustedSenderError();
        }
        const picked = await dialog.showOpenDialog({
            title: '选择 GGUF 模型文件',
            properties: ['openFile'],
            filters: [{ name: 'GGUF Model', extensions: ['gguf'] }],
        });
        if (picked.canceled || !picked.filePaths[0]) {
            return { canceled: true };
        }
        return { canceled: false, path: picked.filePaths[0] };
    });

    ipcMain.handle('test-local-llm', async (event) => {
        if (!isTrustedSender?.(event.sender)) {
            throw untrustedSenderError();
        }
        const startedAt = Date.now();
        await ask('test', { modelPath: requireModelPath() });
        return { latencyMs: Date.now() - startedAt };
    });

    app?.on('will-quit', () => { quitting = true; stop(); });

    instance = {
        stop,
        runCompletion: async ({ systemPrompt, sourcePrompt, schema, maxTokens, temperature }) => ask('complete', {
            modelPath: requireModelPath(),
            systemPrompt,
            sourcePrompt,
            schema: schema ?? null,
            maxTokens: maxTokens ?? null,
            temperature: typeof temperature === 'number' ? temperature : null,
        }),
    };
    return instance;
};

/**
 * Entry point used by aiTextClient.cjs's `runAiJsonCompletion` for `AI_PROVIDER === 'local'`.
 * Signature mirrors the cloud providers so the dispatch stays uniform.
 */
const runLocalJsonCompletion = async ({ store, systemPrompt, sourcePrompt, schema, schemaName, maxTokens, temperature }) => {
    if (!instance) {
        throw new Error('Local LLM host is not initialized');
    }
    console.log(`[ai] local completion via node-llama-cpp${schemaName ? ` (schema: ${schemaName})` : ''}`);
    return instance.runCompletion({ systemPrompt, sourcePrompt, schema, maxTokens, temperature });
};

module.exports = { createLlmHost, runLocalJsonCompletion, LOCAL_LLM_MODEL_PATH_KEY };
