const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');

// electron/llm/worker.cjs
// The local LLM, in a process that owns no window.
//
// Same reason it lives outside main.cjs as the analysis worker does: the weights are hundreds of
// megabytes that only a child process's death reliably gives back, and a native crash must not take
// the window down. Nothing in this file may require 'electron' - it is plain Node, reached over
// utilityProcess. The model path arrives on every request (the user can change it while the app
// runs; see host.cjs), and a different path reloads the model on the spot.
//
// Output is constrained to the caller's JSON schema via llama.cpp GBNF grammars rather than trusted
// to prompt discipline. These are small local models - a 1.7B instruct model will wrap JSON in
// prose, drop a brace, or answer in the wrong language at the worst of times - and the schema is
// exactly the thing we know. With the grammar on, even the smallest viable model emits parseable
// JSON every time; without it, sanitizer retries would be the feature's main activity.

/**
 * Context window for every request.
 *
 * 8192 covers the worst sanctioned input with room to spare: ~2000 characters of lyrics plus the
 * theme system prompt on the way in, and a full DualTheme JSON (two colour sets plus extracted word
 * colours) on the way out. node-llama-cpp clamps this to the model's trained context, so a small
 * model just gets what it has. A KV cache this size is the worker's steady-state memory after load.
 */
const CONTEXT_SIZE = 8192;

/**
 * Cap on the test prompt's answer, which exists to time model load + one forward pass, not to
 * produce anything readable. The complete cap arrives per request from the caller's own token budget.
 */
const TEST_MAX_TOKENS = 16;

/** Loaded runtime, kept across requests. `path` is the model the current weights belong to. */
let loaded = null;

/**
 * node-llama-cpp is an ESM package with top-level await, so it must come through dynamic import -
 * require() rejects it outright (ERR_REQUIRE_ASYNC_MODULE). In packaged builds the bare specifier
 * cannot resolve either: Electron's asar patch covers CJS require, not ESM, so afterPack lays the
 * runtime and its dependency closure out as real files under resources/llm-runtime and the host
 * hands us that directory here. Dev runs resolve the bare specifier normally.
 */
const importNodeLlamaCpp = async () => {
    const runtimeDir = process.env.FOLIA_LLM_RUNTIME_DIR;
    if (!runtimeDir) {
        return import('node-llama-cpp');
    }
    const pkgJsonPath = path.join(runtimeDir, 'node-llama-cpp', 'package.json');
    const entry = JSON.parse(fs.readFileSync(pkgJsonPath, 'utf8')).main || 'dist/index.js';
    return import(pathToFileURL(path.join(runtimeDir, 'node-llama-cpp', entry)).href);
};

const getLlamaInstance = async () => {
    // Dynamic import, deliberately not require(): node-llama-cpp's entry is an ESM graph with
    // top-level await, which Node refuses to pull in through require() (ERR_REQUIRE_ASYNC_MODULE).
    const { getLlama } = await importNodeLlamaCpp();
    // `build: "never"`: a missing prebuilt binary must fail with a message the settings page can
    // show, not silently start a minutes-long llama.cpp source compile on the user's machine.
    return getLlama({ build: 'never' });
};

/**
 * Loads the requested .gguf, reusing whatever is already resident when the path is unchanged.
 * Returns null only for a clear precondition failure - everything else throws, because the caller
 * (and through it the renderer) needs the error text to act on.
 */
const ensureModel = async (modelPath) => {
    if (loaded && loaded.path === modelPath) {
        return loaded;
    }

    if (!fs.existsSync(modelPath)) {
        throw new Error(`模型文件不存在: ${modelPath}`);
    }
    if (fs.statSync(modelPath).isDirectory() || !modelPath.toLowerCase().endsWith('.gguf')) {
        throw new Error(`${modelPath} 不是 GGUF 模型文件，请在设置中选择 .gguf 文件。`);
    }

    if (loaded) {
        try { loaded.model?.dispose(); } catch { /* a dying model is still a dying model */ }
        loaded = null;
    }

    const startedAt = Date.now();
    const llama = await getLlamaInstance();
    const model = await llama.loadModel({ modelPath });
    const context = await model.createContext({ contextSize: CONTEXT_SIZE });
    loaded = { path: modelPath, llama, model, context };
    console.log(`[llm] model loaded in ${((Date.now() - startedAt) / 1000).toFixed(1)}s`
        + `, rss ${Math.round(process.memoryUsage().rss / 1048576)}MB: ${modelPath}`);
    return loaded;
};

/**
 * Adds the missing "#" to bare hex colour values.
 *
 * The shared sanitizer accepts only #-prefixed hex, and small local models omit the prefix
 * constantly (measured on a 0.8B quant) - without this, most generated themes collapse onto the
 * sanitizer's fallback palette. Scoped to the exact colour key names of the theme schema so it can
 * never touch free-form text like names or descriptions.
 */
const COLOR_KEY_HEX_PATTERN = /("(?:backgroundColor|primaryColor|accentColor|secondaryColor|color)"\s*:\s*)"([0-9a-fA-F]{6}|[0-9a-fA-F]{3})"/g;
const normalizeBareHexColors = (text) => text.replace(COLOR_KEY_HEX_PATTERN, '$1"#$2"');

/** One stateless completion: fresh sequence and session per request, so no chat state ever leaks. */
const runCompletion = async (payload) => {
    const { modelPath, systemPrompt, sourcePrompt, schema, maxTokens, temperature } = payload || {};
    if (typeof sourcePrompt !== 'string' || !sourcePrompt.trim()) {
        throw new Error('本地模型收到了空的输入提示。');
    }

    const resident = await ensureModel(modelPath);
    const sequence = resident.context.getSequence();
    try {
        // Same ESM-with-top-level-await constraint as getLlamaInstance above.
        const { LlamaChatSession, resolveChatWrapper, ChatMLChatWrapper } = await importNodeLlamaCpp();
        // Some third-party quants ship without an embedded chat template, and then the default
        // wrapper resolution returns undefined - measured on Qwen3.5-0.8B-Q4_K_M: plain generation
        // still works, but grammar-constrained generation dies on the spot with zero tokens.
        // ChatML is Qwen-family's wire format and the sensible fallback there; models with a
        // template keep their resolved wrapper.
        const chatWrapper = resolveChatWrapper({ model: resident.model }) ?? new ChatMLChatWrapper();
        const session = new LlamaChatSession({
            contextSequence: sequence,
            systemPrompt: typeof systemPrompt === 'string' ? systemPrompt : '',
            chatWrapper,
        });
        // The whole reason this feature works on small models: generation-level JSON enforcement.
        // Compiling the schema into a grammar is cheap and per-request, because the schema arrives
        // over IPC as a fresh object every time and there is nothing sound to cache it by.
        const grammar = schema && typeof schema === 'object'
            ? await resident.llama.createGrammarForJsonSchema(schema)
            : null;
        const startedAt = Date.now();
        const text = await session.prompt(sourcePrompt, {
            ...(Number.isFinite(maxTokens) && maxTokens > 0 ? { maxTokens } : {}),
            ...(Number.isFinite(temperature) && temperature >= 0 && temperature <= 2 ? { temperature } : {}),
            ...(grammar ? { grammar } : {}),
        });
        console.log(`[llm] completion finished in ${((Date.now() - startedAt) / 1000).toFixed(1)}s`);
        return normalizeBareHexColors(text);
    } finally {
        sequence.dispose();
    }
};

const HANDLERS = {
    complete: runCompletion,
    // The settings page's test button: model load plus one tiny answer, timed. Nothing readable is
    // expected back - the point is "this model loads and produces tokens on this machine".
    test: async (payload) => {
        await runCompletion({
            ...payload,
            systemPrompt: 'You are a connection test.',
            sourcePrompt: 'Reply with the single word OK.',
            maxTokens: TEST_MAX_TOKENS,
        });
        return 'ok';
    },
};

/** One at a time. Two concurrent inferences would double the machine share and thrash the KV cache. */
let queue = Promise.resolve();

process.parentPort.on('message', ({ data }) => {
    const { id, kind, payload } = data || {};
    const handler = HANDLERS[kind];
    const run = queue.then(async () => {
        if (!handler) {
            return { ok: false, error: `Local LLM worker has no handler for "${kind}"` };
        }
        try {
            const text = await handler(payload);
            return { ok: true, text };
        } catch (error) {
            // Errors are values here, not rejections: the host distinguishes "the model said no"
            // (must reach the renderer verbatim) from "the process died" (its own retry and message).
            const message = error instanceof Error ? error.message : String(error);
            console.warn(`[llm] ${kind} failed: ${message}`);
            return { ok: false, error: message };
        }
    });
    // The queue must not inherit a rejection, or one failure poisons every later request.
    queue = run.then(() => undefined, () => undefined);
    run.then(result => process.parentPort.postMessage({ id, result }));
});
