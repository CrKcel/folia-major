import { createRequire } from 'node:module';
import { afterEach, describe, expect, it, vi } from 'vitest';

// test/unit/lyrics/aiTextClientLocalProvider.test.ts
// The local node-llama-cpp provider rides the same runAiJsonCompletion dispatch as the cloud ones,
// so a mistake there silently sends theme generation to Gemini - or worse, throws "GEMINI_API_KEY is
// not configured" on a fully offline setup. These tests pin the dispatch: what reaches the host, and
// that the host's own errors surface verbatim instead of being retried as cloud calls.

const client = createRequire(import.meta.url)('../../../electron/aiTextClient.cjs');
const hostModule = createRequire(import.meta.url)('../../../electron/llm/host.cjs');

const realRunLocalJsonCompletion = hostModule.runLocalJsonCompletion;

afterEach(() => {
    hostModule.runLocalJsonCompletion = realRunLocalJsonCompletion;
});

const run = (overrides: Record<string, unknown> = {}) => client.runAiJsonCompletion({
    store: {
        get: (k: string) => ({
            AI_PROVIDER: 'local',
            OPENAI_API_TEMPERATURE: '0.3',
        }[k]),
    },
    systemPrompt: 'sys',
    sourcePrompt: 'src',
    schema: { type: 'object' },
    schemaName: 'theme',
    maxTokens: 4096,
    ...overrides,
});

describe('local provider dispatch', () => {
    it('routes AI_PROVIDER=local to the llm host with schema and resolved temperature', async () => {
        const spy = vi.fn(async (_call: Record<string, unknown>) => '{"ok":true}');
        hostModule.runLocalJsonCompletion = spy;

        await expect(run()).resolves.toBe('{"ok":true}');
        expect(spy).toHaveBeenCalledTimes(1);
        const call = spy.mock.calls[0]![0];
        expect(call.schema).toEqual({ type: 'object' });
        expect(call.schemaName).toBe('theme');
        expect(call.temperature).toBe(0.3);
        expect(call.maxTokens).toBe(4096);
    });

    it('surfaces host errors verbatim - no cloud fallback, no retry', async () => {
        hostModule.runLocalJsonCompletion = vi.fn(async () => {
            throw new Error('尚未选择本地 GGUF 模型');
        });

        await expect(run()).rejects.toThrow('尚未选择本地 GGUF 模型');
    });

    it('still dispatches gemini for anything that is not openai or local', async () => {
        const spy = vi.fn(async () => '{"ok":true}');
        hostModule.runLocalJsonCompletion = spy;
        const fetchImpl = vi.fn(async () => Response.json({
            candidates: [{ content: { parts: [{ text: '{"ok":true}' }] } }],
        }));

        await expect(run({
            store: { get: (k: string) => ({ AI_PROVIDER: 'gemini', GEMINI_API_KEY: 'k' }[k]) },
            customFetch: fetchImpl,
        })).resolves.toBe('{"ok":true}');
        expect(spy).not.toHaveBeenCalled();
        expect(fetchImpl).toHaveBeenCalledTimes(1);
    });
});
