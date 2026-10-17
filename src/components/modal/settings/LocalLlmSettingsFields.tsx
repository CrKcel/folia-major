import React, { useState } from 'react';
import { FolderOpen, Loader2, Play } from 'lucide-react';
import { useTranslation } from 'react-i18next';

// src/components/modal/settings/LocalLlmSettingsFields.tsx
// Fields for the "local model" AI provider: the .gguf path on disk plus a load test.
// Split out of DesktopSettingsSubview because the picker/test interaction has its own state
// (see skills/file-modularization); the provider button row stays in the parent.

type LocalLlmSettingsFieldsProps = {
    modelPath: string;
    onModelPathChange: (path: string) => void;
    fieldClass: string;
    isDaylight: boolean;
};

type TestStatus = 'idle' | 'testing' | 'ok' | 'error';

const LocalLlmSettingsFields: React.FC<LocalLlmSettingsFieldsProps> = ({
    modelPath,
    onModelPathChange,
    fieldClass,
    isDaylight,
}) => {
    const { t } = useTranslation();
    const [testStatus, setTestStatus] = useState<TestStatus>('idle');
    const [testError, setTestError] = useState('');
    const [testLatencyMs, setTestLatencyMs] = useState(0);

    const ghostButtonClass = isDaylight
        ? 'border-black/10 bg-black/[0.025] hover:bg-black/[0.055]'
        : 'border-white/10 bg-white/5 hover:bg-white/10';

    // Native .gguf picker. The result only lands in the form state - the save button persists it
    // like every other field on this page.
    const handleChooseModel = async () => {
        const picked = await window.electron?.chooseLocalLlmModel?.();
        if (picked && !picked.canceled && picked.path) {
            onModelPathChange(picked.path);
        }
    };

    // Loads the configured model and answers one short prompt, so a bad download or an
    // unsupported quantization is discovered here instead of on the first theme generation.
    const handleTest = async () => {
        setTestStatus('testing');
        setTestError('');
        try {
            const result = await window.electron?.testLocalLlm?.();
            setTestLatencyMs(result?.latencyMs ?? 0);
            setTestStatus('ok');
        } catch (error) {
            setTestError(error instanceof Error ? error.message : String(error));
            setTestStatus('error');
        }
    };

    return (
        <>
            <label className="block space-y-1 text-left">
                <span className="text-xs opacity-60" style={{ color: 'var(--text-secondary)' }}>
                    {t('options.localLlmModelPath')}
                </span>
                <div className="flex gap-2">
                    <input
                        type="text"
                        value={modelPath}
                        onChange={(e) => onModelPathChange(e.target.value)}
                        placeholder="~/models/qwen3-1.7b-q4_k_m.gguf"
                        className={fieldClass}
                        style={{ color: 'var(--text-primary)' }}
                    />
                    <button
                        type="button"
                        onClick={handleChooseModel}
                        className={`shrink-0 inline-flex items-center gap-1.5 rounded-lg border px-3 py-2 text-xs font-medium transition-all active:scale-95 ${ghostButtonClass}`}
                        style={{ color: 'var(--text-primary)' }}
                    >
                        <FolderOpen size={14} />
                        {t('options.localLlmChooseModel')}
                    </button>
                </div>
                <span className="block text-[10px] opacity-45 leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                    {t('options.localLlmModelPathDesc')}
                </span>
            </label>

            <div className="flex items-center gap-3">
                <button
                    type="button"
                    onClick={handleTest}
                    disabled={testStatus === 'testing' || !modelPath.trim()}
                    className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition-all active:scale-95 disabled:cursor-not-allowed disabled:opacity-40 ${ghostButtonClass}`}
                    style={{ color: 'var(--text-primary)' }}
                >
                    {testStatus === 'testing'
                        ? <Loader2 size={14} className="animate-spin" />
                        : <Play size={14} />}
                    {t('options.localLlmTest')}
                </button>
                {testStatus === 'ok' && (
                    <span className="text-xs text-emerald-500">
                        {t('options.localLlmTestOk', { seconds: (testLatencyMs / 1000).toFixed(1) })}
                    </span>
                )}
                {testStatus === 'error' && (
                    <span className="text-xs text-red-500 truncate" title={testError}>
                        {testError}
                    </span>
                )}
            </div>
        </>
    );
};

export default LocalLlmSettingsFields;
