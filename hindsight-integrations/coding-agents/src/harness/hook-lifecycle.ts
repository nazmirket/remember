/** Shared hook lifecycle for the two supported coding agents. */
import { runHook, type HookSpec } from "../core/hook";
import { runRetainHook, type RetainHookSpec } from "../core/retain-hook";
import { runSessionStartHook, type SessionStartHookSpec } from "../core/session-start";
import { readCodexTranscript } from "../core/transcript-codex";

export type HookHarnessName = "claude-code" | "codex";
export type HookLifecycle = "sessionStart" | "prompt" | "stop";
export interface HookInstallSpec {
  event: string;
  entry: string;
  timeout?: number;
}
export interface HookHarnessSpec {
  configStyle: "nested";
  install: Record<HookLifecycle, HookInstallSpec>;
  sessionStart: SessionStartHookSpec;
  prompt: HookSpec;
  retain: RetainHookSpec;
}

const standardSessionStart = (harness: HookHarnessName): SessionStartHookSpec => ({
  harness,
  parse: (ev) => ({
    cwd: ev.cwd as string | undefined,
    sessionId: ev.session_id as string | undefined,
  }),
  emit: (out) => ({
    ...(out.systemMessage ? { systemMessage: out.systemMessage } : {}),
    hookSpecificOutput: {
      hookEventName: "SessionStart",
      ...(out.additionalContext ? { additionalContext: out.additionalContext } : {}),
    },
  }),
});
const claudePrompt: HookSpec = {
  harness: "claude-code",
  parse: (ev) => ({
    prompt: ev.prompt as string | undefined,
    cwd: ev.cwd as string | undefined,
    sessionId: ev.session_id as string | undefined,
  }),
  emit: (context, notice) => ({
    ...(notice ? { systemMessage: notice } : {}),
    hookSpecificOutput: { hookEventName: "UserPromptSubmit", additionalContext: context },
  }),
};
const codexPrompt: HookSpec = {
  ...claudePrompt,
  harness: "codex",
  parse: (ev) => ({
    prompt: (ev.prompt as string | undefined) ?? (ev.user_prompt as string | undefined),
    cwd: ev.cwd as string | undefined,
    sessionId: ev.session_id as string | undefined,
  }),
};
const notGrokHosted = (): boolean => !process.env.GROK_HOOK_EVENT;

export const HOOK_HARNESSES: Record<HookHarnessName, HookHarnessSpec> = {
  "claude-code": {
    configStyle: "nested",
    install: {
      sessionStart: { event: "SessionStart", entry: "claude-sessionstart-hook.js", timeout: 30 },
      prompt: { event: "UserPromptSubmit", entry: "claude-hook.js", timeout: 30 },
      stop: { event: "Stop", entry: "claude-stop-hook.js", timeout: 60 },
    },
    sessionStart: { ...standardSessionStart("claude-code"), accept: notGrokHosted },
    prompt: { ...claudePrompt, accept: notGrokHosted },
    retain: {
      hostTimeoutSec: 60,
      harness: "claude-code",
      accept: notGrokHosted,
      parse: (ev) => ({
        sessionId: ev.session_id as string | undefined,
        transcriptPath: ev.transcript_path as string | undefined,
        cwd: ev.cwd as string | undefined,
        lastAssistantMessage: ev.last_assistant_message as string | undefined,
      }),
    },
  },
  codex: {
    configStyle: "nested",
    install: {
      sessionStart: { event: "SessionStart", entry: "codex-sessionstart-hook.js", timeout: 30 },
      prompt: { event: "UserPromptSubmit", entry: "codex-hook.js", timeout: 30 },
      stop: { event: "Stop", entry: "codex-stop-hook.js", timeout: 60 },
    },
    sessionStart: standardSessionStart("codex"),
    prompt: codexPrompt,
    retain: {
      hostTimeoutSec: 60,
      harness: "codex",
      parse: (ev) => ({
        sessionId: ev.session_id as string | undefined,
        transcriptPath: ev.transcript_path as string | undefined,
        cwd: ev.cwd as string | undefined,
      }),
      readTranscript: readCodexTranscript,
    },
  },
};

export const runHarnessSessionStart = (harness: HookHarnessName): Promise<void> =>
  runSessionStartHook(HOOK_HARNESSES[harness].sessionStart);
export const runHarnessPrompt = (harness: HookHarnessName): Promise<void> =>
  runHook(HOOK_HARNESSES[harness].prompt);
export const runHarnessRetain = (harness: HookHarnessName): Promise<void> =>
  runRetainHook(HOOK_HARNESSES[harness].retain);
