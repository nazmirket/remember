/** Skill directories for the supported coding agents, relative to the user's home. */
export const SKILL_DIRS: Record<string, string[]> = {
  "claude-code": [".claude", "skills"],
  codex: [".agents", "skills"],
};
export const resolveSkillDirs = (harness: string, _home: string): string[] | undefined =>
  SKILL_DIRS[harness];
