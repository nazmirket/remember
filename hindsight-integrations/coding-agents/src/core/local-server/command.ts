/** Invoke only this fork's checked-out embedded manager. */
export interface EmbedCommandOptions {
  embedPackagePath?: string;
}
export function getEmbedCommand(opts: EmbedCommandOptions = {}): string[] {
  if (!opts.embedPackagePath) throw new Error("Local hindsight-embed checkout is required");
  return ["uv", "run", "--directory", opts.embedPackagePath, "hindsight-embed"];
}
