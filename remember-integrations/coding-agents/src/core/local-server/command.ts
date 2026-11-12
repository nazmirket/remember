/** Invoke only Remember's checked-out embedded manager. */
export interface EmbedCommandOptions {
  embedPackagePath?: string;
}
export function getEmbedCommand(opts: EmbedCommandOptions = {}): string[] {
  if (!opts.embedPackagePath) throw new Error("Local remember-embed checkout is required");
  return ["uv", "run", "--directory", opts.embedPackagePath, "remember-embed"];
}
