import { homedir } from "node:os";
import { isAbsolute, join, resolve } from "node:path";

export function rememberHome(home = homedir(), env: NodeJS.ProcessEnv = process.env): string {
  const override = env.REMEMBER_HOME?.trim();
  return override ? (isAbsolute(override) ? override : resolve(override)) : join(home, ".remember");
}
