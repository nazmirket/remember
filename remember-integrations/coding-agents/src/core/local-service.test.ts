import { realpathSync } from "node:fs";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { RememberClient } from "./remember";
import { daemonEnv } from "./daemon";
import { resolveConfig } from "./config";

const embed = join(process.cwd(), "..", "..", "remember-embed");
const source = realpathSync(
  join(embed, "..", "remember-api-slim", "remember_api", "api", "http.py")
);

afterEach(() => vi.unstubAllGlobals());

describe("local memory boundaries", () => {
  it("does not send a recall or token to an unrelated service on the local port", async () => {
    const fetch = vi
      .fn()
      .mockResolvedValue(new Response(JSON.stringify({ status: "healthy" }), { status: 200 }));
    vi.stubGlobal("fetch", fetch);
    const client = new RememberClient({
      apiUrl: "http://127.0.0.1:8888",
      checkoutEmbedPath: embed,
      bank: "test",
      apiToken: "secret",
    });
    await expect(client.reflect("private context", { timeoutMs: 1000 })).rejects.toThrow(
      "does not belong to this Remember checkout"
    );
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch.mock.calls[0][0]).toBe("http://127.0.0.1:8888/health");
    expect(fetch.mock.calls[0][1].headers).toBeUndefined();
  });

  it("accepts the checked-out service and sends the memory request", async () => {
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ checkout_source: source }), { status: 200 })
      )
      .mockResolvedValueOnce(new Response(JSON.stringify({ text: "answer" }), { status: 200 }));
    vi.stubGlobal("fetch", fetch);
    const client = new RememberClient({
      apiUrl: "http://127.0.0.1:8888",
      checkoutEmbedPath: embed,
      bank: "test",
    });
    await client.reflect("question", { timeoutMs: 1000 });
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(fetch.mock.calls[1][0]).toContain("/reflect");
  });

  it("rejects an external endpoint before issuing any request", async () => {
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);
    const client = new RememberClient({
      apiUrl: "https://example.com",
      checkoutEmbedPath: embed,
      bank: "test",
    });
    await expect(client.reflect("private context", { timeoutMs: 1000 })).rejects.toThrow(
      "local daemon"
    );
    expect(fetch).not.toHaveBeenCalled();
  });

  it("overrides inherited remote database and bind settings", () => {
    const cfg = resolveConfig({ daemonProfile: "my/profile" });
    const env = daemonEnv(cfg, {
      REMEMBER_API_DATABASE_URL: "postgres://remote/db",
      REMEMBER_API_HOST: "0.0.0.0",
    });
    expect(env.REMEMBER_API_DATABASE_URL).toBe("pg0://remember-embed-my-profile");
    expect(env.REMEMBER_EMBED_API_DATABASE_URL).toBe(env.REMEMBER_API_DATABASE_URL);
    expect(env.REMEMBER_API_HOST).toBe("127.0.0.1");
  });
});
