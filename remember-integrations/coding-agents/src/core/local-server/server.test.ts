import { describe, it, expect, vi } from "vitest";
import { RememberServer } from "./server.js";

const spawnMock = vi.hoisted(() => vi.fn());
vi.mock("child_process", () => ({ spawn: spawnMock }));

describe("RememberServer construction", () => {
  it("defaults base URL to http://127.0.0.1:8888", () => {
    const server = new RememberServer();
    expect(server.getBaseUrl()).toBe("http://127.0.0.1:8888");
    expect(server.getProfile()).toBe("default");
  });

  it("honours custom profile, port, and host", () => {
    const server = new RememberServer({
      embedPackagePath: "/fork/remember-embed",
      profile: "app",
      port: 9077,
      host: "0.0.0.0",
    });
    expect(server.getProfile()).toBe("app");
    expect(server.getBaseUrl()).toBe("http://0.0.0.0:9077");
  });

  it("accepts open env pass-through without complaining about unknown keys", () => {
    const server = new RememberServer({
      embedPackagePath: "/fork/remember-embed",
      env: {
        REMEMBER_API_LLM_PROVIDER: "openai",
        REMEMBER_API_LLM_MODEL: "gpt-4o-mini",
        // A field that does not exist today — should still be accepted
        REMEMBER_FUTURE_FLAG: "enabled",
      },
    });
    expect(server).toBeInstanceOf(RememberServer);
  });

  it("exposes checkHealth that returns false when no daemon is running", async () => {
    // Random high port that nothing is listening on.
    const server = new RememberServer({
      embedPackagePath: "/fork/remember-embed",
      port: 1,
      readyTimeoutMs: 100,
    });
    const healthy = await server.checkHealth();
    expect(healthy).toBe(false);
  });

  it("hides daemon command windows on Windows", async () => {
    const child = {
      stdout: { on: vi.fn() },
      stderr: { on: vi.fn() },
      on: vi.fn((event: string, handler: (code?: number) => void) => {
        if (event === "exit") handler(0);
        return child;
      }),
    };
    spawnMock.mockReturnValueOnce(child);

    const server = new RememberServer({ embedPackagePath: "/fork/remember-embed" });
    await server.stop();

    expect(spawnMock).toHaveBeenCalledWith(
      "uv",
      [
        "run",
        "--directory",
        "/fork/remember-embed",
        "remember-embed",
        "daemon",
        "--profile",
        "default",
        "stop",
      ],
      expect.objectContaining({ stdio: "pipe", windowsHide: true })
    );
  });

  it("uses the same caller-supplied environment for startup and shutdown", async () => {
    const child = {
      stdout: { on: vi.fn() },
      stderr: { on: vi.fn() },
      on: vi.fn((event: string, handler: (code?: number) => void) => {
        if (event === "exit") handler(0);
        return child;
      }),
    };
    spawnMock.mockReset();
    spawnMock.mockReturnValue(child);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("ready")));
    try {
      const server = new RememberServer({
        embedPackagePath: "/fork/remember-embed",
        env: {
          PATH: "/caller-toolchain/bin",
          REMEMBER_TEST_SENTINEL: "same-environment",
          REMEMBER_OMITTED: undefined,
        },
        platformCpuWorkaround: false,
      });
      await server.start();
      await server.stop();
      expect(spawnMock).toHaveBeenCalledTimes(3);
      for (const call of spawnMock.mock.calls) {
        expect(call[2].env).toEqual(
          expect.objectContaining({
            PATH: "/caller-toolchain/bin",
            REMEMBER_TEST_SENTINEL: "same-environment",
          })
        );
        expect(call[2].env.REMEMBER_OMITTED).toBeUndefined();
      }
    } finally {
      vi.unstubAllGlobals();
      spawnMock.mockReset();
    }
  });
});
