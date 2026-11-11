"""The headless development entrypoint runs the checkout that owns it."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("caller", ["checkout", "outside", "other-checkout"])
def test_dev_entrypoint_resolves_its_own_checkout(tmp_path: Path, caller: str) -> None:
    root = tmp_path / "hindsight checkout"
    scripts = root / "scripts" / "dev"
    scripts.mkdir(parents=True)
    for name in ("start.sh", "start-api.sh"):
        shutil.copy2(SOURCE_ROOT / "scripts" / "dev" / name, scripts / name)
    (root / ".env").write_text("HINDSIGHT_API_PORT=18888\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    other = tmp_path / "other checkout"
    other.mkdir()
    (other / ".env").write_text("HINDSIGHT_API_PORT=28888\n")
    cwd = {"checkout": root, "outside": outside, "other-checkout": other}[caller]

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python3").symlink_to(sys.executable)
    calls = tmp_path / "calls.json"
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "open(os.environ['DEV_SCRIPT_CALLS'], 'w').write(json.dumps({"
        "'args': sys.argv[1:], 'cwd': os.getcwd(), 'port': os.environ.get('HINDSIGHT_API_PORT')}))\n"
    )
    uv.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("HINDSIGHT_")}
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["DEV_SCRIPT_CALLS"] = str(calls)
    result = subprocess.run(["bash", str(scripts / "start.sh")], cwd=cwd, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    record = json.loads(calls.read_text())
    assert record == {"args": ["run", "hindsight-api", "--host", "127.0.0.1"], "cwd": str(root), "port": "18888"}
