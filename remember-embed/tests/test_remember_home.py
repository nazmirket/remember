from remember_embed.remember_home import remember_home


def test_home_uses_remember_and_accepts_override(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("REMEMBER_HOME", raising=False)
    assert remember_home() == tmp_path / ".remember"
    monkeypatch.setenv("REMEMBER_HOME", str(tmp_path / "custom"))
    assert remember_home() == tmp_path / "custom"
