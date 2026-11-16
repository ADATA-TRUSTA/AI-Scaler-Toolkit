"""comfyui._move_into_place falls back to a copy when a rename is denied."""

from pathlib import Path

from launcher import comfyui


def test_rename_denied_falls_back_to_copy(tmp_path, monkeypatch):
    src = tmp_path / "work" / "ComfyUI-1"
    src.mkdir(parents=True)
    (src / "main.py").write_text("x", encoding="utf-8")
    dest = tmp_path / "ComfyUI"

    def denied(self, target):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(Path, "replace", denied)
    comfyui._move_into_place(src, dest)
    assert (dest / "main.py").is_file()
    assert not src.exists()


def test_plain_rename_still_used(tmp_path):
    src = tmp_path / "a"
    src.mkdir()
    (src / "f").write_text("1", encoding="utf-8")
    comfyui._move_into_place(src, tmp_path / "b")
    assert (tmp_path / "b" / "f").is_file()
