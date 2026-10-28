from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from town_model.dataset import generator


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _make_remote(tmp_path: Path) -> tuple[Path, str]:
    upstream = tmp_path / "upstream"
    subprocess.run(["git", "init", "-b", "main", str(upstream)], check=True, capture_output=True)
    _git(upstream, "config", "user.name", "Town Test")
    _git(upstream, "config", "user.email", "town-test@example.invalid")
    source_file = upstream / "libs" / "pizzaiolo.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("class Pizzaiolo: pass\n", encoding="utf-8")
    _git(upstream, "add", ".")
    _git(upstream, "commit", "-m", "test source")
    commit = _git(upstream, "rev-parse", "HEAD")

    remote = tmp_path / "pizzaiolo.git"
    subprocess.run(
        ["git", "clone", "--bare", str(upstream), str(remote)],
        check=True,
        capture_output=True,
    )
    return remote, commit


@pytest.mark.parametrize("pre_existing_unchecked_clone", [False, True])
def test_pinned_source_cache_checks_out_files(
    tmp_path: Path, monkeypatch, pre_existing_unchecked_clone: bool
) -> None:
    remote, commit = _make_remote(tmp_path)
    monkeypatch.setattr(generator, "GENERATOR_REPOSITORY", str(remote))
    monkeypatch.setattr(generator, "GENERATOR_COMMIT", commit)
    monkeypatch.delenv("PIZZAIOLO_SOURCE_DIR", raising=False)

    project_root = tmp_path / "project"
    cache = project_root / "artifacts" / "sources" / "pizzaiolo"
    if pre_existing_unchecked_clone:
        cache.parent.mkdir(parents=True)
        subprocess.run(
            ["git", "clone", "--no-checkout", str(remote), str(cache)],
            check=True,
            capture_output=True,
        )

    checkout = generator.ensure_pizzaiolo_repo(project_root)

    assert (checkout / "libs" / "pizzaiolo.py").is_file()
    assert _git(checkout, "status", "--porcelain") == ""
