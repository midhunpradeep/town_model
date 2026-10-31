from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
import threading

import pytest

import town_model.inspection.relation_diagnostics as relation_diagnostics
from town_model.inspection.relation_diagnostics import save_relation_diagnostic


def _publish_process(result, destination, outcome_queue, barrier=None):
    """Independent process publisher; optionally pause at the atomic boundary."""
    if barrier is not None:
        real_link = os.link
        real_replace = os.replace

        def synchronized_link(source, target, *args, **kwargs):
            barrier.wait(timeout=20)
            return real_link(source, target, *args, **kwargs)

        def synchronized_replace(source, target, *args, **kwargs):
            barrier.wait(timeout=20)
            return real_replace(source, target, *args, **kwargs)

        relation_diagnostics.os.link = synchronized_link
        relation_diagnostics.os.replace = synchronized_replace

    try:
        published = save_relation_diagnostic(result, destination)
        outcome_queue.put(("published", result["writer"], str(published)))
    except FileExistsError:
        outcome_queue.put(("exists", result["writer"], None))
    except Exception as error:  # surfaced in the parent as an assertion failure
        outcome_queue.put(("error", result["writer"], repr(error)))


def _large_result(writer: str) -> dict:
    return {"writer": writer, "payload": list(range(10_000))}


def _temporary_entries(destination: Path) -> list[Path]:
    return list(destination.parent.glob(f".{destination.name}.*.tmp"))


def test_single_writer_publishes_complete_json_and_cleans_temporary_file(tmp_path) -> None:
    destination = tmp_path / "single.json"
    result = _large_result("solo")

    published = save_relation_diagnostic(result, destination)

    assert published == destination
    assert json.loads(destination.read_text(encoding="utf-8")) == result
    assert _temporary_entries(destination) == []


def test_existing_regular_destination_is_immutable(tmp_path) -> None:
    destination = tmp_path / "existing.json"
    original = '{"original": true}\n'
    destination.write_text(original, encoding="utf-8")

    with pytest.raises(FileExistsError, match="immutable.*already exists"):
        save_relation_diagnostic(_large_result("loser"), destination)

    assert destination.read_text(encoding="utf-8") == original
    assert _temporary_entries(destination) == []


def test_existing_symlink_and_dangling_symlink_are_occupied_destinations(tmp_path) -> None:
    target = tmp_path / "symlink-target.json"
    target.write_text('{"owner": "target"}\n', encoding="utf-8")
    destination = tmp_path / "existing-link.json"
    destination.symlink_to(target)

    with pytest.raises(FileExistsError, match="immutable.*already exists"):
        save_relation_diagnostic(_large_result("loser"), destination)

    assert destination.is_symlink()
    assert target.read_text(encoding="utf-8") == '{"owner": "target"}\n'

    dangling_target = tmp_path / "not-created.json"
    dangling = tmp_path / "dangling-link.json"
    dangling.symlink_to(dangling_target)
    with pytest.raises(FileExistsError, match="immutable.*already exists"):
        save_relation_diagnostic(_large_result("loser"), dangling)
    assert dangling.is_symlink()
    assert not dangling_target.exists()
    assert _temporary_entries(destination) == []
    assert _temporary_entries(dangling) == []


def test_two_independent_processes_have_one_atomic_no_clobber_winner(tmp_path) -> None:
    context = multiprocessing.get_context("spawn")
    destination = tmp_path / "contended.json"
    barrier = context.Barrier(3)
    outcome_queue = context.Queue()
    results = [_large_result("writer-a"), _large_result("writer-b")]
    processes = [
        context.Process(target=_publish_process, args=(result, destination, outcome_queue, barrier))
        for result in results
    ]
    reader_started = threading.Event()
    reader_stop = threading.Event()
    reader_errors: list[BaseException] = []
    reader_payloads: list[dict] = []

    def observe_publication() -> None:
        reader_started.set()
        barrier.wait(timeout=20)
        while not reader_stop.is_set():
            try:
                text = destination.read_text(encoding="utf-8")
            except FileNotFoundError:
                continue
            try:
                reader_payloads.append(json.loads(text))
            except BaseException as error:
                reader_errors.append(error)
            return

    for process in processes:
        process.start()
    reader = threading.Thread(target=observe_publication, daemon=True)
    reader.start()
    assert reader_started.wait(timeout=5)
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    reader_stop.set()
    reader.join(timeout=5)

    outcomes = [outcome_queue.get(timeout=5) for _ in processes]
    assert sorted(outcome[0] for outcome in outcomes) == ["exists", "published"]
    assert reader_errors == []
    assert len(reader_payloads) == 1
    winner = next(outcome[1] for outcome in outcomes if outcome[0] == "published")
    expected = next(result for result in results if result["writer"] == winner)
    assert json.loads(destination.read_text(encoding="utf-8")) == expected
    assert reader_payloads[0] == expected
    assert _temporary_entries(destination) == []


def test_separate_process_retries_cannot_replace_first_publication(tmp_path) -> None:
    context = multiprocessing.get_context("spawn")
    destination = tmp_path / "sequential-processes.json"

    first_queue = context.Queue()
    first = context.Process(target=_publish_process, args=(_large_result("first"), destination, first_queue))
    first.start()
    first.join(timeout=20)
    assert first.exitcode == 0
    assert first_queue.get(timeout=5)[0] == "published"
    saved = destination.read_bytes()

    second_queue = context.Queue()
    second = context.Process(target=_publish_process, args=(_large_result("second"), destination, second_queue))
    second.start()
    second.join(timeout=20)
    assert second.exitcode == 0
    assert second_queue.get(timeout=5)[0] == "exists"

    assert destination.read_bytes() == saved
    assert json.loads(saved)["writer"] == "first"
    assert _temporary_entries(destination) == []


def test_serialization_failure_removes_temporary_file_and_publishes_nothing(tmp_path) -> None:
    destination = tmp_path / "serialization-failure.json"

    with pytest.raises(TypeError, match="not JSON serializable"):
        save_relation_diagnostic({"invalid": object()}, destination)

    assert not destination.exists()
    assert _temporary_entries(destination) == []


def test_failure_before_publication_preserves_destination_and_cleans_temp(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "permission-failure.json"

    def fail_link(*args, **kwargs):
        raise PermissionError("hard links are not permitted")

    monkeypatch.setattr(relation_diagnostics.os, "link", fail_link)
    with pytest.raises(PermissionError, match="atomic no-clobber hard-link publication failed"):
        save_relation_diagnostic(_large_result("writer"), destination)

    assert not destination.exists()
    assert _temporary_entries(destination) == []


def test_post_publication_cleanup_failure_leaves_safe_hardlink_for_recovery(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "cleanup-failure.json"
    real_unlink = os.unlink

    def fail_temporary_unlink(path, *args, **kwargs):
        if Path(path).name.startswith(f".{destination.name}."):
            raise PermissionError("simulated interruption before temporary unlink")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(relation_diagnostics.os, "unlink", fail_temporary_unlink)
    result = _large_result("published-before-cleanup")
    with pytest.raises(PermissionError, match="simulated interruption"):
        save_relation_diagnostic(result, destination)

    leftovers = _temporary_entries(destination)
    assert len(leftovers) == 1
    assert destination.is_file()
    assert json.loads(destination.read_text(encoding="utf-8")) == result
    assert os.path.samefile(leftovers[0], destination)

    # Recovery is safe only for a temporary hard link to the already-published
    # destination; never sweep unrelated or still-being-written temporary files.
    monkeypatch.setattr(relation_diagnostics.os, "unlink", real_unlink)
    leftovers[0].unlink()
    assert _temporary_entries(destination) == []
    with pytest.raises(FileExistsError):
        save_relation_diagnostic(_large_result("must-not-replace"), destination)
