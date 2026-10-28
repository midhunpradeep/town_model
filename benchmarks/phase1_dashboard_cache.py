"""Reproduce Phase 1 inspection/explorer cache timings at 48 and 4,800 samples.

The 4,800-row fixture is a synthetic scale copy of the committed 48-row
example, not Pizzaïolo-generated data. It has 4,800 distinct image files and
14,400 distinct annotation files (the dimension-valid segmentation matrices
contain only background to keep fixture size modest), plus the larger
JSONL/manifest.
Run from the repository root with the project environment installed:

    python benchmarks/phase1_dashboard_cache.py --iterations 5
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import shutil
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Callable

from town_model.application import (
    _invalidate_dataset,
    inspect_dataset,
    load_explorer,
    load_verified_sample_image,
)
from town_model.dataset.constants import (
    AUDIT_DIR,
    MANIFEST_FILENAME,
    REPORT_DIR,
    SAMPLES_FILENAME,
    SPLIT_DIR,
    SPLIT_NAMES,
    VOCABULARY_DIR,
)
from town_model.dataset.schema import DatasetRecord, load_records, write_records, write_split_manifest
from town_model.dataset.splitting import assign_splits
from town_model.dataset.statistics import build_statistics
from town_model.dataset.validation import canonical_json_bytes, sha256_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPOSITORY_ROOT / "examples" / "pizza-48"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_asset(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _asset_checksums(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == MANIFEST_FILENAME or relative.startswith(f"{REPORT_DIR}/"):
            continue
        result[relative] = sha256_file(path)
    return result


def build_scaled_fixture(destination: Path, sample_count: int = 4_800) -> Path:
    """Create a validator-compatible, explicitly synthetic 4,800-sample fixture."""

    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    source_records = load_records(EXAMPLE / SAMPLES_FILENAME)
    source_rows: dict[str, dict[str, str]] = {}
    with (EXAMPLE / AUDIT_DIR / "generator_master.csv").open(encoding="utf-8", newline="") as file:
        source_rows = {row["ref"]: row for row in csv.DictReader(file)}
    with (EXAMPLE / AUDIT_DIR / "generator_master_with_source_labels.csv").open(encoding="utf-8", newline="") as file:
        labeled_rows = {row["ref"]: row for row in csv.DictReader(file)}
    raw_fields = list(next(iter(source_rows.values())).keys())
    labeled_fields = list(next(iter(labeled_rows.values())).keys())

    for relative in (
        "ontology/pizzaiolo.xml",
        "labels/concepts.json",
        f"{VOCABULARY_DIR}/ontology_classes.json",
        f"{VOCABULARY_DIR}/detector_categories.json",
    ):
        source = EXAMPLE / relative
        destination_path = destination / relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination_path)
    for split in SPLIT_NAMES:
        (destination / SPLIT_DIR).mkdir(parents=True, exist_ok=True)
    (destination / REPORT_DIR).mkdir(parents=True, exist_ok=True)

    clones: list[dict] = []
    raw_clones: list[dict[str, str]] = []
    labeled_clones: list[dict[str, str]] = []
    for index in range(sample_count):
        original = source_records[index % len(source_records)]
        original_data = copy.deepcopy(original.to_dict())
        sample_id = f"scale_{index:05d}"
        image_path = f"images/{sample_id}.png"
        source_annotation_paths = original.source_annotations
        renamed_annotations = {
            kind: f"labels/{sample_id}_{kind}.{('txt' if kind == 'segmentation' else 'json')}"
            for kind in ("boxes", "contours", "segmentation")
        }
        _copy_asset(EXAMPLE / original.image_path, destination / image_path)
        for kind, old_relative in source_annotation_paths.items():
            target = destination / renamed_annotations[kind]
            if kind == "segmentation":
                target.parent.mkdir(parents=True, exist_ok=True)
                row = " ".join(["0"] * 224) + "\n"
                target.write_text(row * 224, encoding="utf-8")
            else:
                _copy_asset(EXAMPLE / old_relative, target)

        original_data["sample_id"] = sample_id
        original_data["image"]["path"] = image_path
        original_data["audit"]["raw_generator"]["source_ref"] = sample_id
        original_data["audit"]["raw_generator"]["original_annotation_filenames"] = {
            kind: Path(relative).name for kind, relative in renamed_annotations.items()
        }
        original_data["source_annotations"] = renamed_annotations
        for component in original_data["components"]:
            component["component_id"] = component["component_id"].replace(original.sample_id, sample_id, 1)
            component["annotation_path"] = renamed_annotations["boxes"]
        clones.append(original_data)

        raw_row = dict(source_rows[original.sample_id])
        raw_row.update(
            ref=sample_id,
            img_name=f"{sample_id}.png",
            boxes_name=Path(renamed_annotations["boxes"]).name,
            contours_name=Path(renamed_annotations["contours"]).name,
            segmentation_name=Path(renamed_annotations["segmentation"]).name,
        )
        raw_clones.append(raw_row)
        labeled_row = dict(labeled_rows[original.sample_id])
        labeled_row.update(raw_row)
        labeled_clones.append(labeled_row)

    # Match the established deterministic type-stratified split policy.
    split_seed = int(json.loads((EXAMPLE / MANIFEST_FILENAME).read_text(encoding="utf-8"))["resolved_config"]["split_seed"])
    assignment = assign_splits(
        [{"sample_id": value["sample_id"], "generator_type": value["generator_type"]} for value in clones],
        train_size=0.6,
        valid_size=0.2,
        test_size=0.2,
        seed=split_seed,
    )
    for value in clones:
        value["split"] = assignment[value["sample_id"]]
    records = [DatasetRecord.from_dict(value) for value in clones]
    write_records(records, destination / SAMPLES_FILENAME)
    for split in SPLIT_NAMES:
        write_split_manifest(
            (record for record in records if record.split == split),
            destination / SPLIT_DIR / f"{split}.json",
        )
    (destination / AUDIT_DIR).mkdir(parents=True, exist_ok=True)
    for relative, rows, fields in (
        (f"{AUDIT_DIR}/generator_master.csv", raw_clones, raw_fields),
        (f"{AUDIT_DIR}/generator_master_with_source_labels.csv", labeled_clones, labeled_fields),
    ):
        with (destination / relative).open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    manifest = json.loads((EXAMPLE / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    manifest["benchmark_fixture"] = {
        "kind": "synthetic_scaled_copy",
        "source_example_dataset_id": manifest["dataset_id"],
        "description": "Rows and paths are duplicated from the 48-image example for performance analysis; not Pizzaïolo generation.",
    }
    manifest["sample_record_count"] = sample_count
    manifest["sample_ids"] = [record.sample_id for record in records]
    config = manifest["resolved_config"]
    config["total_instances"] = sample_count
    config["pizza_type_counts"] = dict(sorted(Counter(record.generator_type for record in records).items()))
    manifest["split_policy"]["sizes"] = {
        split: sum(record.split == split for record in records) for split in SPLIT_NAMES
    }
    # Use the actual class vocabulary (the manifest stores its path/hash metadata).
    classes = json.loads((destination / f"{VOCABULARY_DIR}/ontology_classes.json").read_text(encoding="utf-8"))["classes"]
    manifest["statistics"] = build_statistics(records, classes)
    manifest["asset_checksums"] = _asset_checksums(destination)
    identity = {
        "schema_version": manifest["schema_version"],
        "resolved_config": config,
        "generator_commit": manifest["source"]["generator_commit"],
        "ontology_sha256": manifest["ontology"]["sha256"],
        "asset_checksums": manifest["asset_checksums"],
    }
    manifest["dataset_id"] = "pizza-" + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()[:20]
    manifest["manifest_id"] = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in manifest.items() if key != "manifest_id"})
    ).hexdigest()
    _write_json(destination / MANIFEST_FILENAME, manifest)
    return destination


def _timed(call: Callable[[], object]) -> float:
    started = time.perf_counter()
    call()
    return time.perf_counter() - started


def _repeated(call: Callable[[], object], iterations: int) -> dict[str, object]:
    values = [_timed(call) for _ in range(iterations)]
    return {"seconds": values, "median_seconds": statistics.median(values)}


def benchmark_dataset(root: Path, iterations: int) -> dict[str, object]:
    _invalidate_dataset(root.resolve())
    first_inspection = _timed(lambda: inspect_dataset(root))
    repeated_inspection = _repeated(lambda: inspect_dataset(root), iterations)

    _invalidate_dataset(root.resolve())
    first_explorer = _timed(lambda: load_explorer(root))
    explorer = load_explorer(root)
    repeated_explorer = _repeated(lambda: load_explorer(root), iterations)
    def select_sample_image():
        current = load_explorer(root)
        return load_verified_sample_image(current, current.records[0])

    selection = _repeated(select_sample_image, iterations)

    # Corrupt only the disposable JSONL, then time watcher-driven invalidation
    # and full revalidation. The resulting current_fail is expected.
    samples = root / SAMPLES_FILENAME
    with samples.open("ab") as file:
        file.write(b"\n")
    invalidation = _timed(lambda: inspect_dataset(root))
    invalidated_status = inspect_dataset(root).integrity_status

    return {
        "fixture": str(root),
        "samples": explorer.summary.sample_count,
        "file_count": sum(path.is_file() for path in root.rglob("*")),
        "logical_file_bytes": sum(path.stat().st_size for path in root.rglob("*") if path.is_file()),
        "initial_inspection_seconds": first_inspection,
        "repeated_unchanged_inspection": repeated_inspection,
        "initial_explorer_load_seconds_cold": first_explorer,
        "repeated_unchanged_explorer_load": repeated_explorer,
        "typical_image_selection_interaction": selection,
        "relevant_change_invalidation_seconds": invalidation,
        "invalidation_status": invalidated_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")

    with tempfile.TemporaryDirectory(prefix="town-phase1-cache-bench-") as temporary:
        temp = Path(temporary)
        small = temp / "pizza-48-copy"
        shutil.copytree(EXAMPLE, small)
        large = build_scaled_fixture(temp / "pizza-synthetic-4800")
        results = {
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "filesystem_mount": str(os.statvfs(temp).f_fsid),
                "note": "Synthetic scale fixture is not generator output; 48-image case is a disposable byte-copy of examples/pizza-48.",
            },
            "small_48": benchmark_dataset(small, args.iterations),
            "synthetic_4800": benchmark_dataset(large, args.iterations),
        }
        print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
