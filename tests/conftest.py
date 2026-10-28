from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw

from town_model.config import DatasetConfig
from town_model.dataset.constants import TOPPING_COLUMNS
from town_model.dataset.ontology import ASSET_ONTOLOGY_PATH
from town_model.dataset.preparation import prepare_dataset

_PINNED_CONCEPT_NAMES = (
    "None", "DeepPanBase", "ThinAndCrispyBase", "ParmesanTopping", "SlicedTomatoTopping",
    "GreenPepperTopping", "OnionTopping", "PeperoniSausageTopping", "HamTopping",
    "PeperonataTopping", "RocketTopping", "SpinachTopping", "JalapenoPepperTopping",
    "AnchovyTopping", "MushroomTopping", "PrawnTopping", "OliveTopping", "SultanaTopping",
    "GarlicTopping", "America", "England", "France", "Italy",
)


def fake_generate_raw_dataset(
    dataset_dir: Path,
    config: DatasetConfig,
    *,
    project_root: Path,
    source_repo: Path | None = None,
    progress: Any = None,
) -> dict[str, int]:
    """Write upstream-shaped deterministic fixtures without importing the generator."""

    counts = config.allocate_type_counts()
    (dataset_dir / "images").mkdir(parents=True)
    (dataset_dir / "labels").mkdir()
    (dataset_dir / "ontology").mkdir()
    (dataset_dir / "csv").mkdir()
    shutil.copyfile(ASSET_ONTOLOGY_PATH, dataset_dir / "ontology" / "pizzaiolo.xml")

    rows = []
    image_index = 0
    toppings = list(TOPPING_COLUMNS)
    for generator_type, count in counts.items():
        for local_index in range(count):
            sample_id = f"img_{image_index:05d}"
            image_index += 1
            if generator_type == "American" and local_index == 0:
                selected_counts = {"PeperonataTopping": 1, "PeperoniSausageTopping": 1}
                base = "DeepPanBase"
                origin = "America"
            elif generator_type == "random":
                selected_counts = {toppings[local_index % len(toppings)]: 2 if local_index == 0 else 1}
                base = "ThinAndCrispyBase"
                origin = None
            else:
                selected_counts = {toppings[(image_index + local_index) % len(toppings)]: 2 if local_index == 0 else 1}
                base = "ThinAndCrispyBase"
                origin = ("England", "France", "Italy")[local_index % 3]

            counts_by_topping = {name: 0 for name in TOPPING_COLUMNS}
            counts_by_topping.update(selected_counts)
            has_topping = [name for name in TOPPING_COLUMNS if counts_by_topping[name] > 0]
            boxes: dict[str, list[list[int]]] = {}
            contours: dict[str, list[list[list[list[int]]]]] = {}
            for category in has_topping:
                quantity = counts_by_topping[category]
                boxes[category] = [
                    [12 + occurrence * 22, 18 + local_index * 4, 16, 14]
                    for occurrence in range(quantity)
                ]
                contours[category] = [
                    [[[20 + occurrence, 20]], [[21 + occurrence, 20]], [[21 + occurrence, 21]], [[20 + occurrence, 21]]]
                    for occurrence in range(quantity)
                ]
            boxes[base] = [[1, 1, 222, 222]]
            contours[base] = [[[[1, 1]], [[2, 1]], [[2, 2]], [[1, 2]]]]
            if origin:
                boxes[origin] = [[180, 12, 30, 24]]
                contours[origin] = [[[[180, 12]], [[181, 12]], [[181, 13]], [[180, 13]]]]

            with Image.new("RGB", (224, 224), (242, 226, 190)) as image:
                draw = ImageDraw.Draw(image)
                draw.ellipse((5, 5, 219, 219), fill=(213, 143, 68), outline=(120, 68, 36), width=3)
                image.save(dataset_dir / "images" / f"{sample_id}.png")
            boxes_name = f"{sample_id}_bboxes.json"
            contours_name = f"{sample_id}_contours.json"
            segmentation_name = f"{sample_id}_segmentation.txt"
            (dataset_dir / "labels" / boxes_name).write_text(json.dumps(boxes, indent=2), encoding="utf-8")
            (dataset_dir / "labels" / contours_name).write_text(json.dumps(contours, indent=2), encoding="utf-8")
            concept_ids = {name: index for index, name in enumerate(_PINNED_CONCEPT_NAMES)}
            segmentation = [[0] * 224 for _ in range(224)]
            for index, category in enumerate(boxes):
                segmentation[0][index] = concept_ids[category]
            (dataset_dir / "labels" / segmentation_name).write_text(
                "".join(" ".join(map(str, row)) + "\n" for row in segmentation), encoding="utf-8"
            )

            row = {
                "ref": sample_id,
                "img_name": f"{sample_id}.png",
                "type": "None" if generator_type == "random" else generator_type,
                "hasTopping": repr(has_topping),
                **counts_by_topping,
                "hasBase": base,
                "hasCountryOfOrigin": origin or "",
                "boxes_name": boxes_name,
                "contours_name": contours_name,
                "segmentation_name": segmentation_name,
            }
            rows.append(row)

    columns = [
        "ref", "img_name", "type", "hasTopping", *TOPPING_COLUMNS,
        "hasBase", "hasCountryOfOrigin", "boxes_name", "contours_name", "segmentation_name",
    ]
    with (dataset_dir / "csv" / "pizzaiolo_dataset.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (dataset_dir / "labels" / "concepts.json").write_text(
        json.dumps({str(index): name for index, name in enumerate(_PINNED_CONCEPT_NAMES)}, indent=2) + "\n",
        encoding="utf-8",
    )
    return {name: count for name, count in counts.items() if count}


@pytest.fixture
def small_config() -> DatasetConfig:
    return DatasetConfig(
        total_instances=6,
        pizza_types=("American", "random"),
        type_counts={"American": 3, "random": 3},
        generation_seed=7,
        split_seed=17,
    )


@pytest.fixture
def prepared_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_config: DatasetConfig) -> Path:
    from town_model.dataset import preparation

    monkeypatch.setattr(preparation, "generate_raw_dataset", fake_generate_raw_dataset)
    output = tmp_path / "prepared"
    return prepare_dataset(small_config, output, project_root=tmp_path)
