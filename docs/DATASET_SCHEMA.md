# Prepared dataset wire schema (version 1.0.0)

This is the public Phase 1 data contract. A dataset is usable with the Python standard library plus an image reader; loading samples does not import the generator, an OWL reasoner, PyTorch, Torchvision, or Streamlit. The machine-readable sample schema is [`schemas/sample-v1.schema.json`](../schemas/sample-v1.schema.json); cross-record, ontology, file-reference, and closed-world checks are enforced by the validator.

## Files and serialization

```text
<dataset>/
  manifest.json
  samples.jsonl
  ontology/pizzaiolo.xml
  vocabularies/ontology_classes.json
  vocabularies/detector_categories.json
  images/<generator-ref>.png
  labels/<generator-ref>_bboxes.json
  labels/<generator-ref>_contours.json
  labels/<generator-ref>_segmentation.txt
  labels/concepts.json
  splits/{train,validation,test}.json
  audit/generator_master.csv
  audit/generator_master_with_source_labels.csv
  reports/validation.json
```

- `samples.jsonl` is UTF-8, one JSON object per line, with deterministic key ordering. It is the sample interface; generated images, annotation files, and generator metadata facts are retained as assets/audit information. The set-valued `hasTopping` list is canonically sorted as documented in provenance.
- JSON manifests/vocabularies/split files use UTF-8 JSON with two-space indentation. Dataset-relative paths are POSIX paths and may not escape the dataset root.
- In the pinned generator CSV, an absent `hasCountryOfOrigin` is an empty field: `Pizzaiolo.cook` appends Python `None` to the row, and pandas `to_csv` serializes it as an empty cell. Preparation reads with `keep_default_na=False`, so only an empty string (or an in-memory Python `None`) means absent. Nonempty strings such as `NaN`, `nan`, `None`, and `none`, whitespace-only values, and unsupported country names are malformed/unsupported and rejected. The generator's separate `type` column writes the literal `None` for anonymous/random recipes; that context-specific representation is handled separately and is not accepted as an absent origin.
- Images are PNG in RGB, with original width and height recorded. The pinned generator currently emits 224×224 images. Segmentation text is emitted by `numpy.savetxt` as a 224-row by 224-column unsigned-integer matrix. `labels/concepts.json` maps those generator-specific IDs (`0` is background/`None`; nonzero values name the 22 rendered detector categories); segmentation IDs are not detector category IDs.
- Original boxes are `[x,y,width,height]` from OpenCV `boundingRect`. Wire boxes use finite original-image pixel `xyxy`: `[x, y, x + width, y + height]`. They are not normalized, resized, or clipped. The coordinates express pixel-edge extents; model adapters may transform them but may not reinterpret the wire values.
- `schema_version` is semantic `major.minor.patch`, currently `1.0.0`. A reader must reject an unknown major version. The ontology snapshot and vocabulary hashes are required compatibility checks; there is no implicit class-ID migration.

## Canonical vocabularies

`ontology_classes.json` contains every named class declared in the pinned ontology (66 entries), sorted by full IRI and assigned zero-based `class_id` in that order. IDs are stable for this schema and ontology snapshot. A changed ontology receives a new hash/schema compatibility decision; consumers must not silently assume IDs survive ontology evolution.

Each class entry includes:

- `class_id`, full class `name` and `iri`, and `entity_kind: "owl_class"`;
- `trainable`, `fixed_positive`, `pizza_supervision`, `component_supervision`, and `unsupported`;
- `supervision_status`: `trainable`, `fixed_positive`, or `unsupported`.

The supported vocabulary has 61 classes: 60 trainable classes plus fixed-positive `DomainThing`. The five unsupported classes `ValuePartition`, `Spiciness`, `Hot`, `Medium`, and `Mild` remain in the inventory with unknown masks; they are not fabricated negatives. `owl:Thing`/`owl:Nothing` and named individuals are not class output rows.

`detector_categories.json` is separate. It reserves category ID 0 for background and assigns IDs 1–22 to the 16 concrete topping classes, two bases, and four rendered flag identities. `America`, `England`, `France`, and `Italy` are OWL individuals and visual detector categories, not ontology-class IDs. Each flag's ontology targets are `Country` and its ancestors (`DomainThing`). `Germany` is an ontology individual but is not a rendered detector category.

## JSONL sample record

A sample has exactly these semantic fields (arrays shown shortened):

```json
{
  "schema_version": "1.0.0",
  "sample_id": "img_00000",
  "split": "train",
  "generator_type": "American",
  "image": {
    "path": "images/img_00000.png",
    "width": 224,
    "height": 224,
    "color_space": "RGB"
  },
  "pizza": {
    "ontology_targets": [0, 1, 0],
    "ontology_known_mask": [1, 1, 1],
    "source_inferred_labels": ["DomainThing", "Food", "Pizza", "American"]
  },
  "components": [
    {
      "component_id": "img_00000/HamTopping/occurrence_000",
      "detector_category_id": 8,
      "detector_category": "HamTopping",
      "bbox_xyxy_original": [27.0, 44.0, 58.0, 77.0],
      "ontology_targets": [0, 1, 0],
      "ontology_known_mask": [1, 1, 1],
      "annotation_path": "labels/img_00000_bboxes.json"
    }
  ],
  "audit": {
    "usage": "audit_only_not_predictive_input",
    "raw_generator": {
      "source_ref": "img_00000",
      "source_type": "American",
      "generator_type": "American",
      "has_topping": ["HamTopping"],
      "topping_counts": {"HamTopping": 1},
      "has_base": "DeepPanBase",
      "country_of_origin": "America",
      "original_annotation_filenames": {}
    }
  },
  "source_annotations": {
    "boxes": "labels/img_00000_bboxes.json",
    "contours": "labels/img_00000_contours.json",
    "segmentation": "labels/img_00000_segmentation.txt"
  }
}
```

The values above are illustrative. Both target arrays have 66 integer `0/1` entries in ascending `class_id` order. For each known target, `1` is positive and `0` is known negative; a mask value `0` means unknown and its target value is stored as `0`. The dataset reader selects trainable IDs from the vocabulary and handles fixed-positive `DomainThing` separately. `sample_id` preserves the pinned generator `ref`; each component occurrence ID adds the detector category and its zero-based, three-digit occurrence index in source bounding-box order. Repeated same-class toppings remain separate records.

`source_inferred_labels` preserves the audited source's direct image-level multilabel decisions, including all 32 possible image labels and overlap among recipe/derived classes. The target vector additionally expands their named class ancestors. Component targets add the concrete component class itself, its declared ancestors, and the established ontology-equivalent `VegetarianTopping`/`SpicyTopping` memberships where applicable. A country flag identity is retained only as detector metadata; it never becomes an invented OWL class.

## Known, negative, and unknown supervision

For a validated record produced by this pinned complete synthetic generator, ingredient counts/list, base, origin, and visible annotation facts define a deliberately closed dataset contract. An absent applicable class can therefore be a known negative. For example, a generated blank origin means no flag was selected/rendered and makes origin-dependent pizza labels negative. This is not a general claim that missing OWL assertions are false under open-world semantics.

All 61 supported classes are known at the pizza and component point levels under the generated entity-type contract; absent targets are known negatives. `DomainThing` is positive and fixed for all observed entities. The five unsupported value-partition classes have target 0/mask 0. Malformed counts, invalid bases, an unsupported origin, incomplete file references, or metadata/annotation disagreement are errors, not negative labels.

## Manifest identity and validation report

`manifest.json` records resolved generation settings, seeds, source and preparation revisions, pinned generator commit/license, ontology/class/category hashes, split policy and sizes, asset SHA-256 checksums, class-support statistics, and schema/coordinate/supervision policies.

- `dataset_id` is `pizza-` plus a SHA-256-derived digest over schema version, resolved config, generator revision, ontology hash, and all immutable asset checksums.
- `manifest_id` is SHA-256 over canonical JSON of the manifest excluding its own `manifest_id` field.
- Repeating the same deterministic generation yields the same IDs/checksums; an existing output path is rejected, never overwritten. Failed runs publish no partial dataset.
- `splits/{train,validation,test}.json` contain stable sample IDs. Splitting preserves the tested two-stage, generator-type-stratified policy (60/20/20 defaults); per-type rounding is explicit in the actual counts.
- `reports/validation.json` is a machine-readable result linked to `dataset_id`, with a stable report ID, check statuses, errors, actionable file/sample/field details, warnings, and per-class support.
- For every split, support is separated by pizza/component level and reports known positives, known negatives, and unknowns. Missing positive support is a warning in this data-only phase; it is not silently fixed by altering splits.

The report file is a derived audit artifact and is excluded from immutable asset hashes. Revalidating does not change the dataset ID or immutable data files.

## Preserved annotation integrity and inspection trust

The validator follows the pinned Pizzaïolo source representation rather than treating annotation files as generic formats:

- Bounding-box categories, contour categories, and prepared component categories must agree. The contour document is a category-to-list mapping of OpenCV contours serialized as integer `N×1×2` points. Each contour must contain at least two points and every point must lie inside the source image. For each category, the number of serialized contours must be at least its number of bounding boxes: bounding boxes count external/root contours, while contour files also preserve nested contours/holes, so equality is not required. Points are not required to trace the box edges.
- Segmentation dimensions must equal the prepared image dimensions; every used ID must exist in `concepts.json`, and each non-background ID must identify a category present in that sample's boxes/components. The validator does not require a pixel for every annotation category or require pixel/contour/box geometry to be identical: upstream masks overlap and are clipped/composited, so occlusion and overwrite semantics can remove or alter visible pixels.
- The prepared annotation files are preserved byte-for-byte. Validation is structural and cross-file; it does not reconstruct masks from contours or assert object shape quality.

`inspect` and the dashboard compute integrity against the current manifest and declared assets before publishing support statistics. A saved `reports/validation.json` is surfaced only as historical evidence, never as the current status. Successful current validation supplies the displayed statistics; failed or unavailable validation withholds those counts, and the dashboard disables image browsing. Explicit validation always runs the full integrity gate. To avoid repeating full file hashing, JSON parsing, image checks, and segmentation parsing on every Streamlit rerun, the shared inspection service caches a validation report process-locally, keyed by the manifest and stat state (device/inode, size, modification time, and change time) of every manifest-declared asset. A changed state triggers a full revalidation; each new CLI process performs its own validation. This cache is designed for ordinary artifact changes, not adversarial modification that also defeats filesystem change metadata; explicit validation remains the forced full check.
