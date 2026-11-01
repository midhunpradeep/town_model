# Representing Ontologies Using Town Embeddings

The repository has independently accepted historical Phase 1 (Pizzaïolo dataset preparation), Phase 2 (synthetic geometry/classification), and Phase 3 (**pre-detector, GT-ROI image-to-ontology**) milestones. The later T0–T5 classifier teardown is also accepted and defines the current execution boundary: this is a **classifier-free reconstruction foundation**, not an operational neural/box/town classifier system. Historical acceptance records remain evidence for their original scopes; they do not make removed classifier behavior executable or establish scientific conclusions.

**Current teardown acceptance:** [Definitive T5 classifier teardown review](reviews/classifier_teardown_t5_final_acceptance.md) accepts teardown implementation `19222424f5ad8347deddc6dfb2d7187bd85609c0`; its final review/report commit is `2b9df576aac2fd95f093bbb3b046813620a822a5`. Phase 1–3 acceptance records are separately indexed in [ACCEPTANCE_INDEX.md](docs/ACCEPTANCE_INDEX.md). The [original canonical A1 scientific ledger](docs/reconstruction/A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md) contains 70 active approvals for F0–F10; its identity and decision locations are in the [completeness audit](docs/reconstruction/A1_DECISION_COMPLETENESS_AUDIT.md). A separate [F11 scientific addendum](docs/reconstruction/A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md) records twelve author-reported post-R1 approvals, bringing the source-level frontier to 82 before the separately reported F12 approvals. R1 was independently accepted and integrated at `100d04d4ecae4f58f525bbdac76b87bf28026d5c`; its review did not cover F11 or F12. F11 was independently accepted and integrated under [issue #12](https://github.com/midhunpradeep/town_model/issues/12) at `a050234ae490d650e28391f180670d1c38183192`. The author-reported [F12 scientific addendum](docs/reconstruction/A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md) records twelve further approvals, bringing the source-level frontier to 94; F12.1–F12.8 are a grouped block without per-ID formula mapping. Its repository revision is tracked under [issue #13](https://github.com/midhunpradeep/town_model/issues/13) and awaits independent review/integration. No classifier reconstruction, detector, or later scientific training stage has been implemented. Phase 4/detector work remains on hold and requires separately scoped authorization. See [current project status](docs/CURRENT_PROJECT_STATUS.md) and the [implementation-readiness frontier](docs/reconstruction/A1_IMPLEMENTATION_READINESS.md).

## Current classifier-reconstruction boundary

Historical classifier execution remains intentionally disconnected after accepted T0–T5 teardown. Synthetic/image training, evaluation, resume, preflight, geometry interpretation/export, relation diagnosis, background workers, and classifier-dependent dashboard actions fail closed with `ClassifierBackendUnavailableError` before model/run side effects. No replacement backend exists. Dataset preparation/validation/inspection, classifier-free vision/ROI and point infrastructure, and read-only historical metadata/snapshot loading remain available. The T5 verdict is final for teardown scope; the historical Phase 1–3 reports do not describe current executable model commands. See the [T5 review](reviews/classifier_teardown_t5_final_acceptance.md), [acceptance index](docs/ACCEPTANCE_INDEX.md), and [T0–T5 teardown map](docs/reconstruction/CLASSIFIER_TEARDOWN_MAP.md) (historical planning artifact).

## Install and run

Python 3.11–3.12 and Git are required. For a CPU-only install, install the PyTorch CPU wheel first, then the project and tests:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.5.1+cpu'
python -m pip install -e '.[test]'
```

The image adapter is pinned to PyTorch 2.5.1 and Torchvision 0.20.1. Use matching wheels for your platform; CUDA execution has not yet been verified in this implementation.

Generation is explicit. The sample config requests a modest 48-image dataset (three images per generator type):

```bash
town generate --config dataset.toml --output artifacts/datasets/pizza-48
town validate artifacts/datasets/pizza-48
town inspect artifacts/datasets/pizza-48
town dashboard
```

The first generation downloads the pinned upstream source into the ignored `artifacts/sources/` cache. It is never patched. Set `PIZZAIOLO_SOURCE_DIR` to an already available checkout at the pinned commit to avoid cloning. Output paths must not already exist; datasets are published atomically and never replaced in place. To rerun, choose a new output path.

See [Dataset schema](docs/DATASET_SCHEMA.md), [provenance and licensing](docs/PROVENANCE.md), and the [Phase 1 implementation report](docs/PHASE_1_REPORT.md). A saved small real-generator demonstration is in [`examples/pizza-48`](examples/pizza-48).

## Phase 2 synthetic classifier experiments — historical milestone

Phase 2's neural-baseline, box and town implementations and their executable mathematics have been removed from active source. Their historical reports and acceptance evidence remain; they are not retained model backends. `town experiment fixed|mapping`, `town evaluate`, `town resume`, and `town export-geometry` remain visible for compatibility and return an explicit unavailable-operation error without creating or changing artifacts. The Geometry Laboratory likewise presents an unavailable state. See [Phase 2 scope](docs/PHASE_2_SCOPE.md) and the [historical implementation report](docs/PHASE_2_IMPLEMENTATION_REPORT.md); these describe the accepted milestone at that time, not current classifier execution.

## Phase 3 image-to-ontology embeddings — historical accepted milestone

The Phase 3 classifier implementations and their executable mathematics have been removed; the accepted pre-detector GT-ROI milestone remains historical evidence. Classifier-independent dataset, RGB preprocessing, MobileNetV3-Large/FPN, ROI/ROIAlign, visual feature extraction/cache, `ImagePointModel`, point composition, read-only run metadata, and checksummed historical snapshot loading remain available. Image classifier preflight, run, matrix, resume, checkpoint evaluation, worker launch/stop, snapshot generation, and geometry/relation interpretation remain unavailable and fail closed. The Phase 3 system did not implement or train an image detector. Detector pretraining, predicted-ROI learning, joint training, scientific comparisons, and final held-out evaluation remain unimplemented. See [Phase 3 scope](docs/PHASE_3_SCOPE.md), [architecture](docs/PHASE_3_ARCHITECTURE.md), the [historical implementation report](docs/PHASE_3_IMPLEMENTATION_REPORT.md), and the [definitive Phase 3 acceptance review](reviews/phase_03_definitive_acceptance_review.md).

## Tests

```bash
pytest
```

## Scope

Phase 1–3 remain accepted historical implementation milestones; T0–T5 teardown is the current classifier-execution boundary. Acceptance does not turn exploratory results into scientific conclusions. The original A1 ledger (70 F0–F10 approvals), separate accepted/integrated F11 addendum (twelve author-reported approvals), author-reported F12 addendum (twelve further approvals, repository review pending), and derived [readiness frontier](docs/reconstruction/A1_IMPLEMENTATION_READINESS.md) distinguish authority, open details, and planning candidates. The source-level frontier is 94; F12.1–F12.8 are recorded as a grouped block without an invented per-ID mapping. R1 is independently accepted at `100d04d4ecae4f58f525bbdac76b87bf28026d5c`; its review did not cover F11 or F12. F11 is independently accepted/integrated at `a050234ae490d650e28391f180670d1c38183192`; F12 review/integration is tracked separately under #13. Later detector/scientific training is not implemented and remains on hold. See [current project status](docs/CURRENT_PROJECT_STATUS.md), [acceptance index](docs/ACCEPTANCE_INDEX.md), [T5 teardown review](reviews/classifier_teardown_t5_final_acceptance.md), and [Phase 3 scope](docs/PHASE_3_SCOPE.md).
