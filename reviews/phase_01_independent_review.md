# Independent Phase 1 Specification and Technical Review

**Project:** *Representing Ontologies Using Town Embeddings*
**Review date:** 2026-10-28
**Verdict:** **REQUEST CHANGES**
**Reviewed implementation handoff SHA:** `99956608ac88fd5ab3011d28d57937c7b5e32983`
**Checkout reviewed:** `220318f914668cba914b504596dda49a298e006f` (`main`, also `origin/main` at review time)

This was an independent review of the delivered Phase 1 data-preparation subsystem, its dataset contract, CLI, validation/reporting, and local Streamlit dashboard. It did not modify application code or begin Phase 2. The implementation source, tests, package metadata, README, schema, and provenance docs in the checkout are identical to those at the handoff SHA; the checkout additionally contains the implementation report and prepared example dataset. The named handoff SHA exists in the local object database but is not contained by a local or remote branch/tag, which is recorded as finding **P1-M2** below. The review began from a clean checkout.

The accepted Phase 1 implementation prompt was not found among the searched project files/locations. The review therefore cross-checked the supplied review brief against the committed v1.1 expanded specification, schema, README, provenance, implementation report, tests, and artifacts. This limits direct prompt-to-code traceability, but does not prevent the findings below from being tested against the committed acceptance contract.

## A. Executive verdict

**REQUEST CHANGES.** The Phase 1 implementation is substantial and broadly well-bounded: it installs in a clean environment, regenerates and validates the small pinned-source example, maintains the ontology/class/individual distinction, preserves the audited image-label behavior, provides stable per-occurrence component records, and exposes useful CLI and dashboard workflows. The 37-test suite passes. The principal Phase 1 acceptance blockers are:

1. **One critical semantic input-validation defect:** malformed origin sentinels such as the literal string `"NaN"` are normalized to “no origin,” then treated as a complete known-negative synthetic fact. This contradicts the explicit rule that malformed/incomplete metadata must fail rather than silently become negative.
2. **One major reporting/inspection defect:** inspect and dashboard loading trust manifest statistics and saved validation status without checking the current records/assets. A changed sample can therefore be browsed alongside stale support numbers and a displayed `PASS`; the explicit validator does detect the same corruption.
3. **Two moderate completeness/reproducibility gaps:** contour validation checks little beyond JSON readability/category names, and the handoff revision recorded in the example/report is not reachable from any published ref in the reviewed checkout.

**Finding counts:** 1 Critical, 1 Major, 2 Moderate, 0 Minor (**4 total**). These findings do not invalidate the normal 48-image output generated in this review; they show gaps in the stated validation and traceability contract that should be corrected before Phase 2 relies on these artifacts.

## B. Review basis and environment

### Revision and provenance

- Project remote: `midhunpradeep/town_model` (`origin` uses the repository's GitHub SSH URL).
- Handoff SHA named in `docs/PHASE_1_REPORT.md` and the example manifest: `99956608ac88fd5ab3011d28d57937c7b5e32983`.
- Checkout `HEAD`: `220318f914668cba914b504596dda49a298e006f`, clean at review start and equal to `origin/main`.
- A path-restricted Git comparison confirmed that `src/`, `tests/`, `pyproject.toml`, `README.md`, the schema/provenance docs, and the expanded specification are identical at the handoff SHA and checkout `HEAD`. The checkout adds the report/example artifacts. The handoff commit object is present locally, but no branch or tag contains it; a fresh clone cannot resolve that provenance SHA unless it is published or replaced.
- The upstream Pizzaïolo generator was checked out clean at pinned commit `6c0bcb221a957c184012beaf103bebcde4b1f793`. Review-generated data and the clean virtual environment were kept under `/tmp`; no application files were changed.

### Runtime used

A fresh virtual environment installed the project and test extras from the checkout. Versions resolved during review:

| Component | Version |
| --- | ---: |
| Python | 3.12.0 |
| Streamlit | 1.66.1 |
| pytest | 8.4.2 |
| pandas | 3.1.0 |
| NumPy | 2.5.4 |
| OpenCV headless | 4.14.0.94 |
| Owlready2 | 0.51 |
| Pillow | 12.3.0 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| PyTorch | Not installed (not required for this phase) |

`pip check` reported no broken requirements. Dependency ranges are intentionally broad and there is no lock file; the environment above records what this review resolved, not a pinned project environment.

## C. Specification compliance matrix

Statuses refer to the Phase 1 scope, not the expressly deferred detector/training milestones.

| Requirement / acceptance surface | Status | Independent evidence and qualification |
| --- | --- | --- |
| **Phase boundary:** model-independent dataset prep, CLI, first local dashboard; no later geometry/training implementation | **PASS** | Package is organized under `src/town_model/`; no PyTorch dependency or model code was introduced. README and `docs/PHASE_1_REPORT.md` identify later work as deferred. |
| **Reuse/adaptation of audited generator-label and split behavior** | **PASS** | Compared image-label inference on all 48 generated example rows with `fourth_attempt`: 0 mismatches. Reviewed the preserved 32-label rules and two-stage type-stratified split implementation/tests. The predecessor is not imported at runtime. |
| **Pinned source, isolation, attribution, and no upstream edits** | **PASS** (traceability qualification below) | Worker verifies repository origin, exact commit, and clean status; it isolates process-wide upstream behavior and applies a local deterministic adapter. A generated dataset retained upstream provenance/licensing. The pinned source remained clean after generation. |
| **Complete ontology inventory and class/individual separation** | **PASS** | Independent inventory: 66 OWL classes and 5 named individuals. Vocabulary distinguishes 60 trainable classes, fixed-positive `DomainThing`, and 5 unsupported value-partition classes. `Germany` remains an individual and is excluded from detector categories; `HotTopping` is not promoted to an OWL class. |
| **Image-label semantics and target ancestry** | **PASS** for observed source rows | All 48 example rows matched the predecessor's inference. Reviewed target expansion and verified required example targets/ancestors, including class-only output, flag categories, and fixed-positive semantics. Unsupported classes retain unknown masks. |
| **Unknown, negative, and malformed metadata contract** | **FAIL — P1-C1** | A genuinely blank origin is a known absence for this complete synthetic source, as documented. However, `_normalize_optional` also maps literal `"None"`, `"none"`, `"nan"`, and `"NaN"` to `None`, so malformed/nonempty source values can take the same closed-world path instead of being rejected. |
| **Sample schema, IDs, coordinates, image dimensions, boxes, and vocabulary references** | **PASS** for generated records | The sample JSON Schema passes Draft 2020-12 schema checking for a saved record; test suite covers serialization/round-trips. Records preserve generator refs and stable category/occurrence IDs; validation checks bbox ranges, dimensions, target vector lengths, categories, and ontology targets. |
| **Preserved source annotation integrity, including contours/segmentation** | **PARTIAL — P1-M1** | Image, box, and sample relations receive substantive checks. Contour JSON is parsed and unsupported top-level category names are rejected, but per-category presence/count/shape/coordinates are not cross-checked; segmentation is effectively checked for nonempty existence. A structurally empty contour document with coherent asset/manifest hashes still passes validation (reproduced). |
| **Splits, manifest identity, deterministic generation, support statistics** | **PASS** for generated/explicitly validated outputs; **FAIL** for unvalidated inspection — P1-Major | Independently generated 48 records split 16/16/16; repeat generation yielded identical dataset ID and immutable asset checksums. Validator recomputes support statistics and catches mismatches. `inspect_dataset` and dashboard loading instead trust manifest statistics and saved report without verifying current records/assets. |
| **CLI setup, generation, validation, inspect, no-overwrite policy** | **PASS** | Clean editable install; 48-image generation/validation; a 3-image CLI run produced one sample per split and validated; existing output path was rejected. Commands and results are logged in §E. |
| **Dashboard launch, browse, labels, per-occurrence identity, generation/status** | **PASS** via automated verification | Local health endpoint returned `ok`. Streamlit `AppTest` loaded the real 48-image example and rendered an image element, ontology targets, and a 14-row occurrence table; AppTest also completed explicit 3-image generation. The review did not claim manual browser/GUI interaction. |
| **Dashboard support counts and “latest validation” trustworthy for current artifacts** | **FAIL — P1-Major** | Overview derives the support table and `PASS` banner from saved manifest/report. Explorer loads JSONL records but does not run the integrity gate. A changed target with stale manifest/report was browseable and retained stale support numbers; CLI validation detected the discrepancy. |
| **Provenance SHA and independent reproduction from a fresh clone** | **PARTIAL — P1-M2** | The handoff report and example manifest cite SHA `999566…`, whose commit object is not reachable from a branch/tag in the reviewed repository. The checkout source matches it exactly, but that does not make the SHA fetchable by a future reviewer. |
| **Accepted Phase 1 implementation prompt traceability** | **NOT VERIFIED** | The prompt was not found in the searched project locations. The committed v1.1 specification and repository artifacts were used as the available contract. |

## D. Findings

### P1-C1 — Critical: malformed origin sentinels silently become known negatives

**Evidence:** `src/town_model/dataset/preparation.py:94–100` converts the strings `"None"`, `"none"`, `"nan"`, and `"NaN"` to Python `None`. The generator CSV is read with empty-field preservation, so a true blank can be distinguished from a nonempty literal sentinel. Preparation then uses `origin = _normalize_optional(...)` and computes expected rendered categories from that normalized value (`preparation.py:169–212`). Validation accepts `None` as a valid no-origin state (`validation.py:418–420`), and the closed-world masks make supported origin-dependent labels known negatives.

**Reproduction:** Calling `_normalize_optional("NaN")` returns `None`. A source row containing the literal string is therefore normalized indistinguishably from a genuinely blank origin; absent flag boxes are accepted as consistent with “no flag.” The origin-dependent predicates evaluate false and the record can be labeled as known negative rather than rejected. The committed label test covers `origin=None`, but not malformed literal sentinels.

**Why it matters:** `docs/DATASET_SCHEMA.md:104–106` explicitly limits the closed-world interpretation to complete generated rows and says malformed/incomplete metadata is an error, not a negative label. A missing-value convention from another CSV pipeline can silently fabricate false negatives and contaminate image-level ontology supervision.

**Required correction:** Preserve only the source's documented true-blank representation as absence. Reject nonempty sentinel strings (and any unsupported origin) before label inference/box comparison. Add tests proving an actual blank remains valid while nonempty sentinel strings (including `"NaN"` and `"None"`), unsupported `Germany`, and flag/metadata contradictions fail; define whitespace handling from the pinned source contract. If the upstream format legitimately has another null token, define it explicitly and verify against the pinned generator rather than normalizing arbitrary text.

### P1-Major — Major: inspect/dashboard can present stale support and a stale PASS for modified data

**Evidence:** `src/town_model/application.py:106–127` builds `DatasetSummary` from `manifest.json` and the saved validation report; it does not validate current sample/assets or recompute support. `load_explorer` then reads records (`application.py:130–143`). `src/town_model/dashboard/data.py:18–22` only catches load/parse errors. The overview renders the saved validation status and support table directly from the summary (`dashboard/app.py:62–70, 87–107`); the explorer uses the loaded records without the integrity gate (`app.py:127–130`). By contrast, `dataset/validation.py:638–647` correctly rebuilds class-support statistics from records and reports `statistics_mismatch`.

**Reproduction:** Changed a sample target in `samples.jsonl` without updating the manifest. `town inspect`/dashboard loading still succeeded, reported the manifest's unchanged support counts and saved validation `PASS`, and the explorer rendered the sample. Running the explicit validator on the same directory detected the checksum/target/statistics corruption. Thus the dashboard can display values that disagree with the records it is showing.

**Why it matters:** Per-class, per-split positive/negative/unknown support and visible integrity status are important Phase 1 outputs, not cosmetic details. Stale reporting undermines the inspection gate and can lead the user to trust a dataset that the explicit validator rejects.

**Required correction:** Make inspection/loading report an integrity state for the current artifact set before presenting a saved `PASS`, or recompute displayed support from loaded records and explicitly mark it unvalidated/stale until validation succeeds. Do not silently display saved support as current after asset/sample changes. Add a test that mutates a target while preserving the saved manifest/report and asserts the dashboard cannot show stale support plus `PASS`.

### P1-M1 — Moderate: contours/segmentation do not receive the specified structural consistency checks

**Evidence:** `src/town_model/dataset/validation.py:508–524` parses contour JSON and rejects unsupported category keys, but does not verify that contour category keys match the box/component categories, that occurrence counts correspond, or that point arrays are structurally valid and within image dimensions. For segmentation, the check shown only flags an existing zero-byte file. The documented integrity gate calls for annotation references and related artifacts to be valid and mutually consistent (`docs/DATASET_SCHEMA.md:3, 108–117`; expanded spec §10.1).

**Reproduction:** On a disposable copy of the generated 48-image example, replaced one contour document with `{}`, updated that asset's SHA-256 and the dependent dataset/manifest IDs consistently, and ran `validate_dataset`. The result was `valid=True` with no errors: the file is valid JSON and has no unsupported category key, but no contours remain. This isolates the missing semantic check from the general stale-checksum check.

**Why it matters:** The current box-based Phase 1 target path is well-validated, but contours and segmentation are part of the preserved dataset contract and can later be consumed by annotation tooling or model adapters. A structurally incomplete generated artifact can be declared valid.

**Required correction:** Define the minimum supported contour/segmentation structure for this pinned source and validate it (at minimum, contour categories and per-category occurrence counts against boxes/components, valid nonempty point sequences with in-bounds coordinates, and segmentation dimensions/encoding where the upstream format makes those invariants available). Add focused malformed/missing/inconsistent asset tests. Avoid claiming checks the source format cannot substantiate; document any intentionally opaque payload checks separately.

### P1-M2 — Moderate: recorded implementation commit is not reachable from repository refs

**Evidence:** `docs/PHASE_1_REPORT.md:7` and `examples/pizza-48/manifest.json` record preparation commit `99956608ac88fd5ab3011d28d57937c7b5e32983`. In the reviewed clone, that object exists locally but no branch or tag contains it; `main`/`origin/main` point to `220318f914668cba914b504596dda49a298e006f`. The maintained source and tests compare identical across those commits, which reduces uncertainty about what was reviewed, but a fresh clone cannot check out the recorded preparation revision.

**Why it matters:** The example is explicitly a reproducibility artifact and the manifest claims an exact preparation revision. A non-fetchable revision weakens independent auditability and provenance.

**Required correction:** Publish a ref containing the handoff commit (tag/branch) or regenerate/update the example/report so they name a reachable source commit. Ensure the recorded SHA is the exact tree used to create the artifact and is fetchable from a clean clone; do not merely change the string without verifying the artifact provenance.

## E. Verification log

Commands below were run from the repository root unless noted. Full generated datasets, source checkout, and the clean venv were disposable `/tmp` artifacts. For the CLI runs, `REVIEW_TMP=/tmp/town_phase1_review.eHq61p` and `PIZZAIOLO_SOURCE_DIR=/home/midhun/Projects/ovgu/sem8/thesis/baseline_model/pizzaiolo` (the latter was verified clean and pinned):

```bash
town generate --config dataset.toml --output "$REVIEW_TMP/pizza-48" --source-repo "$PIZZAIOLO_SOURCE_DIR"
town validate --json "$REVIEW_TMP/pizza-48"
town inspect --json "$REVIEW_TMP/pizza-48"
town generate --config "$REVIEW_TMP/tiny.toml" --output "$REVIEW_TMP/cli-tiny" --source-repo "$PIZZAIOLO_SOURCE_DIR"
town validate --json "$REVIEW_TMP/cli-tiny"
town dashboard --headless --address 127.0.0.1 --port 18517
```

| Verification | Result |
| --- | --- |
| `python -m venv <review-tmp>/venv` then `<venv>/bin/python -m pip install -e '.[test]'` | **PASS**, fresh environment installed project/test extras. |
| `<venv>/bin/python -m pip check` | **PASS**, “No broken requirements found.” |
| `<venv>/bin/python -m pytest -q` | **PASS: 37 passed**. Two pandas deprecation warnings occurred; no failures or skips. The repository's normal environment also passed all 37 tests. |
| `<venv>/bin/python -m compileall -q src tests` | **PASS**. |
| Draft 2020-12 JSON Schema check (`Draft202012Validator.check_schema` plus one saved JSONL record) | **PASS**, schema valid and sample produced zero schema errors. |
| Independent generation from the pinned clean upstream checkout, 48 records | **PASS**, generated all 16 configured type strata (three each across the configured types), 16/16/16 split sizes. |
| Validation of that 48-image output | **PASS nominally**, zero errors and 178 low-support warnings; no warnings were suppressed. |
| Repeat 48-image generation with identical config/seeds | **PASS**, same dataset ID and all immutable asset checksums. |
| Compare 48 image-label inferences to `fourth_attempt` | **PASS**, zero mismatches. |
| CLI 3-image generation with one selected generator type | **PASS**, 3 records, one per split; subsequent validation returned zero errors. |
| CLI generation to an existing output directory | **PASS**, rejected rather than overwriting. |
| Start dashboard server and request `/_stcore/health` | **PASS**, endpoint returned `ok`. |
| Streamlit `AppTest` on the saved 48-image example | **PASS**, no app exceptions; selected a real image and rendered an image element, corrected ontology-target section, and component-occurrence table (14 rows for the selected image). |
| Streamlit `AppTest` explicit generation, 3 images | **PASS**, generated through the dashboard workflow and completed successfully. |
| Corrupt one sample target without updating manifest, then inspect/browse vs validate | **Finding P1-Major reproduced**: inspect/browse succeeded with stale values; explicit validator reported corruption. |
| Replace one contour document with `{}` and update coherent checksums/IDs | **Finding P1-M1 reproduced**: validator returned valid with no errors. |
| Review malformed-origin normalization | **Finding P1-C1 reproduced**: literal `"NaN"` normalizes to no origin; committed tests do not cover it. |
| Upstream checkout after generation | **PASS**, pinned source remained clean and unmodified. |

The 178 warnings on the 48-image demonstration are expected low-support warnings and are not treated as defects. No claim is made that 48 images constitute adequate model-training data. No 4,800-image run or model training was attempted; later-phase model work was out of scope and the brief explicitly says not to depend on it. A graphical browser/interactive manual GUI pass was not available; dashboard evidence is the automated Streamlit AppTest and HTTP health check only.

## F. Architecture and extension assessment

The main Phase 1 boundaries are appropriate and make later work feasible without importing training concerns into dataset generation:

- `config.py` owns validated settings; `application.py` exposes common operations to CLI and dashboard; `cli.py` and `dashboard/` are adapters.
- The dataset package separates ontology inventory/target rules, source generation, preparation, schema, statistics, splitting, and validation. The pinned upstream source runs in a subprocess, limiting its process-wide working-directory and RNG effects.
- Serialized samples and vocabularies remain framework-neutral. Stable class IDs are tied to the ontology snapshot/hash; detector category IDs remain separate from ontology class IDs; occurrence-level IDs preserve repeated instances.
- The saved validation artifact has useful structured diagnostics and recomputes supervision statistics. The repeat-generation exercise supports the deterministic-generation claim for the tested environment/config.

The main extension risk is that `inspect_dataset` and dashboard loading are currently presentation paths over trusted saved metadata rather than integrity-gated dataset reads. That weakens the otherwise sound data-contract boundary. Before model loaders or larger datasets depend on it, current artifact integrity and displayed support should have one explicit, shared trust/status policy. Contour/segmentation structural guarantees and the reachable preparation revision should also be finalized at that boundary.

The dashboard is sufficient as a first local research interface: it has explicit generation, settings validation, status/error handling, provenance, split-aware support, image browsing, and selectable component occurrences. It is intentionally a single Streamlit app and a one-worker local generation operation; that is acceptable for Phase 1. It does not yet establish any future detector/model API, nor should it be treated as doing so.

## G. Deferred work versus Phase 1 omissions

### Correctly deferred; not findings

- Object detector/model loading or model framework integration.
- Town embeddings, geometry, loss functions, training, evaluation, experiment comparison, or a full model-data loader.
- GPU/PyTorch/TorchVision compatibility and performance.
- Adequate training support from the small demonstration dataset.

These are later-phase items and were not exercised. The Phase 1 package correctly avoids importing PyTorch; its absence from the clean environment is intentional.

### Phase 1 corrections before acceptance

- Reject malformed origin sentinels instead of converting them into complete no-origin facts (**P1-C1**).
- Keep displayed support/validation status consistent with the actual dataset files (**P1-Major**).
- Validate the structure and cross-file consistency of preserved contour/segmentation data to the extent promised by the contract (**P1-M1**).
- Make the preparation revision in the manifest/report reachable and reproducible (**P1-M2**).

## H. Phase 1 acceptance gate

| Gate | Review result |
| --- | --- |
| Fresh install and dependency health | **PASS** |
| Generate configured small dataset from pinned source | **PASS** |
| Validate generated dataset | **PASS for unmodified normal output**, but semantic edge cases remain (**P1-C1**, **P1-M1**) |
| Inspect and report current per-class/per-split support | **PARTIAL**; values are correct for untouched generated output, but stale artifacts can still be displayed as current (**P1-Major**) |
| Launch local dashboard | **PASS** via health endpoint |
| Browse generated images/boxes/ontology labels/occurrences | **PASS** via automated AppTest; manual browser interaction **not verified** |
| Deterministic repeat generation and immutable output policy | **PASS** in reviewed configuration |
| Provenance independently resolvable from fresh clone | **FAIL/PARTIAL** until SHA is published or artifacts are updated (**P1-M2**) |
| No unapproved later-phase features | **PASS** |

**Acceptance recommendation:** Do not mark Phase 1 accepted or begin Phase 2 until P1-C1 and P1-Major are corrected and regression-tested. Close P1-M1 and P1-M2 as part of the Phase 1 data-contract/provenance handoff before downstream phases rely on those guarantees. Re-run the clean-install, generation, validation, CLI, and dashboard smoke checks after changes. The review made no application-code changes.
