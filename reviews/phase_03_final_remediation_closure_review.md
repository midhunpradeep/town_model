# Final Independent Phase 3 Remediation Closure Review

**Project:** *Representing Ontologies Using Town Embeddings*
**Review scope:** Focused closure review of findings D1–D5 for the pre-detector, GT-ROI image-to-ontology milestone.

## A. Verdict

**REQUEST CHANGES**

D1–D4 are closed on the reviewed implementation. D5 adds a usable, mathematically careful relation interface, and its exact-box and sampled-town geometry behavior passed independent checks. Two material D5 gaps remain: a known ontology subclass entailment is labeled `unknown`, and relation-result publication is not no-clobber under concurrent writers. The latter can silently replace an immutable result. The requested changes are limited to ontology metadata and atomic no-replace result publication; no detector or Phase 4 work is requested.

## B. Reviewed revisions

- Initial independent review: `5a23bb034c57719c870a18fed51349502e65f6a0`
- Remediation implementation reviewed: `e1bdd41722c70160e56b4b1735f7ffd2fafb8684`
- Remediation documentation/handoff: `fde83a2dac28a396ffc31f1bbb611c2882505800`
- Final report-only review commit SHA: reported in the final response and by the final branch HEAD. A Git commit cannot embed its own content-derived SHA without changing that SHA.
- Branch: `remediation/phase-3-acceptance-findings`; remote and local HEAD matched at `fde83a2dac28a396ffc31f1bbb611c2882505800` before this review report was created.
- Fetched `origin/remediation/phase-3-acceptance-findings`; verified `e1bdd417` descends from `5a23bb0` and the Phase 2 baseline `fbe1063f2451270a89c9e0c110b988d6aafa4a27`. The docs commit follows the implementation commit. There were no intervening implementation changes after `e1bdd417`.
- Inspected the complete `5a23bb0..e1bdd417` diff: 24 files, 1,967 insertions and 250 deletions. The `e1bdd417..fde83a2` handoff diff changes only the two documentation files reported by Git. The previous independent report is unchanged (`git diff --quiet` returned 0). `main` was not used as a substitute.

## C. Finding closure matrix

| Finding | Closure status | Evidence |
|---|---|---|
| D1 — Scientific class weighting | **CLOSED** | `_positive_weights()` derives separate pizza/component supports from the original training subset only; uses the specified positive-count formulas, explicit all-supported-canonical-class normalization, cap, and negative weight 1. Recomputed vectors match the saved corrected runs. Loss reduction, gradients, checkpoint identity, and legacy incompatibility were verified. |
| D2 — Config preflight | **CLOSED** | Both reproduced invalid configurations now fail through shared validation before run/cache/worker publication. Direct service, CLI, dashboard, and adversarial type/range cases were exercised; the original cases leave no output/cache/reservation artifacts. |
| D3 — Worker and lock recovery | **CLOSED** | Real SIGKILL recovery reconciles run and worker state to interrupted/unknown exit, preserves the latest checkpoint byte-for-byte, and permits compatible resume. Persistent kernel locks recover malformed metadata and protect live writers. POSIX/Windows/network-FS limits are documented. |
| D4 — Snapshot axes | **CLOSED** | Schema v2 declares diagnostic axes; class geometry and per-point activity retain their semantic shapes in B=C, B≠C, one-class, multi-house/window, and empty-component cases. Reloaded arrays match source tensors exactly; v1 is rejected. Saved-town AppTest inspection passed. |
| D5 — Relation diagnostics | **OPEN** | Exact box interval decisions and sampled town witnesses/inconclusive outcomes are correct and reproducible. However, `MushroomTopping ⊑ VegetarianTopping` is known from the pinned equivalent-class axiom but reported as unknown, and concurrent result writers can both succeed and replace the supposedly immutable JSON destination. Details are in H. |

## D. Independent numerical evidence

### D1 weighting and reducer

For the corrected run's selected class order `[2 AnchovyTopping, 29 NonVegetarianPizza]`, the actual original-train subset is 128 images with **selection seed 2025** (training seed 42). Independently resolving that subset gives:

- Component: AnchovyTopping `n+ = 140`; most-supported in-scope class Food `n+ = 2,069`; `sqrt(2069/140) = 3.8442907575483187`.
- Pizza: NonVegetarianPizza `n+ = 75`; most-supported in-scope class Pizza `n+ = 128`; `sqrt(128/75) = 1.3063945294843615`.
- Resolved selected vectors: component `[3.8442907575483187, 1.0]`; pizza `[1.0, 1.3063945294843615]`.

The independently resolved values match all three saved run manifests/checkpoints. Known-positive support is distinct by supervision level; the selected training supports have zero unknowns. The normalization scans all positive-supported canonical trainable classes within each level before projecting to selected classes, so it does not depend on a selected-class subset. It uses only known-positive training counts. This is a coherent, documented common scale; the spec does not mandate an absolute scale.

Additional formula checks:

- Equal supports `[24,24]` normalize to `[1,1]`.
- Counts 24 and 1,404 have uncapped relative inverse-sqrt ratio `sqrt(1404/24) = 7.6485292704`.
- Effective-number weights for counts `[2,10,40]`, normalized to the largest support, are `[5.185363774, 1.512646253, 1]` at β=0.9 and `[16.634584996, 3.461989386, 1]` at β=0.99.
- `[1,10000]` with cap 3 gives `[3,1]`; `none` gives unit vectors; zero-support exploratory entries are finite unit weights, while comparison coverage rejects required zero-support classes. The alternative is named `experimental_inverse_frequency`; legacy `inverse_frequency` is rejected.
- Changing negative/unknown support counts leaves resolved weights unchanged; validation subsets are rejected as weight sources. Tests instrument the actual training call to the reducer and compare its vectors with checkpoint provenance.

For penalties `[2,10,4]`, targets `[1,0,0]`, known mask `[1,1,0]`, and positive weights `[9,1,100]`, independent autograd gives loss **2.8** and gradients **`[0.9, 0.1, 0]`**. Changing the unknown penalty/target does not affect either. Weighted full-batch versus `[1,2,1]` accumulated reduction gives loss **3.45555568** in both paths and maximum gradient difference **0.0**; the all-unknown row has zero gradient.

### Geometry relations

- Closed-box interval tests covered equal, nested, containment failure, partial overlap, boundary touching, strict disjointness, and a degenerate point box. Touching boxes intersect at the verified point `[0,-0.5]`; a disjoint pair produces a positive separating-axis gap; containment counterexamples are checked against both actual membership predicates.
- Town membership independently produced `[true,false,true]` for an overlapping-house case: a closed window boundary is excluded from its house and a second house re-includes the center.
- For the saved town snapshot relation command (source class 2, target 29, containment, N=100, seed 0), the result is sampled empirical with 300 candidates (100 per stated strategy), 29 observed A\B violations, and 8 returned witnesses. Every returned point was independently re-evaluated: all are in A and outside B. Repeating the query gives identical output. Ontology status for this pair is correctly `unknown`.
- Constructed town intersection returned true with 8 verified witnesses; the corresponding disjointness query returned false. A separated-town disjointness query with no sampled intersection returned `null` / `inconclusive_no_sampled_intersection`, never a proof.

## E. Operational evidence

- D2 reproductions—`ImageModelConfig(family="town", windows=0)` and fractional MobileNet block `(3.5,)`—now raise actionable `ValueError`s during shared preflight. Tests exercise direct run service, CLI preflight and launch, dashboard preflight/launch service, and no-side-effect checks. Additional probes rejected boolean integer fields, duplicate/out-of-range/fractional backbone indices, invalid town counts, impossible box/town minimum widths, invalid IDs, incompatible cache/visual policy, predicted ROIs, and unsupported precision. No run directory, manifest/checkpoint directory, feature cache, writer lock, launch reservation, or worker was created for the reproduced invalid cases.
- Forced-death subprocess test: after a valid optimizer-step checkpoint, SIGKILL led inspection to set run and worker status to `interrupted`, `exit_code=null`, and unknown exit status. The `latest.pt` SHA-256 was identical before and after recovery inspection. Compatible resume advanced the checkpoint and completed. Separate subprocess tests confirmed a live writer cannot be displaced, malformed lock JSON is recoverable, and the persistent lock inode remains usable. Recovery acquires the kernel lock before reconciling the authoritative manifest.
- Checkpoints: all corrected baseline/box/town best checkpoints reloaded and were reevaluated on validation; latest checkpoints were also reloaded and reevaluated. Historical nonconformant best artifacts remain separate and their previously reported SHA-256 values still match: baseline `33a0b273…`, box `a523d4db…`, town `28587a4c…`.
- Snapshot tests verified exact source-to-reloaded numerical equality for class geometry/activity, empty component arrays, checksum validation, and v1 rejection. The dashboard reads snapshots rather than deserializing model/optimizer checkpoints.
- CLI relation command saved `/tmp/phase3-independent-relations.json` atomically and reported snapshot-array/checkpoint provenance. The saved town relation panel was exercised through Streamlit AppTest; it rendered the download control with no exception. A separate headless Chrome 134.0.6998.88 session loaded the actual Streamlit dashboard. The relation interaction was AppTest, not a browser-click claim.

## F. Scientific evidence

Corrected run identity is dataset `pizza-898a12a57e50a7a0d690`, manifest SHA `580070e2deb301e28757bf394ecbeb31adfeae5c3eb7b41807750be75644d24d`, train/validation subset IDs `c19c4379…` / `f36b1ad7…`, 128/48 original-partition images, selected IDs `[2,29]`, hybrid mode, d=8, training seed 42, frozen ImageNet MobileNetV3-Large/FPN, no detector-pretrained checkpoint, and no test evaluation. All three runs share visual initialization hash `842a05c3…` and common neural initialization hash `ebdb6263…`. The corrected weight vectors above are present in run provenance and checkpoint resolved configs.

Independently reloaded **best-checkpoint** validation results (pizza / component GT-ROI macro-F1):

| Family | Best epoch / step | Pizza / component macro-F1 | Component / pizza / total validation loss | Best checkpoint SHA-256 prefix |
|---|---:|---:|---:|---|
| Baseline | 8 / 64 | 1.0000 / 0.9709 | 0.314746 / 0.381773 / 0.696519 | `e4f856e2…` |
| Box | 1 / 8 | 0.7368 / 0.0000 | 0.028516 / 0.159973 / 0.188489 | `6e7e34dd…` |
| Town | 7 / 56 | 1.0000 / 0.9709 | 0.003740 / 0.020957 / 0.024697 | `629c44c8…` |

Latest-checkpoint reevaluation gave baseline `0.9818/0.9709`, box `0/0`, town `1.0000/0.9709`. Best-versus-latest selection is therefore material for the box run. Its poor hard-membership F1 is honestly retained; it is not by itself an implementation defect or an acceptance criterion. These are controlled, single-seed, reduced-subset exploratory GT-ROI experiments—not final scientific comparisons or thesis claims. No test outcomes were consulted.

## G. Regression results and limits

Environment: Python 3.12.0, PyTorch 2.5.1+cpu, Torchvision 0.20.1+cpu, NumPy 2.5.4; CUDA unavailable (`torch.cuda.is_available() == False`, CUDA version `None`). No GPU execution is claimed.

Commands and observed results:

- `.venv/bin/python -m pytest -q` — **261 passed, 0 failed, 0 skipped**, 11 existing Streamlit/Pandas deprecation warnings (127.62 s).
- `.venv/bin/python -m pytest -q tests/test_phase3_weighting.py tests/test_phase3_config.py tests/test_phase3_runs.py tests/test_phase3_models.py tests/test_phase3_relations.py tests/test_phase3_training.py tests/test_dashboard_app.py` — **100 passed**, 9 warnings (92.75 s).
- Nine-model real-RGB matrix: `.venv/bin/python -m pytest -q tests/test_phase3_training.py::test_nine_real_image_training_smokes_update_save_reload_and_evaluate` — **9 passed** (15.08 s), all baseline/box/town × global/compositional/hybrid cells.
- Forced-death, malformed-lock, live-writer, and stale-manifest tests — **4 passed** (15.04 s).
- Saved-town snapshot/dashboard relation AppTest — **1 passed** (9.34 s; one deprecation warning).
- `.venv/bin/python -m compileall -q src tests` — pass. `.venv/bin/python -m pip check` — no broken requirements.
- `.venv/bin/town validate artifacts/datasets/pizza-4800-phase3 --no-write-report --json` — valid, 0 errors, 177 low-support warnings. `town inspect` reports `current_pass`, 4,800 samples. Test model outcomes were not read.
- `git diff --check 5a23bb0..e1bdd417` — pass.

The complete suite includes dependency-boundary checks, checkpoint/resume parity, ROI and model regressions, snapshot tests, weighting formulas/gradients, config preflight, lifecycle subprocess tests, relation tests, and dashboard AppTest. POSIX detached-worker behavior was exercised on Linux. CUDA, Windows lock execution, unreliable/network filesystem advisory-lock behavior, and multi-user deployment are not claimed or required.

## H. Remaining findings

### D5a — Known subclass entailment is incorrectly reported as unknown

- **Severity:** Moderate; **affected module:** `src/town_model/dataset/ontology.py::geometric_relation_metadata` (exposed in relation results).
- **Minimal reproduction:** with the pinned ontology, canonical class IDs 26 (`MushroomTopping`) and 63 (`VegetarianTopping`), query containment metadata. `ontology.component_class_targets("MushroomTopping")` includes `VegetarianTopping`, but `geometric_relation_metadata("MushroomTopping", "VegetarianTopping", "containment")` returns `{"status":"unknown", ...}`.
- **Expected:** `entailed`. The ontology declares `MushroomTopping ⊑ VegetableTopping`, and `VegetarianTopping` is equivalent to `PizzaTopping ∩ (CheeseTopping ∪ FruitTopping ∪ VegetableTopping)`. The existing component-target logic already uses this supported equivalent-class fragment.
- **Actual/impact:** metadata searches only named `rdfs:subClassOf` closure and labels this known ontology relation unknown, obscuring an asserted/entailed relation in the required ontology-vs-geometry inspection.
- **Minimal correction:** extend relation metadata for the already-supported named equivalent-class fragment and test this and other supported inclusions. A general OWL reasoner is not requested; unsupported/missing relations must remain unknown.

### D5b — Relation JSON no-overwrite check races

- **Severity:** Moderate; **affected module:** `src/town_model/inspection/relation_diagnostics.py::save_relation_diagnostic`.
- **Minimal reproduction:** two concurrent calls with different results and the same absent destination; synchronize both immediately before `tempfile.mkstemp` so each has passed `path.exists()`. Both calls return success, no `FileExistsError` is raised, and the final `os.replace` silently leaves only one writer's JSON.
- **Expected:** atomic immutable publication: exactly one writer publishes; the other fails without replacing the winner.
- **Actual/impact:** the check-then-`os.replace` sequence is atomic per replacement but not atomic no-clobber publication. A concurrent local CLI invocation can silently replace a previously published relation result, undermining provenance.
- **Minimal correction:** use an atomic no-replace publish primitive (for example, a same-filesystem link/O_EXCL publication protocol) and add a deterministic concurrent-writer regression. No distributed or multi-user service is required.

No other new material defect was confirmed. These findings are limited to D5; the exact/sampled geometry decisions and read-only inspection behavior passed.

## I. Final Phase 3 acceptance-gate matrix

| Gate | Outcome | Evidence |
|---|---|---|
| **3A — Dataset, configuration, architecture** | **PASS** | Canonical prepared data validates; strict shared preflight rejects invalid configs before run publication; dependency-boundary tests pass. |
| **3B — Visual features and ROI contracts** | **PASS** | Previously accepted MobileNetV3/FPN, GT ROIAlign, coordinate and cache pathways remain covered by the full regression suite; no detector training is claimed. |
| **3C — Point models and composition** | **PASS** | Nine real-RGB family×composition smoke cells pass; existing baseline/box/town, geometry, known-mask, DeepSets and empty-set tests pass. |
| **3D — Training and comparisons** | **PASS for the pre-detector milestone** | Corrected vectors and saved checkpoints independently reproduce; weighted reduction/accumulation and checkpoint/resume tests pass. The runs remain exploratory, single-seed evidence, not final thesis comparisons. |
| **3E — Execution, persistence and dashboard** | **FAIL / D5 open** | Worker recovery and snapshot persistence pass, but relation ontology metadata is incomplete for a known entailment and relation-result no-clobber publication is racy. |

## J. Final acceptance and Phase 4 authorization

**Phase 3 is not accepted on this review. Phase 4 is not authorized.**

The smallest necessary remediation is: (1) correctly identify the already-supported equivalent-class subclass entailments in ontology metadata, and (2) make relation-result publication atomic and no-clobber under concurrent local writers. Re-run focused metadata and concurrent-persistence tests plus the full suite. Do not change the valid sampled-town qualification, require box to improve, add detector work, or expand this into a full RQ2 theorem-proving/benchmark project.

This verdict concerns only the pre-detector GT-ROI implementation foundation. It does not confirm the thesis's final RQ1/RQ2 claims.
