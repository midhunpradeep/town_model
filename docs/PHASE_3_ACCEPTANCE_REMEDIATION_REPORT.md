# Phase 3 acceptance remediation report (D1–D5)

**Disposition: READY FOR INDEPENDENT CLOSURE REVIEW.** This is the implementation agent's remediation handoff, not an independent review and not Phase 3 acceptance. No Phase 4 work is included.

## 1. Revision and scope

- Repository branch: `remediation/phase-3-acceptance-findings`
- Exact starting independent-review commit: `5a23bb034c57719c870a18fed51349502e65f6a0`
- Reviewed implementation before remediation: `3c2af2fc154a1823247c8bb5daa58620e018ba1d`
- Final source/test/config/architecture implementation commit: `e1bdd41722c70160e56b4b1735f7ffd2fafb8684`
- The branch was created directly from the review commit after fetching `origin`; no newer commits existed. `main`, the pre-review implementation branch, and previous review documents were not modified.
- Scientific authority: [`pizzaiolo_thesis_implementation_spec_v1_1_expanded.md`](../pizzaiolo_thesis_implementation_spec_v1_1_expanded.md), including §6.2 weighting and §8.3 ontology relation diagnostics.
- This remains the **pre-detector GT-ROI image-embedding milestone**. It does not include detector pretraining, predicted-ROI learning, or joint detector/ontology training.

## 2. Finding disposition

| Finding | Status | Main correction |
|---|---|---|
| **D1 — positive-class weighting formula** | **Closed in implementation; corrected reruns completed** | Positive-count inverse-sqrt/effective-number formulas; explicit all-in-scope normalization, semantic version, persisted vectors, strict legacy-resume rejection. |
| **D2 — incomplete configuration preflight** | **Closed** | Strict shared model/vision validation and constructor-equivalent preflight occur before run/cache/reservation/worker side effects. Dashboard, CLI preflight, and direct service share `resolve_image_experiment()`. |
| **D3 — abrupt-worker and lock recovery** | **Closed** | Kernel advisory-lock ownership, inherited launch reservation, PID-start identity, manifest/status reconciliation, unknown exit status on forced death, compatible recovery. |
| **D4 — snapshot axis ambiguity** | **Closed** | Snapshot schema v2 declares diagnostic axes; geometry parameters remain class-scoped and only point activity is concatenated. Incompatible v1 snapshots are rejected clearly. |
| **D5 — relation diagnostics** | **Closed for the required foundational interface** | Exact closed-box relations; seeded, explicitly inconclusive town sampling with exact membership witnesses; ontology metadata, CLI, dashboard, and reproducible JSON persistence. |

The earlier independent F1–F4 corrections (weighted-loss denominator, geometric hinge defaults, dashboard controls, letterbox global ROI) remain intact.

## 3. D1 — formula, scale and historical compatibility

### Reproduction and root cause

The previous `_positive_weights()` used `sqrt(n_known_negative/n_known_positive)` for the name `inverse_sqrt`, and a ratio of negative/positive effective counts for `effective_number`. Those are not the formulas in spec §6.2. For known-positive counts 24 and 1,404, the required uncapped inverse-sqrt ratio is:

```text
(1/sqrt(24)) / (1/sqrt(1404)) = sqrt(1404/24) = 7.64852927…
```

The old implementation's result also depended on the known-negative counts. The corrected resolver accepts known-positive counts only; it never uses negative or unknown counts.

### Corrected objective and normalization

For each supervision level independently (pizza and component), the raw positive weights are:

```text
inverse_sqrt:      r_c = 1 / sqrt(n_c^+)
effective_number:  r_c = (1 - beta) / (1 - beta**n_c^+)
```

The implemented, documented common-scale convention is **not claimed to be prescribed by the specification**: compute the common scale over all positive-supported canonical trainable classes in that level, so the most-supported class has positive weight exactly 1; apply the configured upper cap afterward; then project the canonical vector onto the selected class IDs. This makes the scale independent of the selected subset. It preserves uncapped class-to-class ratios. The configured cap applies after normalization. Known negatives remain exactly 1, unknown entries contribute nothing to counts or loss, and the existing corrected reducer still divides each instance's weighted numerator by the **sum of active class weights**.

Comparison-quality runs fail for required in-scope classes with no known-positive training support. Exploratory zero-support classes get finite neutral weight 1 and have no positive examples to reweight. `none` returns all ones. The additional alternative is named `experimental_inverse_frequency`, documented as `1/n_c^+`; it is not the authoritative policy. The legacy name `inverse_frequency` is rejected because historical code used the different `n_negative/n_positive` ratio.

The semantic identity is `positive-count-proportional-v2-all-in-scope-most-supported-class-unit-scaled`. The formula, cap, normalization convention, actual vectors, and version are persisted in the resolved config, run manifest, checkpoint, and objective identity. The changed config identity means a legacy checkpoint cannot be resumed silently; a regression test builds a legacy-policy identity and verifies rejection. Existing artifacts were not rewritten.

### Numerical verification and tests

Independent hand calculations and tests cover equal supports; the 24/1,404 ratio; mixed positive/negative supervision; unknown labels and zero unknown gradients; zero-positive support and comparison failure; cap activation; `beta=0.9` and `beta=0.99`; distinct pizza/component support; `none`; the explicitly experimental alternative; and weighted effective-batch gradient parity. Tests also mutate negative and unknown support counts without changing resolved weights and instrument `_instance_terms()` to verify the actual vectors passed by the training loop.

### Historical and corrected experiment consequences

The historical pre-review controlled runs remain unchanged and are **not** measurements of the approved formula. Their stored component vector was `[3.793791921397, 1.0]`, generated from the former negative/positive imbalance implementation. Historical run IDs remain:

- Baseline: `image-f1be1bbcbbfc00c32f5c`
- Box: `image-5d5994778dfda9e96a6f`
- Town: `image-9be47c42d41f8eca4f24`

The canonical prepared dataset was available, so fresh immutable controlled runs were completed at `artifacts/phase3/acceptance-remediation-learning-v2/` (ignored; not committed). Only family varied across runs. Dataset ID and manifest SHA are `pizza-898a12a57e50a7a0d690` and `580070e2deb301e28757bf394ecbeb31adfeae5c3eb7b41807750be75644d24d`; class-manifest SHA is `e803479520664a5de823848c41a29f45a70f6d2a755049a9be14b52312cddd54`. Train/validation subset IDs are `c19c4379f9071a069d4a9bcf4f5a2f26aeb23f78047175cce8ef9b9dc4bb7e7d` / `f36b1ad7304a4925400793c948d3d27c64e304cbf24586f041192ae4682ae80d`; sizes are 128/48 from the original train/validation partitions. No test outcomes were used.

Selected class IDs remain `2 AnchovyTopping` and `29 NonVegetarianPizza`; hybrid composition, `d=8`, seed 42, frozen ImageNet MobileNetV3-Large/FPN features, effective batch 16, AdamW, and the original optimizer/schedule settings are unchanged. Train support is Anchovy component `140 positive / 2,015 negative / 0 unknown` and NonVegetarianPizza pizza `75 / 53 / 0`; validation support is `53 / 743 / 0` and `28 / 20 / 0`, respectively. Cross-level selected classes are known-negative with no positives and are excluded from macro-F1 by the existing rule.

Resolved positive vectors are in selected order `[AnchovyTopping, NonVegetarianPizza]`:

```text
component: [3.8442907575483187, 1.0]
pizza:     [1.0, 1.3063945294843615]
cap:       10.0 (not activated for these entries)
```

The scale uses all in-scope supported canonical classes: the largest component support is `Food=2,069`, and the largest pizza support is `Pizza=128`. The component Anchovy weight is therefore `sqrt(2069/140)`; the pizza NonVegetarianPizza weight is `sqrt(128/75)`. This differs from the previous implementation and is persisted in each run.

### Corrected learning results

Initial metrics are separately persisted in each checkpoint/run metric record. Best and latest checkpoints were independently reloaded and evaluated on validation. F1 order is **pizza / component GT-ROI**. Loss order is **component / pizza / total**.

| Family | Initial validation loss | Initial validation F1 | Final train loss | Final train F1 | Final validation loss | Final validation F1 | Best monitor |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | `0.743825 / 0.770858 / 1.514683` | `0.2353 / 0.1249` | `0.306769 / 0.361044 / 0.667813` | `1.0000 / 0.9630` | `0.307677 / 0.365763 / 0.673440` | `0.9818 / 0.9709` | `0.9854` |
| Box | `0.746521 / 0.564251 / 1.310772` | `0.7368 / 0.1249` | `0.024276 / 0.062511 / 0.086787` | `0.0000 / 0.0000` | `0.024870 / 0.062396 / 0.087267` | `0.0000 / 0.0000` | `0.3684` |
| Town | `0.028991 / 0.219043 / 0.248034` | `0.0000 / 0.0000` | `0.003036 / 0.018533 / 0.021569` | `1.0000 / 0.9964` | `0.003430 / 0.018823 / 0.022253` | `1.0000 / 0.9709` | `0.9854` |

| Family | Best checkpoint (epoch/step) | Best validation loss | Best validation F1 | Best SHA-256 | Latest checkpoint (epoch/step) | Latest validation F1 | Latest SHA-256 |
|---|---|---:|---:|---|---|---:|---|
| Baseline (`image-96e7ed30100b22c6e5de`) | 8 / 64 | `0.314746 / 0.381773 / 0.696519` | `1.0000 / 0.9709` | `e4f856e22948cf13ca1e80589a1b9442c0a95ae159b3475c517e225a96013fc2` | 12 / 96 | `0.9818 / 0.9709` | `72e8e0a5c8975b963f413f784eec1665c88136cc029ddec83c82bd7becb44e95` |
| Box (`image-f8e90b2b8f3dc833271f`) | 1 / 8 | `0.028516 / 0.159973 / 0.188489` | `0.7368 / 0.0000` | `6e7e34dd1bf4ca672af9e460f73eced1a76328164d142935a2b09fa6ab9e61ea` | 9 / 72 | `0.0000 / 0.0000` | `dc95e57105983891279dcf5194ddddc73001e7849900ec9d2a3246904f2473f9` |
| Town (`image-5dc37ef671f4e81187e4`) | 7 / 56 | `0.003740 / 0.020957 / 0.024697` | `1.0000 / 0.9709` | `629c44c816a851dd99d641336ff0c4ae8f55d7fd66e49afe264bfd0afaf8a597` | 12 / 96 | `1.0000 / 0.9709` | `2869bb4f7562c1ea27bf17b4c19102f53e76240fa3252549d705df0dbe252022` |

At optimizer step 1, active classifier/component-projection/DeepSets/global-projection/fusion gradient L2 norms were baseline `0.5842 / 0.8955 / 0.1811 / 0.3613 / 0.3069`, box `0.0525 / 2.6268 / 0.5806 / 0.8114 / 0.9782`, and town `0.0439 / 0.0879 / 0.2970 / 0.5845 / 0.4153`. FPN and MobileNet were frozen as configured. The box hinge objective decreases substantially, but hard-membership F1 degrades to zero; the best box checkpoint still has pizza F1 `0.7368` and component F1 `0.0000`. No box implementation defect was confirmed, and no box-math change was made to improve F1. These are exploratory single-seed results, not a family ranking or thesis conclusion.

## 4. D2 — complete early preflight

### Reproductions and root cause

- `ImageModelConfig(family="town", windows=0)` previously passed the config check (`windows < 0`) although `TownGeometry` requires at least one window; model construction happened after run publication.
- `VisionConfig(visual_policy="trainable_fpn_last_blocks", trainable_backbone_blocks=(3.5,))` previously passed config validation although the vision constructor only accepts integer MobileNet feature indices.

The two validation layers had divergent preconditions and service execution constructed/published run artifacts before reaching the later constructor failure.

### Correction and verification

A pure `validate_image_model_parameters()` helper is shared by config preflight and `ImageOntologyModel`; it validates integer counts (rejecting booleans), positive town windows, family-specific parameter compatibility, class names/order, geometry feasibility, and window settings. `validate_mobilenet_block_indices()` is shared by config and the vision constructor; it rejects fractional/boolean/duplicate/out-of-range indices. Vision runtime/version, ROI contract, resource/optimizer settings, class IDs/order, and incompatible feature-cache combinations are checked before execution. `resolve_image_experiment()` also instantiates the lightweight point/model structure and validates the actual ordered vocabulary before output creation; it does not construct Torchvision execution modules.

Direct service, dashboard launch/preflight, CLI `preflight`, and CLI `run` tests assert the run root, manifest, checkpoints, configured feature cache, worker directory, and reservation directory remain absent for both reproduced configurations. Class vocabulary/order and model-specific incompatibilities are also tested. The dashboard and CLI call the same resolver; invalid input is not converted into a failed partial run.

## 5. D3 — crash-safe local lifecycle

### Root cause and correction

The prior `O_EXCL` lock wrote owner JSON after file creation; a crash in that interval left malformed metadata that code treated as a permanently live owner. Worker status was trusted independently of process/manifest state, and detached launch reservation JSON did not survive abrupt termination coherently.

Run writers and detached launches now use persistent lock inodes and kernel advisory locking (`flock` on POSIX, byte-range lock for synchronous use on Windows). Metadata is descriptive only and written under the lock; it is never used to displace a writer. Detached POSIX workers inherit the launch-reservation lock descriptor, so termination releases it in the kernel. No lock file is deleted by age or malformed-content heuristics. A lock probe is held while stale manifest recovery is written, avoiding a race with a newly resumed writer. Linux `/proc` process start-time tokens disambiguate PID reuse; remote/uncertain owners are treated conservatively. Network filesystems without reliable advisory locking are unsupported. Detached worker descriptor handoff is POSIX-only; synchronous run locking has a Windows byte-range fallback.

Worker status is reconciled against the authoritative manifest and liveness, not taken as proof by itself. `SIGKILL` yields `interrupted`, `exit_code: null`, and an explicit unknown/abnormal-exit reason; no unobserved exit code is invented. The latest atomic checkpoint remains unchanged and resume requires the existing strict identity check.

### Lifecycle evidence

Real subprocess integration tests passed for normal detached completion, graceful stop, graceful resume, forced worker death, stale status reconciliation, orphan launch reservation reacquisition, malformed partial lock metadata, a live writer that cannot be displaced, concurrent launch rejection, and latest-checkpoint reload/resume. The forced-death test recorded the latest checkpoint hash before/after inspection, verified worker/run state became interrupted with unknown exit status, and resumed compatibly to completion. Effective-batch checkpoint/resume parity matched uninterrupted training.

## 6. D4 — explicit snapshot diagnostic axes

### Reproduction and correction

The old serializer used `value.shape[0] == batch_size` to infer semantics. For `B=C=4`, town frame geometry `[C,H,d]` collided with the batch dimension and was treated like per-image data. The correction introduces a named `GEOMETRY_DIAGNOSTIC_AXES` contract and `image-research-snapshot-v2`:

- Box geometry: `[C,d]`.
- Town frames: `[C,H,d]`.
- Town windows: `[C,H,W,d]`.
- Point activity/margins: `[point,C,H]` or `[point,C,H,W]`, reshaped by declared point semantics only.
- Persisted component activity: `[component_occurrence,C,...]`; pizza activity: `[image,C,...]`.

Empty point batches preserve empty activity tensors and class geometry. Snapshot v2 records the axis schema, checksums, and exact classifier parameter arrays; reload verifies their numerical equality with the header geometry. Old v1 snapshots without trustworthy axes fail with a clear incompatibility error rather than being silently reinterpreted.

### Round-trip evidence

Tests cover `B=C=4`, `B≠C` (`B=2,C=1`), one image/one class, multi-house/multi-window towns, multiple components, an empty component set, box geometry, town geometry, and point-wise house/window activity. Serialized arrays were reloaded and checked with exact numerical equality against source model tensors. Reloaded points and geometry were rendered through the dashboard helpers; a real `AppTest` also rendered a saved town snapshot. Snapshot bounds, checksums, and atomic publication remain in place.

## 7. D5 — minimum ontology relation diagnostics

### API and semantics

`src/town_model/inspection/relation_diagnostics.py` is the common inspection implementation. It consumes the existing `BoxClassifier`/`TownClassifier` geometry contract or versioned saved snapshot geometry; Streamlit does not reimplement relation calculations. It supports:

- `containment`: `A ⊆ B`.
- `intersection`: `A ∩ B != ∅`.
- `disjointness`: `A ∩ B = ∅`.

For closed axis-aligned boxes, containment, intersection, and disjointness are exact interval-algebra decisions. Boundary touching is nonempty intersection. Failed containment returns a membership-verified point in `A\B`; overlap returns an intersection witness; exact disjointness returns a separating-axis certificate. Tolerance is explicitly zero under the stored float32 corners.

Town queries are reproducible samples: `N` uniform points from `[-1,1]^d`, `N` proposals from source house frames, and `N` from target house frames; every candidate is checked by the accepted exact membership semantics (closed frames, excluded closed windows, overlapping-house re-inclusion). Results report seed, `3N` candidates, sampling domain/policy, observed counts, and verified witness/counterexample points. A sampled witness can establish an existential result/counterexample; no finite sample establishes universal containment or disjointness. No witness yields `outcome: null` and `inconclusive`, not `true`.

Each result also stores class IDs/names, query, exact/sampled method and scope, latent dimension/domain, boundaries/tolerance, geometry hashes, snapshot/checkpoint provenance, and ontology relation status. The canonical OWL metadata distinguishes asserted/entailed subclass and disjointness from unknown/not-established relationships. Named subclass closure and explicit `owl:disjointWith`/`owl:AllDisjointClasses` data are read; absence is not false. Snapshot IDs/names are checked against the canonical ontology vocabulary. No new ontology inference rule or training-time relation loss is added.

### Examples, CLI, persistence, and UI

Box unit tests cover equal/coincident (including degenerate point boxes), nested, partial overlap, boundary touch, and strictly disjoint cases. Town tests cover overlapping houses, window interior removal and boundary exclusion, second-house re-inclusion, deterministic counterexample sampling, and a no-counterexample result that remains inconclusive. JSON results can be atomically saved with immutable destination semantics and reference the checksummed snapshot. CLI:

```text
town experiment image relations SNAPSHOT \
  --source-class-id ID --target-class-id ID \
  --relation containment|intersection|disjointness \
  --sample-count 10000 --seed 0 --output result.json
```

The read-only dashboard panel calls the same API and offers JSON download. In a real Chrome smoke, saved town regions `American`/`AmericanHot` (IDs 0/1) produced sampled containment `outcome=false`: 14 verified violations among 300 candidates (100 per stratum), seed 0. Ontology metadata was `unknown`; this is a geometric diagnostic, not an OWL claim. The deterministic no-violation test correctly returns `outcome=null`/inconclusive.

## 8. Regression, data, UI, and hardware evidence

- Full regression: `.venv/bin/python -m pytest -q` — **261 passed, 0 failed, 0 skipped**, 11 existing Streamlit/Pandas `Pandas4Warning` deprecations.
- Nine real-RGB smoke matrix: isolated test — **9 passed** for all `{baseline, box, town} × {global, compositional, hybrid}` cells; each performs optimizer updates, checkpoint save/reload, and validation equality.
- Lifecycle subset including parity, graceful resume, forced death, malformed lock, and live-writer protection — **5 passed**.
- Snapshot round trips, relation tests, weighting numerical/gradient tests, preflight no-side-effect tests, architecture-boundary tests, and Streamlit `AppTest` are part of the full passing suite.
- `.venv/bin/python -m compileall -q src tests` — pass.
- `.venv/bin/python -m pip check` — no broken requirements.
- `git diff --check` — pass.
- Canonical prepared dataset validation (read-only): `pizza-898a12a57e50a7a0d690`, **0 errors, 177 low-support warnings**; inspection `current_pass`, 4,800 samples. Original splits/schema/labels/test partition were not modified. No test-model outcomes were inspected.
- Runtime: Python 3.12.0, PyTorch 2.5.1+cpu, Torchvision 0.20.1+cpu. `torch.cuda.is_available()` is false and CUDA version is `None`; **no GPU execution is claimed**.
- Real browser: headless Google Chrome `134.0.6998.88` over CDP loaded the Image Embeddings dashboard, selected a v2 town snapshot, ran the relation query, and displayed the downloadable JSON control. Screenshot is only `/tmp/phase3-browser-smoke.png`; it is not committed. No multi-user deployment or distributed scheduler was tested.

## 9. Files and data integrity

Implementation commit `e1bdd41722c70160e56b4b1735f7ffd2fafb8684` contains the D1–D5 source, focused tests, explicit experimental config naming, and the Phase 3 architecture clarification. Key files include `training/image_trainer.py`, `experiments/image_config.py`, `experiments/image_runs.py`, `experiments/image_worker.py`, `vision_validation.py`, `models/geometry.py`, `models/image_embeddings.py`, `inspection/image_snapshots.py`, new `inspection/relation_diagnostics.py`, `dataset/ontology.py`, `cli.py`, and `dashboard/image_embeddings.py`.

Generated datasets, feature caches, browser-smoke run, corrected learning runs, and checkpoints remain ignored under `artifacts/`; none is staged or committed. The canonical dataset schema/splits/labels, sealed test partition, prior learning artifacts, previous reports, and independent reviews were left intact. `docs/PHASE_3_IMPLEMENTATION_REPORT.md` receives only a clearly marked addendum; prior results and limitations remain historical.

## 10. Remaining issues and recommendation

- All corrected experiments are still **single-seed pre-detector exploratory GT-ROI** results on reduced 128/48 subsets. No comparison-quality multi-seed study, detector pretraining, predicted-ROI training, joint training, or final held-out test evaluation is claimed.
- The box run has decreasing hinge loss but poor final hard-membership F1; this is reported, not tuned away.
- The normalization scale is an explicitly documented implementation convention because the scientific specification requires normalization but does not dictate an absolute scale. Independent review should verify the selected all-canonical/in-scope convention and its interpretation.
- Town relation results remain sampled where exact proofs are not implemented; sample absence is never called proof. The full supervisor-approved RQ2 scoring protocol remains deferred.
- Snapshot v1 is rejected; there is no automatic migration for arrays whose semantic axes cannot be recovered safely.
- CUDA, network-filesystem advisory-lock behavior, Windows detached workers, and multi-user deployment are unverified/out of scope. Synchronous Windows lock support is present but was not exercised here.

**Recommendation:** conduct the narrow independent closure review of D1–D5 against this report and implementation SHA. This report does not declare Phase 3 accepted, authorize Phase 4, or merge into `main`.
