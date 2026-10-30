# Phase 2 remediation report — independent review findings F1–F7

**Status:** remediation implemented; ready for a genuinely independent closure review. This report does **not** claim independent acceptance.

## Revision and ancestry

- Accepted Phase 1 baseline: `bf2e3080a07734c50b9c679aa812d1e1e933db91`
- Reviewed Phase 2 implementation: `e241ce26140b36d28be61cc8dbb076693c4e6b80`
- Independent review commit: `ac1e54b9962be38deeb9a4edfbc3264004d7826d`
- Remediation branch: `remediation/phase-2-review-findings`
- The remediation branch was created at the review commit, which is a descendant of the reviewed implementation and Phase 1 baseline. No changes were made to `main`.

The original review and its **REQUEST CHANGES** verdict remain unchanged in [`reviews/phase_02_independent_review.md`](../reviews/phase_02_independent_review.md). The original implementation report is explicitly marked as a pre-review historical record; this report describes the remediated state.

## Findings closure matrix

| Finding | Root cause | Minimal correction and regression evidence | Status |
|---|---|---|---|
| **F1 — overlapping-town plot** | Opaque white window rectangles were painted after the model-evaluated town-membership raster and could erase another house's re-inclusion. | Membership raster remains the authoritative fill. Every intersecting frame and window is drawn as a non-filled boundary, with house/window identity and an explicit “house-specific exclusion” legend; the green fill is labeled model-evaluated final town membership. Renderer tests inspect the actual `QuadMesh` values and `Rectangle` primitives for the requested two-house case, a genuine exclusion, and an exact 3D slice. | Closed |
| **F2 — zero hinge margin** | Evaluation and resume used truthiness (`margin or 0.1`), mapping a configured `0.0` to `0.1`. | `_configured_margin` defaults only missing/`None` values to `0.1`; zero remains zero through training, evaluation, and resume. Tests compare saved/reloaded loss, immutable config/hash, zero-margin split-run parameter and Adam-state parity, and the unaffected default `0.1`. A separate CLI split-run check also reached exact parity. | Closed |
| **F3 — sparse high-dimensional windows** | A fixed `0.72` per-axis fraction compounded exponentially with dimension. | Added configurable, seeded, label-agnostic `window_target_volume` (default `0.25`), with per-axis fraction `q(d)=v**(1/d)`. This changes initialization only; hard membership, margins, loss, capacity, containment, and no-reset policy are unchanged. Deterministic multi-seed volume, occupancy, overlap, gradient and optimization comparisons are recorded below. New runs persist the setting under schema 2; schema-1 runs retain the old 0.72-per-axis initialization when loaded. | Closed; activity remains sparse in some branches as expected and is reported, not auto-corrected. |
| **F4 — infeasible minimum widths** | Validation allowed near-domain-width requests even though the added float32 reserve made the transform infeasible. | Reject impossible widths before parameter creation. Box and contained-town feasibility now includes the reserve; tests cover defaults, values below/at/above feasible limits, invalid/nonfinite requests, near-maximum accepted configs, extreme raw values, physical corners, minimum widths and clearances. No post-step corner clipping was added. | Closed |
| **F5 — temporally misleading snapshot diagnostics** | Historical geometry was shown with final-run activity and last-step gradients. | Each snapshot now stores its own train-point occupancy/house/window activity and geometry diagnostics. The viewer reports activity with snapshot identity/step; gradients are explicitly labeled final-step-only and separate. Old snapshots without activity remain readable and show a compatibility note; final-only metrics are not copied into historical snapshots. Tests exercise initial/intermediate/final states, a substantial activity change, and absent optional diagnostics. | Closed |
| **F6 — corrupt checkpoint handling** | Direct `torch.load` calls allowed pickle/EOF/zip deserialization exceptions to escape application error handling. | Centralized deserialization in `_load_torch_artifact`; known pickle, EOF, OS and PyTorch archive failures become `SavedArtifactError` with artifact path and cause. Structural/config/fixture/model/optimizer compatibility checks remain strict. CLI reports parser error (exit 2), dashboard reader/AppTest returns a failed-run message, and resume rejects invalid files. | Closed |
| **F7 — stale README scope** | Final README paragraph described the repository as Phase 1 only and said Phase 2 geometry/training were deferred. | Replaced the contradictory paragraph with a concise synthetic Phase 2 summary and links to the scope and implementation reports. It explicitly keeps detector and image-to-town functionality deferred. README warns that pickle checkpoints are only for trusted artifacts. | Closed |

## Numerical parameterization and initialization policy

Let `eps32 = torch.finfo(torch.float32).eps` and `R = 16*eps32` (the existing safety reserve). The configured minimum `m` is accepted only when:

- **Pure box:** `0 < m < 2 - 2R` (`2 - 32*eps32`, strictly).
- **Contained town:** `0 < m < (2 - 5R)/3` (`(2 - 80*eps32)/3`, strictly), which additionally leaves a frame large enough for one window and two positive clearances after the reserve.

The transforms reserve `m+R` for an axis width/clearance; town frames reserve `3*(m+R)+R`. Values at or above either limit are rejected before trainable parameters are created. Tests verify physical corners stay in `[-1,1]`, widths and both contained-window clearances remain at least the requested minimum in float32, including extreme raw values. This safety reserve is not a membership/loss epsilon.

Town initialization defaults to a **provisional target fraction** `v=0.25` of the available clearance-adjusted interior span above the minimum window-width floor. For each axis, `window_width = floor + q(d)*(interior-floor)`, where `q(d)=v**(1/d)`. The setting is configurable in `TownGeometry`, `TownClassifier`, `SyntheticConfig`, and the experiment CLI (`--window-target-volume`). Positions, frame sizes and house offsets remain seeded and label-agnostic. The target is not a claim that 25% is scientifically optimal and does not maximize holes; the actual window/frame volume fraction is measured, including the small floor effect. The resolved config records it. Checkpoint/config schema is version 2. Schema-1 checkpoint loading remains supported and uses `q=0.72`, preserving the old initialization for legacy state/snapshot use rather than silently applying the new default.

## F3 — before/after initialization evidence

### Uniform-point diagnostic

The old initialization was reconstructed as `v_old=0.72**d` (hence `q_old=0.72`) and compared with `v_new=0.25`. For each `d∈{2,8,16}`, ten independently seeded geometries were evaluated against 10,000 independently generated uniform points per seed in `[-1,1]^d`; point and geometry seeds used separate deterministic streams. Each geometry had two classes, two houses/class and one window/house. Labels were not used for geometry initialization. For the gradient probe only, both policies used the same fixed-window synthetic target rule and `m=0.1`. Reported values are means across seeds and physical windows unless otherwise stated.

| d | Policy (per-axis q) | Mean window/frame volume fraction | Mean points in each window (of 10,000; range) | Mean house-membership points (of 10,000) | Effective exclusions after house union (% of point/class probes) | Window grad L2 / exact-zero fraction | Frame grad L2 / exact-zero fraction |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | old (.72) | .518389 | 4,521.8 (45.22%; range 4,312–4,748) | 4,219.0 | 39.72% | .03045 / 1.25% | .00975 / 0% |
| 2 | new (.5000) | .249997 | 2,189.7 (21.90%; range 2,065–2,333) | 6,551.1 | 17.10% | .02755 / 0% | .00645 / 0% |
| 8 | old (.72) | .072214 | 411.8 (4.12%; range 330–476) | 5,288.9 | 2.50% | .00358 / 0% | .00371 / 0% |
| 8 | new (.8409) | .249977 | 1,416.3 (14.16%; range 1,267–1,621) | 4,284.4 | 9.81% | .00438 / .31% | .00362 / 0% |
| 16 | old (.72) | .005215 | 17.5 (.175%; range 9–26) | 3,216.0 | .055% | .000459 / 10.16% | .00299 / 0% |
| 16 | new (.9170) | .249951 | 812.5 (8.13%; range 681–980) | 2,421.0 | 6.04% | .000951 / .86% | .00248 / 0% |

The per-window counts are actual uniform-point occupancy, not volume-derived estimates. The configured `v` applies to span above the minimum window floor, so the reported physical window/frame fractions differ slightly from `v`. All physical corners and minimum width/clearance checks passed for every initialization. Every window was active in this large 10,000-point fixture; the separate 320-point runs below show expected sampling sparsity in the old policy. At `d=16`, mean occupancy improved about **46.4×**, effective exclusions about **110×**, and the window-parameter exact-zero gradient fraction fell from **10.16% to 0.86%** in the fixed-target gradient probe. At `d=2`, the new policy deliberately uses a smaller window; it still had nonzero occupancy and gradients, demonstrating that the policy is not simply maximizing low-dimensional holes.

Seeded diversity was checked separately: identical seeds reproduce physical geometry exactly; alternate seeds differ; houses/windows are not coincident. The target volume was stable across seeds while centers and offsets varied.

### Short optimization comparison

To measure activity at the experiment's smaller point budget, old/new initialization were run on matched fixed-window fixtures at `d=2,8,16`, seeds `73,74,75`, 320 points, 12 Adam steps, learning rate `.03`, margin `.1`, and snapshot interval 6. Each model had 224 training points; initialization activity is measured before optimization and final activity after step 12. Values below are means over three seeds; window counts sum four class/house windows. Gradients are the step-1 (initial-state) and step-12 window-group statistics.

| d | Policy | Initial → final window observations | Active windows initial → final (of 4) | Window grad norm; zero fraction initial → final | Frame grad norm; zero fraction initial → final | House-specific membership points initial → final |
|---:|---|---:|---:|---:|---:|---:|
| 2 | old | 379 → 337 | 4 → 4 | .03243 → .02310; 0% → 4.17% | .00982 → .00706; 0% → 0% | 419 → 440 |
| 2 | new | 166 → 180 | 4 → 4 | .02793 → .01946; 0% → 16.67% | .00672 → .00587; 0% → 4.17% | 632 → 606 |
| 8 | old | 26 → 33 | 4 → 4 | .00513 → .00365; 28.65% → 40.10% | .00516 → .00580; 1.56% → 10.94% | 454 → 410 |
| 8 | new | 117 → 104 | 4 → 4 | .00550 → .00429; 11.46% → 26.04% | .00565 → .00553; .52% → 9.90% | 362 → 348 |
| 16 | old | 4 → 4 | 2.33 → 2.33 | .00190 → .00138; 86.20% → 90.63% | .00442 → .00458; 10.68% → 15.89% | 276 → 239 |
| 16 | new | 75 → 68 | 4 → 4 | .00183 → .00157; 42.71% → 48.44% | .00426 → .00419; 4.43% → 12.50% | 204 → 173 |

Higher-dimensional activity is materially improved while retaining ordinary sparse min/max/hinge gradients. No window was reset, disabled or automatically relocated. The reduced `d=2` activity is an expected consequence of the chosen quarter-volume provisional default and remains substantial (all four windows observed in all three seeds).

### Genuine d=16 optimization check

A separate saved fixed-window experiment used seed 74, 320 points, 30 steps, `m=.1`, learning rate `.03`, and the new default. The town's initial window point counts (two classes × two houses) were `[[21,26],[21,19]]` and final counts `[[10,10],[23,26]]`; all four windows remained active. Mean window/frame volume fraction changed from `.24995` to `.26069`. Effective exclusions on training points were `[17,10]` initially and `[4,37]` finally. All physical corners remained in-domain; final minimum frame width was `1.67016`, window width `1.48279`, and minimum inner clearance `.02223` (all exceed `1e-5`).

| Family | Training objective, first → final | Validation loss / accuracy / macro-F1 | Final relevant gradient norm (zero fraction) |
|---|---:|---:|---|
| baseline | .70604 → .59667 | .71363 / .5598 / .4369 | classifier `.03938` (0%) |
| box | .09322 → .06952 | .07135 / .6630 / .4800 | geometry `.00534` (9.38%) |
| town | .09495 → .06994 | .08920 / .5924 / .4966 | frames `.00472` (15.63%); windows `.00173` (51.56%) |

Town window gradients were nonzero and finite; a nonzero fraction of parameter elements/branches remained inactive, which is expected under exact min/max and hinge optimization and is not concealed or repaired. These are short synthetic diagnostics, not comparative scientific conclusions.

## F1–F2 and artifact/UI verification details

- **F1 rendered regression:** `tests/test_phase2_dashboard_geometry.py::test_town_plot_keeps_overlap_reinclusion_and_window_boundaries_faithful` inspects the plotted membership mesh at `(0,0)` and confirms it is green/model-member for House 0 window `[-.2,.2]^2` plus House 1 window `[.6,.7]^2`, despite House 0's exclusion. It confirms each house/window is rendered as an outline-only rectangle. A second plot confirms a genuinely excluded center remains absent from the membership mesh and both window outlines remain visible. The same re-inclusion test runs on ordinary 2D and a full-model exact 3D slice.
- **F2 zero-margin workflow:** the fresh `margin=0.0` test compares stored and reloaded validation losses to `1e-12`, verifies schema-2 checkpoint/config identity, and checks default `.1`. The independent CLI run in `/tmp/town-phase2-remediation-zero-cli` trained 2 steps, evaluated, resumed 2 steps, and compared with an uninterrupted 4-step run. For both box and town, saved/reloaded losses matched before and after resume, resolved identity remained unchanged, and final model tensors matched the uninterrupted run exactly (`step=4`). Validation losses: box `.15000550`, town `.03556676`.
- **F5 snapshots:** snapshot activity is computed on the training split for that exact saved geometry and stored in the snapshot. The regression run seed 0 / 80 points / 20 steps changed total observed town-window membership from 52 at initialization to 78 at step 20; initial, step-5 and final displayed activity matched snapshot step identities. Removing optional activity from a historical-style step snapshot left it readable with a clear compatibility note. No historical gradients are fabricated.
- **F6 corruption:** tests cover arbitrary bytes, a truncated checkpoint, a valid checkpoint, a checkpoint with wrong fixture identity, structurally missing fields and incompatible model state, CLI `evaluate`, dashboard reader, Streamlit AppTest, and resume. CLI exits 2 with an actionable path/error and no Python traceback. The dashboard returns a failed-run message instead of raising an AppTest exception. Valid saved runs continue to evaluate and resume; configuration, fixture hash, vocabulary/order, model state and optimizer state checks remain enforced.

## Reproduced three-model synthetic comparisons

Disposable runs used seed 73, `d=2`, 160 points, 25 Adam steps, learning rate `.03`, margin `.1`, and snapshot interval 5. Fixed simple and fixed window fixtures plus a trainable-point mapping window fixture were each run through the same baseline/box/town harness. Training objectives decreased for all nine models; all geometry/encoder gradients were finite and nonzero at the relevant group level. Validation saved/reloaded losses matched exactly at the reported precision. These single-seed runs are smoke diagnostics only; BCE and hinge magnitudes are not compared as equivalent objectives.

| Fixture | Family / objective | Train loss first → final | Validation loss (= reloaded) | Known accuracy / macro-F1 |
|---|---|---:|---:|---:|
| fixed simple | baseline / BCE | .63651 → .45829 | .48161 | .9457 / .9352 |
|  | box / hinge | .19118 → .16067 | .17968 | .4891 / .5953 |
|  | town / hinge | .13131 → .07015 | .08808 | .5543 / .6167 |
| fixed window | baseline / BCE | .74395 → .66603 | .66079 | .6522 / .3509 |
|  | box / hinge | .16971 → .14948 | .15758 | .6522 / .7177 |
|  | town / hinge | .09109 → .05764 | .05934 | .6957 / .7314 |
| mapping window | baseline / BCE | .74953 → .46289 | .47118 | .8152 / .6983 |
|  | box / hinge | .37789 → .08973 | .09458 | .4674 / .0000 |
|  | town / hinge | .09735 → .03366 | .03385 | .8370 / .8233 |

The three mapping runs shared identical initial encoder hash `247522025a4415ee739d373c26c66d8257de26c7b7feb4a416d16a2df2657dcb`, then trained independent encoder copies. Weak single-seed box mapping F1 is reported, not hidden; the result is not evidence against the model family or a thesis conclusion.

## Files changed

- `src/town_model/dashboard/geometry.py` — non-destructive window-boundary drawing and explicit final-membership legend.
- `src/town_model/experiments/synthetic.py` — explicit margin resolution, snapshot activity, schema-2 config/checkpoint persistence, schema-1 compatibility, and centralized artifact loading/error translation.
- `src/town_model/models/geometry.py`, `src/town_model/models/classifiers.py` — feasible float32 minimum-width validation and dimension-aware window initialization.
- `src/town_model/dashboard/app.py` — selected-snapshot activity and separately labeled final-step gradients.
- `src/town_model/cli.py` — configurable target window volume.
- `tests/test_phase2_geometry.py`, `tests/test_phase2_experiments.py`, `tests/test_phase2_resume.py`, `tests/test_phase2_dashboard_geometry.py`, new `tests/test_phase2_artifact_errors.py` — regression and verification coverage.
- `README.md` — accurate Phase 2 scope and trusted-checkpoint warning.
- `docs/PHASE_2_SCOPE.md`, `docs/PHASE_2_IMPLEMENTATION_REPORT.md` — current initialization/range details and historical-status clarification.
- This report: `docs/PHASE_2_REMEDIATION_REPORT.md`.

## Exact verification commands and results

- `.venv/bin/pytest -q` — **138 passed, 0 failed, 0 skipped** (five existing Streamlit/Pandas deprecation warnings).
- `.venv/bin/python -m compileall -q src tests benchmarks` — passed.
- `.venv/bin/python -m pip check` — `No broken requirements found`.
- `.venv/bin/town validate examples/pizza-48 --no-write-report --json` — valid, 0 errors, 178 expected low-support warnings.
- `.venv/bin/town inspect examples/pizza-48 --json` — `integrity_status=current_pass`.
- `git diff --quiet bf2e3080a07734c50b9c679aa812d1e1e933db91 -- examples/pizza-48` — passed; the committed 48-image dataset is unchanged.
- Streamlit `AppTest` Geometry Laboratory tests — passed, including corrupt-checkpoint handling, snapshot diagnostics, and plot primitives. AppTest is an in-process test, **not a browser test**.
- `.venv/bin/town dashboard --headless --port 8765` — Streamlit health endpoint returned `ok`; `/` returned HTTP 200. This is a server-health check, **not a browser interaction test**. No successful real-browser test is claimed.
- CLI corruption tests verify actionable exit code 2/no traceback; zero-margin CLI train/evaluate/resume parity is recorded above.

The complete test suite includes Phase 1 dataset, CLI and dashboard regressions. Streamlit/Pandas emitted only its existing `DataFrame.reset_index(inplace=...)` deprecation warnings.

## Backward compatibility, limitations, and remaining defects

- New runs use resolved-config schema 2/checkpoint schema 2, with `window_target_volume` included in immutable configuration identity. Schema-1 checkpoints/configs remain loadable; when the window-volume setting is absent, loading uses the legacy `.72` axis fraction. Checkpoint state and old exact membership semantics are preserved.
- Historical snapshots without activity remain readable. Their historical activity is shown as unavailable; only a selected final state may use the existing final-run activity record. Final-step gradients are always labeled as final-step-only.
- Python pickle checkpoint loading is **not safe for arbitrary untrusted artifacts**. The CLI and dashboard are for trusted local run directories; deserialization errors are handled but no sandbox or safe untrusted-checkpoint format is claimed.
- Exact min/max and hinge gradients remain sparse by design. The corrected initialization materially improves high-dimensional participation, but does not guarantee every window has a nonzero gradient every step and does not represent dynamically inactive windows.
- Experiments remain small synthetic diagnostics. No browser interaction test, image detector, image embedding, image-to-town pipeline, deferred reference loss, independent-window mode, relation proof, or independent review acceptance is claimed.

No unresolved F1–F7 defect is known from the completed regression and verification work. A new independent closure review is still required; no acceptance verdict is pre-claimed.
