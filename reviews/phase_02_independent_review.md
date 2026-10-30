# Independent Phase 2 review — geometry, three-model comparison, and dashboard

**Verdict: REQUEST CHANGES**

The bounded Phase 2 subsystem is substantive: the neural, box, and town classifiers train; their common geometry/loss/reducer contracts mostly agree with the specification; and the synthetic comparisons are genuinely paired on inputs and training budgets. However, I found two material defects: the town plot can show a false hole where another house re-includes points, and valid `margin=0` runs are evaluated/resumed with margin `0.1`. There are also numerical-boundary, high-dimensional initialization, snapshot-diagnostic, and corrupt-checkpoint handling defects. Phase 3 should wait for the two major issues and the bounded scientific-integrity issues to be corrected and independently retested.

This verdict is **not** based on the absence of the image detector, image encoder, historical reference losses, independent-window mode, or advanced dashboard; those are explicitly deferred by the approved Phase 2 scope amendment.

## Review identity and state

- **Handoff:** `docs/PHASE_2_IMPLEMENTATION_REPORT.md` identifies `implementation/phase-2-geometry-classifiers` and Phase 1 base `bf2e3080a07734c50b9c679aa812d1e1e933db91`, but does not state the implementation HEAD SHA.
- **Verified implementation branch:** `implementation/phase-2-geometry-classifiers`.
- **Exact implementation revision reviewed:** `e241ce26140b36d28be61cc8dbb076693c4e6b80` (`docs: record Phase 2 scope, results, and deferred work`). `HEAD`, the local branch, `origin/implementation/phase-2-geometry-classifiers`, and `git ls-remote origin refs/heads/implementation/phase-2-geometry-classifiers` all resolved to this SHA when reviewed.
- **Accepted Phase 1 baseline:** `bf2e3080a07734c50b9c679aa812d1e1e933db91`; verified as an ancestor of the reviewed commit. The review covered the complete diff from that baseline.
- **Initial working tree:** clean on the implementation branch.
- **Review edits:** application code, tests, datasets, and prior reports were not changed. Experiment artifacts were written under `/tmp`; the only intended repository addition is this report.
- **Review environment:** repository `.venv` (Python 3.12.0 virtual environment, `include-system-site-packages = false`), Linux CPU-only. PyTorch 2.5.1+cpu; NumPy 2.5.4; matplotlib 3.11.2; Streamlit 1.66.1; pandas 3.1.0; scikit-learn 1.9.1; pytest 8.4.2; project 0.1.0. `torch.cuda.is_available()` was false. `pip check` passed; `compileall` passed.

I read the Phase 2 implementation report and scope amendment, and reviewed the pertinent scientific-specification contracts for shared classifiers, set/margin semantics, losses/reduction, numerical geometry, controlled experiments, exact slices, persistence, and acceptance. Source and tests were inspected directly; implementer claims were not treated as evidence.

## Test and independent verification summary

### Full suite and Phase 1 regression

Command: `.venv/bin/pytest -q`

**120 passed, 0 failed, 0 skipped.** Four warnings were emitted by Streamlit's chart helper (`Pandas4Warning`, deprecated `DataFrame.reset_index(inplace=...)`); no warning was promoted to a failure.

Independent Phase 1 checks:

- `.venv/bin/town validate examples/pizza-48 --no-write-report --json`: **valid**, 48 samples, 0 errors, 178 low-class-support warnings. The warnings are expected for this small three-per-generator fixture and were not interpreted as invalid data.
- `.venv/bin/town inspect examples/pizza-48 --json`: `integrity_status=current_pass`, 48 samples, no validation error.
- Existing dashboard tests cover the Phase 1 image explorer and ensure current validation failure blocks the explorer and withholds stale support tables. They passed.

### Dashboard execution and UI testing

- Started the documented CLI dashboard (`town dashboard --headless`) on a disposable local port; Streamlit health returned `ok` and `/` returned HTTP 200.
- Streamlit `AppTest` ran as part of the suite for the Geometry Laboratory and Phase 1 explorer. AppTest is an in-process UI test, **not** a browser test. Saved-run/plot helpers were independently exercised for all model families, comparisons, history, snapshots, a high-dimensional sample anchor, and an empty high-dimensional slice.
- A headless Chrome smoke attempt was made. Chrome reached the Streamlit shell but the captured page remained at a loading skeleton/blank state; I do **not** count this as a successful browser smoke test. A real interactive browser verification of widget changes remains unverified. This does not explain away the independently reproduced overlap-rendering defect below.

### Independent mathematical and reducer checks

I independently hand-calculated three-dimensional, two-class, two-house set and margin cases using NumPy expressions for the literal corner inequalities and formulas, rather than the implementation's helper functions. The results agreed for interior/frame-boundary points, closed-window-boundary exclusion, overlap, exclusion by one house with re-inclusion by another, outside-all-houses, multiple classes, and 3D points. In particular, an included frame-boundary point had margin zero, and an excluded window-boundary point also had margin zero; Boolean membership remained authoritative as required. Independent checks confirmed the prescribed box margin `min_i(x_i-l_i,r_i-x_i)` and the signed-distance hinge polarity. Existing gradient tests and fresh runs showed finite, nonzero gradients for representative incorrect examples; no post-step physical-corner repair was found. The reducer's default masked per-instance normalization and gradient scaling matched a hand-computed case: two supervised instances with losses 3 and 8 averaged to 5.5, the all-unknown instance was omitted, masked/fixed-class gradients were zero, and active gradients were `[0.25,0.25]` and `0.5` as expected.

## Independently reproduced experiments

I launched the documented CLI into fresh disposable roots with seed 73, 320 points, 60 Adam steps, learning rate 0.03, margin 0.1, and snapshots every 10 steps:

```bash
town experiment fixed   --output /tmp/town-phase2-independent-WzCcAn --seed 73 --points 320 --steps 60 --snapshot-interval 10
town experiment mapping --output /tmp/town-phase2-independent-WzCcAn --seed 73 --points 320 --steps 60 --snapshot-interval 10 --scenario window
```

This produced both fixed fixtures (`simple`, `window`) and the mapping fixture, all with separate baseline/box/town runs. The fixed runs trained only classifiers. Mapping runs trained independent copies of the same point encoder architecture and classifier. Per-run optimizer instances and parameter objects are separate; the initial encoder SHA-256 matched across families (`247522025a4415ee739d373c26c66d8257de26c7b7feb4a416d16a2df2657dcb`). The fixtures, class order, masks, and train/validation indices were shared within each group. Labels/teacher outputs are supervision only, not encoder inputs. Per-family resolved configs record objectives, dimensions, seeds and model/training settings. All three families completed real optimization and wrote separate checkpoints.

Validation metrics below are held-out predictions with known-label masking. BCE and hinge objective values are **not** comparable across families. The parameter count includes the encoder where present.

| Fresh seed-73 experiment | Family / objective | Parameters | Train loss | Validation loss | Validation known accuracy | Validation macro-F1 | Last-step gradient activity (norm; zero fraction where useful) |
|---|---:|---:|---:|---:|---:|---:|---|
| Fixed simple, d=2 | baseline / BCE | 6 | 0.34409 | 0.35200 | 0.9674 | 0.9665 | head 0.07733 |
|  | box / hinge | 8 | 0.08997 | 0.10513 | 0.6522 | 0.6870 | geometry 0.03539; zero 0.125 |
|  | town / hinge | 32 | 0.03933 | 0.04126 | 0.8315 | 0.8335 | frames 0.00191; windows 0.00760; both zero 0 |
| Fixed window, d=2 | baseline / BCE | 6 | 0.65323 | 0.63204 | 0.6685 | 0.3826 | head 0.00267 |
|  | box / hinge | 8 | 0.13372 | 0.14571 | 0.6413 | 0.6373 | geometry 0.02005; zero 0 |
|  | town / hinge | 32 | 0.03286 | 0.03721 | 0.8587 | 0.8308 | frames 0.00829; windows 0.00437; windows zero 0.25 |
| Mapping window, d=2 | baseline / BCE | 120 | 0.43258 | 0.45806 | 0.7967 | 0.8220 | head 0.02637; encoder 0.00666 |
|  | box / hinge | 122 | 0.05847 | 0.07325 | 0.6758 | 0.6757 | geometry 0.00735; encoder 0.01423; geometry zero 0.125 |
|  | town / hinge | 146 | 0.08531 | 0.09504 | 0.5275 | 0.6769 | frames 0.00078; windows 0.00108; encoder 0.00747; frame/window zero 0.3125/0.625 |

Actual train/validation objective traces at steps `1/10/20/30/40/50/60` (these are per-family objectives, not cross-family comparable):

| Group / family | Training loss trace | Validation loss trace |
|---|---|---|
| Fixed simple / baseline | .6553/.5689/.5014/.4496/.4077/.3728/.3441 | .6595/.5772/.5098/.4561/.4137/.3797/.3520 |
| Fixed simple / box | .1804/.1717/.1605/.1473/.1304/.1103/.0900 | .2027/.1939/.1822/.1680/.1504/.1289/.1051 |
| Fixed simple / town | .1136/.0866/.0668/.0560/.0477/.0428/.0393 | .1203/.0910/.0726/.0600/.0501/.0447/.0413 |
| Fixed window / baseline | .7450/.6987/.6728/.6601/.6546/.6533/.6532 | .7298/.6787/.6486/.6347/.6311/.6318/.6320 |
| Fixed window / box | .1786/.1717/.1640/.1565/.1485/.1406/.1337 | .2003/.1928/.1840/.1757/.1665/.1551/.1457 |
| Fixed window / town | .1014/.0713/.0526/.0448/.0398/.0357/.0329 | .0991/.0688/.0527/.0458/.0419/.0387/.0372 |
| Mapping window / baseline | .7281/.6213/.5415/.4890/.4610/.4439/.4326 | .7367/.6497/.5884/.5189/.4851/.4670/.4581 |
| Mapping window / box | .3424/.1040/.0942/.0872/.0777/.0678/.0585 | .3211/.1070/.1106/.1110/.0986/.0874/.0733 |
| Mapping window / town | .2460/.1003/.0969/.0901/.0880/.0862/.0853 | .2522/.0975/.0981/.0951/.1000/.0972/.0950 |

The baseline performed well on the simple linearly separable fixture; town performed well on the nonconvex fixed fixture. The box's lower performance on the simple half-space targets is a representational/optimization diagnostic, not evidence of a broken classifier. Mapping-town's seed-73 accuracy is weak, but it has active encoder and geometry gradients and decreasing training objective; I do not classify that outcome alone as an implementation failure.

Additional checks:

- `.venv/bin/town evaluate <run> --split validation` reloaded the default-margin checkpoints and reproduced their saved validation losses to printed precision.
- Two independently launched seed-89 mapping runs in separate roots produced identical fixture hashes and byte-equal final model and encoder tensors for all three families. The saved step-2 snapshot's predictions and point representations matched reevaluation of its persisted step-2 model/encoder state.
- A fresh fixed window experiment at `d=16`, seed 74, 320 points and 60 steps yielded validation losses/accuracy/F1: baseline `.69044/.5815/.3546`, box `.06397/.7065/.5173`, town `.07440/.6739/.5511`; parameters were 34/64/256. Town frame/window gradients were active but the last-step window group had zero fraction `0.8984`; one of four windows had no observed train membership. At initialization the four windows contained only `[0,1,0,0]` of the 320 uniform points. The initialized window volume was about `0.52%` of its frame volume in this d=16 case. This is evidence for finding F3, not a claim that town cannot learn in d=16; the d=16 mapping fixture, whose points lie on a low-dimensional teacher manifold, did activate windows.
- An actual persisted d=16 fixed-window baseline produced an empty exact `two_outer_lobes` slice at the zero anchor. Slice membership was evaluated by the original model in all dimensions, not by projecting the region.

## Findings

### F1 — Major: opaque window fills erase overlap re-inclusion in the town plot

**References:** `src/town_model/dashboard/geometry.py:130-147`, especially `144-147`.

**Reproduction:** render a saved or equivalent two-house 2D town where both frames are `[-0.8,0.8]^2`, house 0 has closed window `[-0.2,0.2]^2`, and house 1 has window `[0.6,0.7]^2`. At `(0,0)`, house 0 excludes the point but house 1 includes it. The actual model and the plot's membership mesh both report `True`. The renderer then draws house 0's window as an opaque-white (`alpha=0.8`) rectangle over the mesh, visually showing a hole at `(0,0)` despite the second house's re-inclusion.

**Expected:** the displayed set must agree with the union's hard predictions, including overlap. A window excluded from one house cannot visually erase another house's membership.

**Observed/impact:** selected-point membership is correct, but the dominant filled region is misleading for overlapping houses in 2D and in high-dimensional slices. This violates the central geometry/debugging-view acceptance condition and can cause incorrect research interpretation.

**Minimal correction:** keep the model-evaluated membership raster authoritative; render window boundaries with outline/hatching or mask white fills by the final town-membership result. Add a renderer test that probes a point re-included by another house.

### F2 — Major: valid zero hinge margin is replaced by 0.1 on reload and resume

**References:** `src/town_model/experiments/synthetic.py:753, 818, 825-833`.

**Reproduction:** run `run_experiment("fixed", "window", ..., config=SyntheticConfig(seed=31, points=64, steps=2, margin=0.0, snapshot_interval=1))`; then compare `metrics.json` validation loss with `evaluate_saved_run(run, "validation")`, or resume one step.

**Expected:** `m=0` is valid (`SyntheticConfig.validate` accepts finite nonnegative margins). Train, validation, reload and resume all use exactly zero hinge margin.

**Observed:** `float(config_margin or 0.1)` converts the valid zero into `0.1`. In my run, saved-vs-reloaded validation loss changed from `.1537713` to `.2137986` for box and `.0339790` to `.0830411` for town. Resuming also optimized and reported the wrong objective. Default `m=.1` runs are unaffected.

**Impact:** evaluation results no longer describe the configured run, and resumed optimization changes the scientific objective without a config change. This is a material reproducibility/objective defect.

**Minimal correction:** use an explicit `is None` branch (preserving numeric zero) at every evaluation/resume call. Add zero-margin tests for reloaded metrics and resumed-step parity.

### F3 — Moderate: dimension-independent 72%-per-axis windows are nearly inactive for uniform d=16 points

**References:** `src/town_model/models/geometry.py:314-328`, especially `318`; the accepted specification requires dimension-aware initialization at Q131-Q132 and §§5.4/10.2.

**Reproduction:** initialize default `TownGeometry(16, classes=2, houses=2, windows=1)` and use uniform points in `[-1,1]^16`, as in the fixed-point experiment. The code uses a 0.72 per-axis fraction, which compounds across dimensions; in the observed seed-74 d=16 run, initial window memberships over 320 points were `[0,1,0,0]` across the four windows, window/frame volume was about `0.0052`, and the last-step window-gradient zero fraction was about `0.90` with one inactive window.

**Expected:** broad, label-agnostic but dimension-aware window initialization that produces non-negligible exclusions/activity in the intended high-dimensional regime; report initial activity as well as final activity.

**Observed/impact:** the initialization is mathematically valid and nontrivial per axis, but almost no uniform fixture points are initially excluded in d=16. The town starts close to a box-like behavior for the fixed-point comparison and window gradients are highly sparse. The mapping d=16 case is less affected because its encoded points are concentrated on a lower-dimensional manifold.

**Minimal correction:** choose per-axis window fractions from a documented target volume/activity regime as a function of dimension (still without label-informed placement), and add d=16 uniform-fixture initialization/gradient tests. Treat the selected occupancy target as a visible provisional hyperparameter.

### F4 — Moderate: accepted near-2 `min_width` can generate out-of-domain boxes

**References:** `src/town_model/models/geometry.py:39-52, 179-196`.

**Reproduction:** `BoxGeometry(2, 1, min_width=1.999999)` passes `_safe_floor`'s documented `(0,2)` validation. The added float32 reserve makes `width_floor > 2`; initialization then yields a low corner of `-1.0000009536743164` and width `2.0000009536743164`.

**Expected:** any accepted configuration and raw-parameter values must keep physical corners in `[-1,1]^d` and satisfy the requested minimum width. If a requested floor is infeasible after the numerical reserve, reject it clearly.

**Observed/impact:** an accepted API value violates the domain invariant; box membership at the domain edge can consequently be wrong. The default `1e-5` configuration and tested raw extremes remain in bounds.

**Minimal correction:** validate `min_width + reserve < 2` (with a defensible strictness rule) before constructing a box, and add boundary tests around the maximum feasible width. Do not silently clip physical corners.

### F5 — Moderate: snapshot selection is paired with final-step activity/gradient diagnostics

**References:** `src/town_model/dashboard/app.py:447-456, 492-493`.

**Reproduction:** select `initial` or an intermediate saved snapshot in the Geometry Laboratory. Physical corners and predictions are loaded from that selected snapshot, but the JSON activity comes from final-run `metrics.json`; gradient diagnostics are also only `gradient_norms_last_step`. The UI labels the panel merely “Gradient and activity diagnostics” and does not identify it as final-step-only.

**Expected:** diagnostics shown beside a selected historical state should refer to that step, or be explicitly labeled as final-run diagnostics and kept visually separate from the selected snapshot.

**Observed/impact:** a user can read final occupancy/activity as if it described the displayed initial/intermediate geometry. Snapshots themselves load real states correctly, but the associated diagnostic context is inconsistent.

**Minimal correction:** persist per-snapshot activity and, if needed, per-step gradients; otherwise label and separate final-step diagnostics and never put them in the selected-snapshot geometry object.

### F6 — Moderate: corrupted checkpoints escape the CLI/dashboard error handling

**References:** `src/town_model/experiments/synthetic.py:704-705`; `src/town_model/dashboard/geometry.py:56-81`; `src/town_model/cli.py:232-233`.

**Reproduction:** replace a disposable run's `checkpoint.pt` with `b"not a pytorch checkpoint"`, then run `town evaluate <run>`. `torch.load` raises `_pickle.UnpicklingError`; the CLI exits with a Python traceback rather than the standard actionable parser error. `load_geometry_view` catches `OSError`, `ValueError`, `KeyError`, `TypeError`, and `RuntimeError`, but not `pickle.UnpicklingError`/EOF errors, so the same malformed checkpoint can escape its expected dashboard error result.

**Expected:** corrupted or incompatible artifacts fail clearly and locally without crashing the dashboard or producing an unhandled traceback.

**Observed/impact:** ordinary config/fixture mismatches are handled, but a truncated/arbitrary invalid checkpoint is not. This fails the explicit corrupted-checkpoint acceptance check and weakens the research instrument's robustness.

**Minimal correction:** catch the relevant deserialization errors at the persistence boundary, re-raise a concise `ValueError` naming the path and artifact, and cover both CLI and dashboard paths with a corrupted-checkpoint test.

### F7 — Minor: README's final scope paragraph contradicts Phase 2 delivery

**Reference:** `README.md:59-61` (contradicts `README.md:3,31-51`).

**Expected:** one consistent statement of implemented Phase 1/Phase 2 work and deferred work.

**Observed/impact:** the final paragraph says geometry, losses, training, and comparison are “explicitly deferred,” although the README above it and this branch deliver them. The Phase 2 report and amendment themselves are generally honest, but onboarding documentation is contradictory.

**Minimal correction:** replace the stale Phase 1-only scope paragraph with a short Phase 2 summary and link to the deferrals in `docs/PHASE_2_SCOPE.md`.

## Closure matrix

| Acceptance item | Status | Independent basis |
|---|---|---|
| 1. Baseline head and BCE | **PASS** | Shared `Linear(d,C)` logits, inclusive zero threshold, unreduced `BCEWithLogits`; trained and evaluated on both fixtures. |
| 2. Pure-box membership/margins | **PASS** | Inclusive physical comparisons and `min` coordinate margin agree with independent checks; gradients observed. See F4 for an extreme *configuration* range defect. |
| 3. Town membership/margins | **PASS** | Independent multi-class/3D house/window/set and margin checks agree; closed-window boundary is excluded regardless of zero margin. |
| 4. Numerical parameterization | **FAIL** | Default and extreme raw transforms pass, but F4 demonstrates accepted `min_width` can violate `[-1,1]`. |
| 5. Stable initialization | **FAIL** | Seeded, label-agnostic and noncoincident; d=16 uniform initialization has practically inactive windows (F3). |
| 6. Signed-distance loss | **PASS** | Training implements `relu(m-(2y-1)s)` for both geometries with correct polarity, hinge and active gradients. Reload/resume with `m=0` is separately defective (F2). |
| 7. Shared masked reducer | **PASS** | Manual value/denominator/gradient calculation agrees; unknown and fixed-class observations are excluded and all-unknown rows omitted. |
| 8. Fixed-point comparison | **PASS** | Same feature/target/mask/split fixtures and 60-step classifier-only budgets across all families; simple and nonconvex fixtures both trained. |
| 9. Trainable-encoder comparison | **PASS** | Same architecture and exact initial state hash; separate model/encoder objects and optimizer states; gradients through encoder and heads. |
| 10. Gradient correctness | **PASS** | Fresh runs show finite, nonzero activity in all families; selected incorrect examples and high-dimensional gradients pass. F3 limits town-window activity for uniform d=16 initialization. |
| 11. CLI/checkpoint reproducibility | **FAIL** | Default-margin deterministic reruns/reloads pass, but valid zero-margin reload/resume is wrong (F2), and corrupted checkpoints are not handled cleanly (F6). |
| 12. Actual 2D/high-dimensional geometry visualization | **FAIL** | Persisted state and full-dimensional slices are evaluated correctly, including an independently verified empty d=16 slice; overlap rendering is misleading (F1). |
| 13. Dashboard integration | **FAIL** | Phase 1 and Phase 2 AppTests and server health pass, but F1/F5/F6 prevent acceptance as a fully trustworthy instrument; no successful browser interaction test. |
| 14. Phase 1 regression status | **PASS** | Full regression suite passes; committed 48-image example validates and inspects as current-pass. |
| 15. Honest deferred-work documentation | **FAIL** | Phase 2 report/scope clearly state limitations, but README's stale scope paragraph falsely says delivered geometry/training are deferred (F7). |

## Nonblocking research limitations and out-of-scope items

These are not Phase 2 defects and are not reasons for rejection by themselves: no image detector/CNN or image-to-town pipeline; no `signed_bce`, reference-box BCE or reference-town loss; no independent-window mode; no component/image-level training or test-set protocol; no formal relation-proof/advanced analytical dashboard; no capacity matching between baseline, box and town. These are correctly identified as deferred in the approved Phase 2 amendment/report. Performance differences above are small synthetic diagnostics, not evidence for thesis RQ1/RQ2. The one-seed experiments are useful smoke/diagnostic runs, not comparison-quality multi-seed scientific conclusions.

## Final decision

**REQUEST CHANGES.** The three model families are implemented and trainable, and fixed/mapping comparisons are genuinely controlled for data, split, step budget, and initial encoder state (but not capacity or objective scale). The core signed-distance formula and hard set semantics are correct. However, the overlap plot violates the required faithful visualization semantics, and zero-margin checkpoint evaluation/resume uses a different objective than configured. Correct F1/F2, then address the associated numerical/high-dimensional and artifact/UI defects before treating Phase 2 as closed. Phase 3 should **not** begin on the basis of this revision until those material issues are closed and retested.
