# Independent Phase 2 Remediation Closure Review

## Final verdict: **REQUEST CHANGES**

The F1, F2, F3, F5, F6, and F7 corrections pass independent checks. The physical transforms themselves now reject the original near-domain-width defect and remained valid under float32 stress tests. However, the run-level configuration path still accepts infeasible `min_width` values and begins training before the town constructor rejects them. This violates the explicit requirement that invalid configurations fail **before training**. Phase 2 is not formally accepted and Phase 3 is **not authorized to begin** until that preflight gap is fixed and tested.

## Revision and review boundary

- Repository: `midhunpradeep/town_model`
- Branch: `remediation/phase-2-review-findings` (present locally and at `origin`)
- **Exact implementation SHA reviewed:** `19fdc76de4c88c58b1c448f6e74e29da72100e21`
- Accepted Phase 1 baseline: `bf2e3080a07734c50b9c679aa812d1e1e933db91`
- Original Phase 2 implementation: `e241ce26140b36d28be61cc8dbb076693c4e6b80`
- First review: `ac1e54b9962be38deeb9a4edfbc3264004d7826d`
- The baseline, original implementation, and first review are all ancestors of the reviewed HEAD. The remediation branch HEAD equals the remote remediation branch HEAD.
- Reviewed the entire remediation diff, `ac1e54b..19fdc76` (15 files; 921 insertions, 71 deletions), including implementation, tests, docs, and the remediation report. `git diff --check` passed.
- The historical `reviews/phase_02_independent_review.md` remains unchanged and still records **REQUEST CHANGES**.

I read the complete expanded implementation specification, Phase 2 scope, implementation report, original independent review, remediation report, and README. The original review’s deferred image/detector pipeline, reference losses, independent-window mode, and advanced dashboard work were not treated as Phase 2 blockers.

## Environment and test method

Environment: repository `.venv`, Linux, Python 3.12.0, PyTorch 2.5.1+cpu, NumPy 2.5.4, matplotlib 3.11.2, Streamlit 1.66.1, pandas 3.1.0, scikit-learn 1.9.1, pytest 8.4.2. CUDA was unavailable; all training/integration runs were CPU runs.

Commands and results:

- `.venv/bin/pytest -q` — **138 passed, 0 failed, 0 skipped**. Five Streamlit/Pandas `Pandas4Warning` messages concern deprecated `DataFrame.reset_index(inplace=...)` in Streamlit’s chart helper.
- `.venv/bin/python -m compileall -q src tests benchmarks` — passed.
- `.venv/bin/python -m pip check` — no broken requirements.
- `.venv/bin/town validate examples/pizza-48 --no-write-report --json` — valid, 0 errors, 178 low-class-support warnings.
- `.venv/bin/town inspect examples/pizza-48 --json` — `integrity_status=current_pass`, 48 samples.
- `git diff --quiet bf2e3080a07734c50b9c679aa812d1e1e933db91 -- examples/pizza-48` — passed; the committed Phase 1 sample dataset is unchanged.

All experiment/checkpoint corruption tests used disposable `/tmp` directories. I did not change implementation files, tests, existing reports, or repository experiment artifacts. The working tree was clean before review work and remained clean after the independent checks (before adding this report).

## F1–F7 closure matrix

| Finding | Status | Independent evidence |
|---|---|---|
| **F1 — Overlap plotting** | **CLOSED** | Independently rendered the exact requested two-house case: both frames `[-0.8,0.8]^2`, windows `[-0.2,0.2]^2` and `[0.6,0.7]^2`. At `(0,0)` the model, selected-point result, and membership mesh are all `True`; both window rectangles are outline-only (`fill=False`) and the legend identifies final model-evaluated membership. Also exercised a genuine union exclusion, a window partly covered by another house, four overlapping windows, a full-model 3D slice, and an empty slice. No opaque window fill can erase union re-inclusion. Relevant code: `src/town_model/dashboard/geometry.py:130-156, 192-235`.
| **F2 — Zero-margin persistence** | **CLOSED** | Archived and ran the original implementation: at `margin=0`, stored/reloaded validation losses differed by 0.06003 (box) and 0.04906 (town). On remediation, a fresh zero-margin run’s step-1 stored training loss matched an independent hinge recomputation with `m=0` and differed from `m=0.1`. CLI evaluation after reload matched stored metrics; a 2-step + 2-step CLI resume matched an uninterrupted 4-step run exactly in model and Adam state for all three families. Resolved JSON and its checkpoint identity remained unchanged. An independent validation-loss recomputation matched persisted metrics. Default `m=0.1` also evaluated correctly. No remaining objective truthiness default was found. Relevant code: `synthetic.py:339-342, 566-580, 849, 875-928`.
| **F3 — High-dimensional initialization** | **CLOSED** | Independently compared old `q=0.72` against new target volume `0.25` using identical deterministic geometry seeds and uniform points. The policy is dimension-aware, configurable, seeded, label-agnostic, and still uses contained-window geometry. Detailed evidence follows below. Activity and gradients improve materially at `d=8,16`; remaining exact min/max/hinge sparsity is visible rather than hidden or auto-repaired. Relevant code: `geometry.py:280-379`, `classifiers.py:128-190`, `cli.py` `--window-target-volume`.
| **F4 — Physical-domain invariants** | **OPEN** | The original direct-constructor case `BoxGeometry(2,1,min_width=1.999999)` now fails with a clear float32-reserve error, and 80 raw-extreme combinations satisfied physical bounds, requested widths, and town clearances. But `SyntheticConfig.validate()` does not validate `min_width`; `run_experiment()` trains the baseline and box before an infeasible town width is rejected. Reproduction and required correction are below. Relevant code: `geometry.py:39-52, 303-330`; config/run path: `synthetic.py:52-68, 654-704`.
| **F5 — Snapshot diagnostics** | **CLOSED** | Independently loaded initial, intermediate, and final `d=16` snapshots. For each, recomputed train-point house/window occupancy from that saved model state and fixture points; counts matched the persisted activity exactly. Selected class/sample membership and margin matched the corresponding snapshot output and exact slice. A historical-style intermediate snapshot without activity remains readable and reports activity unavailable rather than copying final values. UI labels gradients as final-step-only and separates them from selected-snapshot activity. Relevant code: `synthetic.py:424-460, 964-1057`; `dashboard/app.py:464-514`.
| **F6 — Corrupt-checkpoint handling** | **CLOSED** | On disposable run copies, tested arbitrary bytes, half-truncated bytes, a valid checkpoint, fixture-ID mismatch, and incompatible model tensor shape. CLI evaluate/resume returned actionable exit code 2 without tracebacks for invalid artifacts; saved-geometry loading and the dashboard reader returned localized errors. Valid runs continued to load/evaluate. Fixture/config identity and parameter-shape checks remain enforced. Relevant code: `synthetic.py:470-485, 737-830`; `dashboard/geometry.py:43-81`; CLI error boundary in `cli.py`.
| **F7 — README consistency** | **CLOSED** | The complete README now states that the three-model synthetic geometry lab, training, comparisons, checkpoint workflows, and incremental Geometry Laboratory are delivered; image embeddings/detection and the image-to-town pipeline remain deferred. The complete Phase 2 scope likewise distinguishes implemented and deferred work. The original independent review’s `REQUEST CHANGES` verdict remains preserved unchanged.

## Mathematical and numerical evidence

### F1 visualization and exact set semantics

The original renderer was independently reproduced from the original implementation archive: its model and membership raster both reported `(0,0)` as inside, while an opaque white window patch with alpha `0.8` covered the point in the rendered axes. In the remediation, windows are boundaries identifying house-specific exclusions; the filled raster is the final union membership. I inspected both the plotted membership values and the actual Matplotlib patch fill/color/fill-order primitives.

For exact set semantics, I also tested closed frame endpoints, closed window-boundary exclusion with signed margin zero, overlapping-house re-inclusion, multiple overlapping windows, and a 3D exact slice. The exact hard membership—not the sign of a zero margin—remains authoritative. A separate d=3 anchor outside all frame extents produced an empty slice, correctly identified as empty.

### F3 initialization comparison

The old per-axis fraction was `q=0.72`, so the implied window/frame volume fraction was approximately `0.72^d`: `0.5184` at d=2, `0.0722` at d=8, and `0.00522` at d=16. I compared it with the new policy using three deterministic seeds, 320 uniform latent points per seed, identical points and frames/seed streams for the old/new policies. Activity counts are summed over two classes × two houses (four windows):

| d | Policy | Mean actual window/frame volume | Window memberships / 320 points | Active windows / 4 (per-seed values where sparse) | Effective exclusions after town union | Window-gradient L2 / zero fraction |
|---:|---|---:|---:|---:|---:|---:|
| 2 | old q=.72 | .51839 | 607.7 | 4.00 | 41.771% | .03348 / 0.00% |
| 2 | new target=.25 | .25000 | 286.3 | 4.00 | 16.927% | .03176 / 0.00% |
| 8 | old q=.72 | .07221 | 58.0 | 4.00 | 2.448% | .00481 / 15.62% |
| 8 | new target=.25 | .24998 | 185.7 | 4.00 | 10.260% | .00498 / 8.33% |
| 16 | old q=.72 | .00521 | 2.3 | 2.00 (`[3,0,3]`) | 0.052% | .00138 / 83.07% |
| 16 | new target=.25 | .24995 | 108.7 | 4.00 (`[4,4,4]`) | 6.146% | .00172 / 22.66% |

The gradient probe used the same deterministic synthetic target rule and signed-distance hinge for both policies; labels were not used to initialize geometry. A separate 10-seed, 10,000-uniform-point check confirmed per-window occupancy changes from 0.170% to 8.157% at d=16 (and from 4.141% to 14.390% at d=8). At d=2 the new policy reduces, rather than maximizes, hole size while preserving substantial occupancy and nonzero gradients.

A fresh d=16 three-family CLI diagnostic (seed 74, 160 points, 8 steps) showed town window observations on the 112-point train split changing from 46 initially to 40 finally; all four windows remained observed. Final window-gradient norm was `0.00277` with a `0.648` element zero fraction: active and finite, but correctly not claimed to be dense. Across the tested seeds, identical seeds reproduced geometry, alternate seeds differed, houses were distinct, corners stayed in `[-1,1]`, and contained windows retained positive clearance.

### F4 transform checks and the remaining preflight defect

Let `R=16*eps32=1.9073486328e-6`. The corrected direct parameterization enforces the pure-box limit `min_width < 2-2R = 1.9999961853` and the contained-town limit `min_width < (2-5R)/3 = 0.6666634878`. I tested small valid widths (`1e-5`, `1e-8`), values immediately below feasible maxima, values at/above maxima, raw parameters at ±`1e30`, box/frame/window corners, frame widths, contained-window widths, both clearances, and domain endpoints. The original accepted `1.999999` box case is now rejected; all 80 raw-extreme combinations passed physical-bound, width, and clearance assertions. Domain endpoints remained included; closed window boundaries remained excluded with margin zero.

**Unresolved run-configuration defect (Moderate; F4 remains open):** `SyntheticConfig` has a `min_width` field, but `SyntheticConfig.validate()` at `src/town_model/experiments/synthetic.py:52-68` does not validate it. The API run path always trains all three families in order (`synthetic.py:654-704`); geometry-specific validation is deferred to each model constructor (`_run_one`, `synthetic.py:500-507`). I reproduced this with:

```python
run_experiment(
    "fixed", "simple", output,
    config=SyntheticConfig(seed=2, points=30, steps=1, min_width=0.7),
)
```

The baseline and box each completed a training step and wrote metrics/checkpoints. Only then did the town constructor raise `ValueError: min_width is infeasible for contained town frames; it must be less than 0.666663488`; the run group was left marked failed. The box-only limit allows `0.7`, but this harness always includes a contained town. This contradicts the acceptance condition that invalid configurations fail before training.

**Minimum correction before acceptance:** validate `min_width` in the composed run configuration against the feasibility constraints of every family the run will launch (in particular the stricter contained-town bound), before creating the run group or training any model. Add a regression proving infeasible values fail with an actionable error and leave no trained family artifacts. The direct geometry constructors should retain their existing local checks.

## F2 persistence and independent model-family regression evidence

The original implementation’s zero-margin defect was reproduced by running the archived `e241ce2` code: stored versus reloaded validation losses were `0.1537713 → 0.2137986` for box and `0.0339790 → 0.0830411` for town. On the remediated version, the zero-margin CLI run used seed 31, 64 points, and 2 steps followed by 2 resumed steps; it matched uninterrupted 4-step model and Adam states exactly for baseline, box, and town. Independent validation objective recomputation matched persisted results (box `0.15214813`, town `0.04171646` after resume). A second fresh step-1 probe independently recomputed train losses at `m=0` and `m=.1`; stored history matched zero in both box (`0.16953273`) and town (`0.03585073`) and not `.1`. Default `m=.1` remained valid.

Fresh short CLI comparisons exercised fixed simple, fixed nonconvex/window, and trainable-point mapping-window fixtures, each with all three families. The fixtures, targets, known masks, and train/validation splits were shared. Initial mapping encoders matched across families, then final encoders diverged under independent optimization. All relevant classifier/geometry/encoder gradients were finite and nonzero at the group level; train objectives decreased in these short diagnostics. Examples (train loss, first recorded step → final):

| Fixture | Baseline BCE | Box signed-distance hinge | Town signed-distance hinge |
|---|---:|---:|---:|
| fixed simple, d=2 | .76609 → .68418 | .17527 → .16951 | .10881 → .08383 |
| fixed window, d=2 | .66507 → .62739 | .21836 → .21293 | .09623 → .06538 |
| mapping window, d=2 | .61310 → .51862 | .40449 → .10367 | .08345 → .04124 |
| fixed window, d=16 | .68156 → .62281 | .08999 → .08296 | .09154 → .08249 |

These are one-seed smoke diagnostics, not performance claims; BCE and hinge magnitudes are different objectives and were not compared as if commensurate. CLI-reloaded validation metrics matched stored results. I independently checked the common masked reducer on a hand example: loss `6.0`; fixed/unknown columns and an unsupervised row contributed neither loss nor gradient, with gradients `[.25,.25]` and `.5` on the two supervised rows.

## Dashboard and Phase 1 verification

- Plot helpers and saved-run views were inspected directly for all three classifier families. Baseline zero-logit boundaries use actual logits; box/town views use actual model membership. d>2 grid values are evaluated in the full space with other coordinates fixed, and point markers are labeled projections.
- Streamlit `AppTest` exercised the Geometry Laboratory with saved runs. Changing model, class, split, point, and snapshot produced selected metrics matching direct model/helper outputs; hashing the selected run directory before/after widget changes showed no changes or retraining. The suite’s corruption AppTest returned a controlled failed-run message.
- A real headless Chrome 134 CDP smoke test opened the documented Streamlit app, navigated to Geometry Laboratory, set a disposable run root, and rendered actual selected metrics. Streamlit health returned `ok` and `/` returned HTTP 200. This is browser interaction, in addition to—not inferred from—AppTest.
- Phase 1 AppTest opened the committed 48-image dataset in the explorer and rendered its image/annotation table. On a disposable copy, I changed one image pixel without updating manifests: validation failed, the dashboard marked sample counts unverified, withheld support tables, and did not crash or expose stale accepted statistics.

## Phase 2 functional-regression matrix

| Area | Status | Evidence / boundary |
|---|---|---|
| Neural baseline | **PASS** | Shared linear head, BCE-with-logits, nonzero classifier/encoder gradients, fresh fixed and mapping runs, CLI checkpoint evaluation. |
| Pure boxes | **PASS** | Training/evaluation, exact closed membership and coordinate-clearance margin independently checked; physical transform tests pass. F4 concerns composed-run preflight, not these direct box invariants. |
| Towns | **PASS** | Training, windows/frames, exact union membership, boundary behavior, overlapping-house re-inclusion, gradients and faithful visualizations verified. |
| Exact membership and margins | **PASS** | Independent corner inequalities and signed-coordinate-margin calculations agree; zero margins at excluded window boundaries do not override Boolean membership. |
| Signed-distance loss | **PASS** | Correct `relu(m-(2y-1)s)` polarity and masked reducer; zero and default margins independently verified across train/evaluate/resume. |
| Masked supervision | **PASS** | Independent hand-value/gradient calculation; shared fixture masks retained across families. |
| Fixed-point training | **PASS** | Fresh fixed simple and window runs trained all three model families from shared points/masks. |
| Trainable-point mapping | **PASS** | Shared feature fixture and byte-identical initial encoder states; independent final encoders and nonzero encoder gradients. |
| Checkpoint/run reproducibility | **PASS for valid configurations** | Reload metrics and split/full optimizer/model state match. **F4 run-level invalid-width preflight remains blocking.** |
| Dashboard geometry inspection | **PASS** | Figure inspection, AppTest, selected saved states, exact high-dimensional slice/empty slice, corrupt-checkpoint errors, and real headless Chrome interaction. |
| Phase 1 regressions | **PASS** | Full tests, committed sample validate/inspect, actual explorer AppTest, changed-data stale-statistics check. |

## Remaining nonblocking limitations

The approved Phase 2 scope remains synthetic point/region classification rather than image-to-town training. Detector/CNN/ROI pathways, Phase 3, reference-inspired losses, independent windows, formal ontology relation proofs, and full experiment-dashboard job management remain deferred. Exact min/max and hinge optimization can still have zero window-parameter gradients on some branches; the new initialization materially improves high-dimensional activity but does not promise every parameter is active every step. The 48-image Phase 1 sample has expected low-support warnings. CPU-only testing does not establish CUDA behavior or thesis-scale performance. These do not block Phase 2 acceptance by themselves.

## Phase 3 authorization

**Refused for now.** Do not begin Phase 3 until the run-level `min_width` preflight is added and independently verified to reject infeasible configurations before any family trains or run artifacts are created. After that targeted correction, a closure review should repeat this check and the regression suite; no other material acceptance condition remains unsatisfied by the evidence above.
