# TOWN-03 CP1 Exploratory Numerical Report

> **Status: CP1 evidence only.** This is exploratory characterization, not formal numerical acceptance or completion of TOWN-03. CP2 is not authorized until the author separately signs off on numerical budgets and the confirmation set. Stop at this report; do not extend the Inspector, integrate, merge, train, or begin CP2.

## 1. Revision and scope

- Repository: `midhunpradeep/town_model`; issue [#23](https://github.com/midhunpradeep/town_model/issues/23).
- Direct author CP1-only confirmation: [issue comment 6813474135](https://github.com/midhunpradeep/town_model/issues/23#issuecomment-6813474135).
- Starting `main` / direct parent: `404d76112a9d53282b3f55698b4decb705a5601c`.
- **Implementation SHA:** `b1cd7c5dee66576c6945c0a7d10a6429a47fa092`.
- Branch/worktree: `implementation/town-03-cp1-signed-clearance`, `/tmp/town-model-town03-cp1`.
- Numerical artifact: [`TOWN03_CP1_EXPLORATORY_RESULTS.json`](TOWN03_CP1_EXPLORATORY_RESULTS.json), SHA-256 `7e31ffeaac15b704e3b582f84e4ef1fd5903591380a54dc98895f36a626457c2`.

Only CP1 pure evaluators, mathematical/autograd tests, T3 test accounting, the bounded exploratory probe, and evidence are included. No production Boolean membership, classifier, Inspector, trainable geometry, objective, optimizer, checkpoint, training, or integration behavior was changed.

## 2. Mathematical implementation

Added public functions in `src/town_model/models/town_geometry.py`:

```python
house_signed_clearance(points, frame_lower, frame_upper, window_lower, window_upper) -> Tensor
town_signed_clearance(points, frame_lower, frame_upper, window_lower, window_upper) -> Tensor
```

They implement the approved raw-coordinate equations using exact `torch.amin`/`torch.amax` reductions:

\[
s_B(x)=\min_j(x_j-l_j,r_j-x_j),\quad
s_H(x)=\min(s_F(x),-s_{W_1}(x),\ldots),\quad
s_T(x)=\max_h s_{H_h}(x).
\]

The functions reuse TOWN-02's physical-corner validators and shapes. They accept finite float32 pure query points outside `[-1,1]^d`; require valid finite ordered physical corners in the domain, exact ranks/shapes, common supported CPU/CUDA device, `d≥1`, and town `H≥1`; accept `N=0`, `W=0`, and valid degenerate fixed geometry. Empty queries return float32 `[0]`. `W=0` returns the frame clearance; `H=1,W=0` is exactly the BOX clearance. Native PyTorch autograd and tie behavior are retained. No smoothing, scaling, clipping, casts, transfers, custom backward, or trainable parameters are introduced.

The 36-test `tests/test_town_signed_clearance.py` covers literal house/town scores; H/W/N cases; box equivalence; overlaps, protrusions, degenerate frames/windows, outside-domain and largest-finite queries; malformed inputs and purity; independent Boolean membership; unique point/corner gradients; native min/max ties; cancellation; and finite differences. `tests/test_t3_architecture.py` separately accounts for 36 collected TOWN-03 mathematical nodes. The protected retirement manifest is byte-identical to the base: SHA-256 `b3b283a4b7849cd5dd5d08b9aa9ecf66888f8435954479a6f3a7823a0802ded3`.

### Boolean/score boundary limitation

A requested CP1 diagnostic called for a zero/positive-clearance versus Boolean-membership sign disagreement. That condition is mathematically impossible under the approved equations: for valid finite inputs, `s_H≥0` iff `house_contains` and `s_T≥0` iff `town_contains` (A1 ledger §2, consequence 1). I did not fabricate an expected disagreement or use clearance to implement membership. Tests call the accepted Boolean APIs independently and record zero-clearance included boundaries (where a strict `score > 0` test would disagree), positive-score members, and negative-score excluded points. The exact `score >= 0` sign relation is reported as a mathematical diagnostic, not a replacement Boolean API.

## 3. Verification commands and actual results

Environment paths below are from the isolated worktree; Python is the repository `.venv`.

- `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q -rs` — **623 passed, 0 failed, 0 skipped**; eight existing Streamlit/Pandas `Pandas4Warning` deprecations; 72.83 seconds.
- Focused BOX/TOWN/T3: `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q -rs tests/test_town_signed_clearance.py tests/test_town_composition.py tests/test_town_geometry.py tests/test_bounded_box.py tests/test_bounded_multi_box.py tests/test_t3_architecture.py` — **332 passed, 0 failed, 0 skipped**.
- New evaluator CUDA subset: `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q -rs tests/test_town_signed_clearance.py -k cuda` — **11 passed, 25 deselected, 0 skipped**; real CUDA execution.
- Full collection: `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest --collect-only -q -p no:cacheprovider tests` — **623 collected**.
- `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m compileall -q src/town_model tests tools` — passed.
- `/home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pip check` — no broken requirements.
- `git diff --check` — passed. Retirement manifest hash remains unchanged as recorded above.

The eight deprecation warnings predate this work; no failures or skips were observed.

## 4. Exploratory numerical characterization

Reproduce the recorded run from the implementation SHA with:

```bash
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python \
  tools/town03_cp1_exploratory_probe.py \
  --output /tmp/town03_cp1_exploratory_results.json
```

The probe uses an independent 100-digit `Decimal` oracle over the actual stored float32 point and corner values. The JSON artifact contains the stored inputs and per-device results for all comparison cases. This run had **18 literal exact fixtures** and **128 deterministic comparison cases**: 36 below/on/above nextafter boundary cases and 92 seeded cases (`seed=20261112`), covering every `d∈{1,2}`, `H∈{1,2}`, `W∈{0,1,2}` combination (10 or 11 cases per combination). Random queries were finite, non-simple float32 values; accepted physical corners included independent, overlapping and protruding windows.

### Environment and device measurements

- Python 3.12.0; PyTorch 2.5.1+cu124.
- CPU: Linux x86_64.
- Actual CUDA: NVIDIA GeForce RTX 2060, compute capability 7.5, driver 580.126.09, CUDA runtime 12.4. CUDA work was synchronized and executed, not skipped.

### Forward results

For each of house and town, on CPU and CUDA:

| Characterization | House | Town |
|---|---:|---:|
| Random/boundary comparison outputs per device | 128 | 128 |
| Maximum absolute error vs exact Decimal oracle | `5.960464477539063e-08` | `5.960464477539063e-08` |
| Mean absolute error | `6.999471224844456e-09` | `6.155460141599178e-09` |
| Maximum ULP error vs correctly rounded oracle | 0 | 0 |
| Outputs with `|oracle| ≤ 1e-6` | 24 | 24 |
| Maximum near-zero absolute error | 0 | 0 |
| Sign/zero crossings | 0 | 0 |

CPU/CUDA comparisons were bit-identical for these 128 outputs per evaluator: maximum absolute and ULP differences were both zero, with no exact-zero or sign mismatches.

Across the **18 independent literal exact fixtures**, maximum ULP error was 0 and there were no sign/zero crossings. Maximum absolute error was 1.0 solely for the largest-finite-float32 query: the exact real clearance is one unit from `-FLT_MAX`, while correctly rounded float32 is `-FLT_MAX`; relative error was `2.94e-39`. This is why a single absolute tolerance is inappropriate over the full finite input range.

### Gradients and ties

- Unique-branch analytic point/corner gradients matched exactly on CPU and CUDA (maximum absolute gradient error 0); the largest unique-case score discrepancy from its decimal literal was `5.96e-9`.
- Selected `amin`/`amax` ties matched independently calculated native equal-split gradients exactly (maximum absolute discrepancy 0). The tied town/house examples show point-gradient cancellation while active corner gradients remain, without custom backward behavior.
- CPU/CUDA gradient and score differences in the probe were zero.
- A non-tie central finite-difference probe with float32 step `0.00100000005` measured gradients `±0.99998707` versus analytic `±1`, maximum absolute discrepancy `1.29342e-5`. The test also checks this against a `2e-5` diagnostic tolerance.

### Memory and unsupported scope

For one synchronized CUDA call at `N=128,d=2,H=2,W=2`, incremental peak allocated memory was 21,504 bytes. The largest constructed window-face tensor had 2,048 float32 elements (8,192 bytes); frame faces had 1,024 elements. The vectorized temporary scales as `O(N·H·W·d)`. No latency benchmark or large-capacity stress test was performed. Numerically characterized capacities are only `d≤2,H≤2,W≤2`; larger capacities are not claimed unsupported by the API, but remain uncharacterized for performance/resource limits.

No sampled numerical failure was found. This small, single-machine exploration is not a universal hardware or numerical guarantee.

## 5. Proposed CP2 limits — pending explicit author signoff

These are recommendations only; they were not used to grant formal acceptance or classify the exploratory samples as passing acceptance. Before CP2, the author must explicitly lock or revise them and the confirmation set.

1. **Forward oracle:** at most **1 float32 ULP** from the correctly rounded high-precision oracle and `|error| ≤ 1e-7·max(1,|oracle|)` for every point/output in the declared tested range.
2. **Near-zero and sign:** exact zero status and sign class must match the oracle; no zero/sign crossing. Report all cases with `|oracle|≤1e-6` separately and require absolute error `≤1e-7` there.
3. **CPU/CUDA:** at most **1 ULP**, with exact zero/sign status agreement, on the locked device matrix.
4. **Unique-branch gradients:** absolute error `≤1e-6` per tested point/corner gradient versus independent analytic values; CPU/CUDA discrepancy `≤1e-6`.
5. **Tied gradients:** absolute error `≤1e-6` versus separately calculated PyTorch `amin`/`amax` equal-split expectations on explicit tie fixtures; report branch activity and cancellation separately.
6. **Finite differences:** maximum absolute discrepancy `≤2e-5` for the predeclared non-tie central probes at float32 step `1e-3`.

The largest-finite query demonstrates that the proposed scale-aware forward bound is preferable to a universal absolute tolerance. The accepted CP2 envelope and device matrix remain author decisions.

## 6. Separate deterministic CP2 confirmation plan (not executed)

Only after separate author numerical signoff:

- Freeze a confirmation manifest and SHA-256 **before** observing its outputs.
- Use an independent deterministic seed `20261113`: 16 new generated cases for each of the 12 `(d,H,W)` combinations (192 cases), using the same documented sorted-corner/non-simple-float32 construction but not the CP1 seed/data.
- Add 36 fixed below/on/above nextafter boundary probes (three per combination), plus 18 newly locked exact/tied/degenerate literal fixtures. Total proposed confirmation outputs: **246**.
- Run the frozen cases through the high-precision oracle and actual CPU plus the declared CUDA device(s); record forward abs/ULP, near-zero/sign/zero crossings, unique-branch and tied gradients, finite differences, and environment. Compare only against the author-locked limits; do not relax limits after seeing results.
- Run scoped/full regressions, collection/T3 accounting and teardown-protection checks. Do not extend the read-only Inspector until CP2 is separately authorized.

This confirmation set is only a proposal; it has not been generated, executed, or accepted.

## 7. Exclusions and stop status

No CP2, formal numerical acceptance, Inspector extension, trainable geometry, semantic class binding, loss/reducer, optimizer, checkpoint, learning/training, image integration, TOWN-L01, classifier reconnection, or research claim was started. `main` was not changed; no merge or independent acceptance is claimed. The exact Boolean membership APIs and accepted TOWN-02 semantics remain separate from signed clearance.

**CP1 status:** mathematical functions, scoped tests, exploratory CPU/CUDA measurements, and this evidence report are prepared for publication. Stop here for the author's separate numerical signoff. Proposed budgets, the separate CP2 confirmation set, and the impossible sign-disagreement diagnostic are the remaining explicit decision/limitation items; CP1 is not formal feature acceptance and TOWN-03 is not declared complete.
