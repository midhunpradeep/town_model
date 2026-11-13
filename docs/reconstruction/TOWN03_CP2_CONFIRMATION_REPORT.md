# TOWN-03 CP2 Confirmation and Inspector Report

> **CP2 execution finished.** This report requests independent fixed-SHA technical review and subsequent author visual acceptance. It is not itself independent acceptance. Do not merge to `main`, start TOWN-L01, or claim TOWN-03 complete.

## 1. Authorization and revisions

- Direct author CP2 numerical signoff and authorization: [issue #23 comment 6814259456](https://github.com/midhunpradeep/town_model/issues/23#issuecomment-6814259456). The approved budgets and confirmation plan were followed without modification.
- CP1 evidence revision reviewed for authorization: `9653f628a2163273896044c9549362701bf76052`.
- CP2 confirmation set was frozen before evaluator execution in commit `af58270e643dacf2110ce5932fb6533c0e8aee55`.
- **CP2 implementation SHA for independent review:** `258aafa6399adff0d106f8a2dfd58db58f862ad5`.
- Original starting `main`: `404d76112a9d53282b3f55698b4decb705a5601c`.
- Branch/worktree: `implementation/town-03-cp1-signed-clearance`, `/tmp/town-model-town03-cp1`.

## 2. Frozen numerical confirmation set

- Manifest: [`TOWN03_CP2_CONFIRMATION_SET.json`](TOWN03_CP2_CONFIRMATION_SET.json).
- SHA-256: `acb1f9c43110ac839530066ef2aa3311755003321b945b65345f3445b4548f3e`.
- Confirmation groups: 192 generated cases (16 for each of the 12 `(d,H,W)` combinations, seed `20261113`), 36 nextafter boundary probes, and 18 independently derived literal/tie/degenerate fixtures; **246 total**.
- Each case stores its actual float32 inputs, independently calculated high-precision Decimal clearance, correctly rounded float32 oracle, and independent direct Boolean membership expectation. The freeze generator does not import or call the production evaluators.

The manifest was committed and its hash recorded on #23 before running the confirmation runner. The runner verifies that hash and re-derives the Decimal and Boolean expectations from stored inputs before comparing production outputs.

## 3. Numerical confirmation results

Runner: `tools/town03_cp2_verify_confirmation_set.py`; machine-readable results: [`TOWN03_CP2_CONFIRMATION_RESULTS.json`](TOWN03_CP2_CONFIRMATION_RESULTS.json), SHA-256 `2adc4a00ae12fc26a40306bf5dbe05a121813a93d38d8b25f438f959c7cd1188`.

**Result: PASS against every author-locked CP2 limit.** The locked manifest produced 246 house and 246 town outputs on each device. The runner recorded no failures and zero independent Boolean mismatches.

| Measurement | House | Town | Locked limit |
|---|---:|---:|---:|
| CPU max absolute error vs Decimal oracle | `5.960464477539063e-08` | `5.960464477539063e-08` | scaled error ≤ `1e-7` |
| CPU maximum ULP error | 0 | 0 | ≤1 ULP |
| CPU maximum scaled error | `5.893974124628436e-08` | `4.606144476972525e-08` | ≤`1e-7` |
| Near-zero outputs (`|oracle|≤1e-6`) | 31 | 31 | absolute error ≤`1e-7` |
| Near-zero maximum absolute error | 0 | 0 | ≤`1e-7` |
| Sign crossings / zero-status mismatches | 0 / 0 | 0 / 0 | 0 / 0 |
| CPU/CUDA maximum absolute difference | 0 | 0 | ≤1 ULP |
| CPU/CUDA maximum ULP difference | 0 | 0 | ≤1 ULP |

Unique-branch point and physical-corner gradients matched independent analytic expectations exactly on CPU and CUDA (maximum discrepancy 0). Selected native `amin`/`amax` tie gradients also matched their independent equal-split expectations exactly; CPU/CUDA gradient difference was 0. The central non-tie float32 finite-difference probes at step `0.0010000000474974513` had maximum absolute discrepancy `1.2934207916259766e-05`, within the locked `2e-5` limit. Detailed per-case and gradient observations are in the results artifact.

Environment: Python 3.12.0, PyTorch 2.5.1+cu124, Linux x86_64; actual synchronized CUDA on NVIDIA GeForce RTX 2060, capability 7.5, driver 580.126.09, CUDA 12.4.

## 4. Bounded read-only Town Inspector extension

Extended only the standalone `src/town_model/dashboard/town_math_inspector.py`. Its original TOWN-01 strict-interior section and five TOWN-02 Boolean composition scenarios remain. A new fixed-fixture TOWN-03 section shows five cases covering retained frame/window boundaries, hole interiors, a degenerate window, a collapsed frame, two-house composition, and overlapping windows.

For each point, the Inspector separately displays literal expected and evaluator actual Boolean membership and signed clearance. It evaluates the accepted Boolean APIs independently; clearance is never converted to membership. The `clearance ≥ 0` column is labeled diagnostic, and the `clearance > 0` column explicitly illustrates why strict-positive thresholding disagrees at included zero-clearance boundaries. Deliberately incorrect score injection produces scenario/point-specific errors. The page adds no controls, data loading, training, or artifact writes.

- Inspector/AppTest/T3 focused selection — **19 passed, 0 failed, 0 skipped**.
- Live standalone Streamlit: `/_stcore/health` returned **200 `ok`**, `/` returned **200**; process stopped.
- The author's visual inspection is **still pending** and is required after independent technical review.

## 5. Regression, architecture, and protected boundaries

- Full repository: `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q -rs` — **632 passed, 0 failed, 0 skipped**, eight pre-existing Streamlit/Pandas deprecation warnings; 77.93 seconds.
- Focused CP2/Inspector/T3: `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q -rs tests/test_town03_cp2_confirmation.py tests/test_town_math_inspector.py tests/test_t3_architecture.py` — **19 passed, 0 failed, 0 skipped**.
- Full collection: `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest --collect-only -q -p no:cacheprovider tests` — **632 collected**.
- `compileall` passed; `pip check` reports no broken requirements; `git diff --check` passed.
- T3 accounting separates 36 TOWN-03 mathematical tests, 6 CP2 confirmation tests, and 3 Inspector tests while preserving existing BOX/TOWN accounting. `docs/reconstruction/T3_TEST_RETIREMENT_MANIFEST.md` is unchanged, SHA-256 `b3b283a4b7849cd5dd5d08b9aa9ecf66888f8435954479a6f3a7823a0802ded3`.

## 6. Scope and limitations

The CP2 confirmation covers only the frozen 246-case set, its declared dimensions/capacities, and the recorded software/device environment. It does not establish universal hardware behavior or large-capacity performance. The Inspector is a fixed, read-only diagnostic, not a classifier integration or a general numerical debugging UI.

The previously reported mathematical limitation remains: under the approved equations, nonnegative signed clearance agrees with exact Boolean membership. A sign-disagreement fixture would contradict the approved mathematics. The Inspector therefore shows boundary differences for a **strict-positive** diagnostic while keeping exact membership independently evaluated.

No trainable geometry, optimizer, objective, learning, classifier integration, TOWN-L01, or research claim was implemented. No merge or change to `main` was made.

## 7. Handoff and stop

CP2 confirmation and the bounded Inspector work are submitted at implementation SHA `258aafa6399adff0d106f8a2dfd58db58f862ad5`. Please conduct an **independent fixed-SHA technical review** of the code, frozen manifest/hash, locked numerical results, Inspector behavior, regressions, T3 accounting, and teardown protections. After that review, the author must perform visual acceptance of the Inspector. Any failed review or visual request returns to the implementer; no integration is authorized by this report.

**Stop here:** no merge into `main`, no TOWN-L01, no training, and no declaration that TOWN-03 is complete.
