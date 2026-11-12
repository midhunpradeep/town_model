# BOX-05 CP2/CP3 Implementation and Verification Report

**Status:** implementation candidate complete; independent fixed-SHA review pending. No optimizer, training loop, learning experiment, BOX-L01, TOWN-03, or TOWN-L01 was started.

- Repository: `midhunpradeep/town_model`
- Issue: [#24](https://github.com/midhunpradeep/town_model/issues/24)
- CP1 base: `b0efdbefd12b42f05c3eb9ceb6fcd4425d55d3c4`
- Accepted integration base: `404d76112a9d53282b3f55698b4decb705a5601c`
- Implementation source SHA verified below: `22c81a3dcdcbe26c71dc99f22383a095ffc7860c`
- This report is a documentation-only follow-up to that tested source SHA; the exact final handoff SHA is recorded in the issue journal.

## Authority and numerical contract

The CP2/CP3 implementation was authorized by the author’s numerical signoff [comment 6813022297](https://github.com/midhunpradeep/town_model/issues/24#issuecomment-6813022297). The only superseding numerical decision is the CPU/CUDA comparison revision in [comment 6814367422](https://github.com/midhunpradeep/town_model/issues/24#issuecomment-6814367422). It replaces the former 4-ULP CPU/CUDA forward criterion with:

- Per-endpoint CPU/CUDA absolute difference `<= 2^-21`.
- Additionally `<= 128 ULP` when both endpoints are outside `z=0.01`.
- Within/at the near-zero region, the absolute limit applies instead of a ULP gate; zero-status and sign differences are recorded and are allowed only there when the absolute limit passes.
- The same absolute/ULP comparison applies to inverse-output endpoints. This does not imply exact Boolean membership agreement across devices.

All other signed-off requirements remain unchanged: exact F12 width-first transform, float32 production geometry, non-toleranced hard physical invariants, target reconstruction and reserve budgets, raw range `[-3,3]`, initialization `sigma_min>=1e-5`, occupancy/aspect/center/RNG/retry policy, gradient limits, and the stop-on-any-failure rule. No transform or physical invariant was changed to address the earlier discrepancy.

## Implementation inventory

Changes relative to CP1 `b0efdb…`:

- `src/town_model/models/bounded_box_training_geometry.py` — typed inverse outcomes, endpoint budget/zero-crossing diagnostics, feasibility preflight, deterministic physical-first initializer, per-attempt records, complete-or-failure publication.
- `src/town_model/models/trainable_bounded_box.py` — minimal float32 `nn.Module` with trainable raw width/center Parameters, per-row ordinary/hosting policies, actual physical validation, diagnostics, immutable BOX-04 identity binding, and differentiable BOX-04 inference.
- `src/town_model/dashboard/box05_inspector.py` and `src/town_model/dashboard/app.py` — deterministic read-only BOX-05 fixture/panel within the existing Mathematical Inspector.
- `tests/test_box05_cp2_cp3.py` — inverse, typed failures/budgets, initialization/retry determinism, class-row independence, module/BOX-04 gradients, CPU/CUDA and 1,024-target confirmation cases.
- `tests/test_dashboard_app.py` — preserves previous Inspector assertions and verifies the added read-only panel.
- `tests/test_t3_architecture.py` — accounts for the new BOX-05 tests while preserving every historical, BOX/TOWN inventory assertion.

`docs/reconstruction/T3_TEST_RETIREMENT_MANIFEST.md` was not modified; SHA-256 remains `b3b283a4b7849cd5dd5d08b9aa9ecf66888f8435954479a6f3a7823a0802ded3`. No TOWN source or BOX-01/02/03/04 implementation was changed.

## Locked numerical confirmation

### Environment and reproduction

- Python 3.12.0; PyTorch `2.5.1+cu124`; NumPy `2.5.4`; CUDA 12.4.
- Actual synchronized accelerator: NVIDIA GeForce RTX 2060, driver `580.126.09`.
- Each listed seed generates 1,024 deterministic, independently sampled float32 physical target boxes, alternating requested occupancies 0.01/0.1 within the approved bands. The same targets/raw inverse values are verified on CPU and actual CUDA.
- Fresh post-revision seeds `2026111203` (ordinary) and `2026111204` (hosting) were added alongside the earlier reproducible seeds `2026111201` and `2026111202`.
- The tests use the pure CP1 decoder, strict independent physical validator, and actual BOX-03/04 inference kernels. CUDA calls are synchronized.

### CPU/CUDA endpoint comparison

Each case below has 4,096 endpoint scalars (1,024 boxes × 2 axes × lower/upper); `outside` is the count for which both compared magnitudes exceed `0.01`.

| Policy / seed | Outside comparisons | Max absolute difference | Max outside ULP | Outside ULP >128 | Zero-status / sign differences |
|---|---:|---:|---:|---:|---:|
| Ordinary / 2026111201 | 4,015 | `1.1920929e-7` | 40 | 0 | 0 / 0 |
| Ordinary / 2026111203 (fresh) | 4,004 | `1.1920929e-7` | 64 | 0 | 0 / 0 |
| Hosting / 2026111202 | 4,016 | `1.1920929e-7` | 64 | 0 | 0 / 0 |
| Hosting / 2026111204 (fresh) | 4,019 | `1.1920929e-7` | 64 | 0 | 0 / 0 |

The absolute cross-device limit is `2^-21 = 4.7683716e-7`; the largest observed difference is 25% of it. Across these cases the maximum outside-zone relative difference (denominator `max(abs(CPU),abs(CUDA))`) was `4.44e-6` ordinary and `4.72e-6` hosting. Near-zero ULP counts reached 4,096 ordinary / 8,192 hosting, but near-zero comparisons use absolute error: the largest observed CPU/CUDA near-zero difference was `7.45e-8`, below `2^-21`. Those ULP counts are not meaningful at/near zero. Relative differences are not gated near zero; their large values there are denominator-sensitive.

A separate targeted exact-zero lower-endpoint target reconstructed as `-7.4505806e-9` on both ordinary devices (one target-to-reconstruction zero-status difference per device, cross-device difference zero). For hosting, CPU reconstructed `0.0` versus CUDA `+1.4901161e-8`: one cross-device/target zero-status difference, no sign-bit disagreement, and absolute difference `1.49e-8`. These are within the revised near-zero absolute rule and are explicitly recorded. No formal 1,024-target batch had a cross-device zero-status or sign difference.

### Target reconstruction, reserves, conditioning, and physical boundaries

All actual CPU/CUDA decoded geometries passed the strict independent float32 physical validator; no tolerance relaxation, clamp, or repair was used.

| Policy | Worst observed target/reconstruction `E_inf` (CPU / CUDA) | Reserve limit | Worst reserve use | Remaining reserve at worst |
|---|---:|---:|---:|---:|
| Ordinary `epsilon=1e-4` | `8.94e-8 / 1.043e-7` | `3.125e-6` | 3.34% | `3.021e-6` |
| Hosting `epsilon=delta=1e-4` | `1.192e-7 / 1.192e-7` | `3.125e-6` | 3.81% | `3.006e-6` |

Across the four formal batches, the smallest measured local `sigma_min` was `0.0673`, above the initialization gate `1e-5`. Minimum full-width slack was `0.1049` ordinary / `0.1002` hosting; minimum domain slack was `0.1994` / `0.2271` respectively. Near-zero target-to-reconstruction errors were at most `8.29e-8`; outside-zone target/reconstruction ULP distance was at most 89, under its 128-ULP limit.

Boundary probes at raw `a,b` endpoints and adjacent in-range float32 values passed strict validation. Maximum CPU/CUDA difference was `1.1920929e-7`, at most 2 ULP, with maximum symmetric relative difference `1.20e-7`. The smallest domain slack was `2.345e-4` at raw `(3,3)` (`upper=0.9997655`); finite raw parameters in the certified range do not exactly attain the domain face. Exact physical face-touch targets remain finite-real-unattainable; the nextafter-inside face target is correctly classified outside the certified raw operating range.

### Inverse/initializer, gradients, and Boolean boundary behavior

- The inverse exposes distinct typed outcomes for invalid physical targets, finite-real unattainability, float32 representability, reconstruction-budget failure, invalid reconstructed geometry, and unsupported numerical configuration. Tests exercise invalid targets, finite-real unattainability, endpoint-budget rejection, and unsupported configurations. No float32-representability or invalid-reconstructed-geometry failure occurred in the locked formal batches; both remain separate classified paths. No partial batch is returned on failure.
- Initialization uses private SHA-256-derived NumPy PCG64 streams keyed by `(master_seed,class_id,attempt,field)`, NumPy `2.5.4`; preflight, at most 64 attempts/class, and complete initialization or typed failure are tested. A deterministic injected RNG-boundary test exercises retry and exhaustion paths without exposing partial parameters. Repeated seeds reproduce raw tensors and reordering class rows preserves each class’s independent sample.
- The fixed Inspector initializer fixture (seed `20261112`) accepted both rows on attempt 1: class 12 ordinary `q_actual=0.0068971`, aspect `3.0602`, `sigma_min=0.06387`; class 27 hosting `q_actual=0.0959477`, aspect `3.1247`, `sigma_min=0.20352`.
- Analytic/autograd maximum difference was `2.98e-8` on CPU and CUDA. The float64 central-difference diagnostic (step `1e-4`) maximum error was `1.9972e-9` against the approved `1e-7` limit. CPU/CUDA local-gradient max difference was checked against `1e-6`, `rtol=0`.
- Actual BOX-03 signed-clearance differentiation on 1,024 interior points per fresh configuration had no active-face disagreements; max CPU/CUDA raw-width gradient difference `5.96e-8`, raw-center `1.192e-7`, point-gradient difference `0`, and nonzero raw gradients on every row. Independent Decimal80 endpoint-reference checks were diagnostic-only; their maximum observed endpoint absolute discrepancy across the four seed/configuration batches was at most `1.17e-7`. Active-face, tied-face, and independent class-row isolation tests pass.
- Actual Boolean membership was compared independently. All center queries agreed (1,024/1,024 inside on both devices). Constructed exact-face queries did show cross-device Boolean disagreements: for fresh ordinary cases, 924/4,096 CPU-face→CUDA and 805/4,096 CUDA-face→CPU; for fresh hosting, 900/4,096 and 803/4,096. These are exact-boundary queries against slightly different device endpoints, not interior discrepancies. BOX-04 exact Boolean semantics were not modified; approximate endpoint agreement is not asserted to imply cross-device set equality.

## Regression and integrity verification

All commands ran on implementation source SHA `22c81a3dcdcbe26c71dc99f22383a095ffc7860c` before this report-only commit:

- `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q tests/test_box05_cp2_cp3.py tests/test_t3_architecture.py` — **26 passed**.
- `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q` — **625 passed, 0 failed, 0 skipped**, 8 existing non-failing Streamlit/Pandas `Pandas4Warning` warnings.
- `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest --collect-only -q` — **625 collected**.
- `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m compileall -q src/town_model tests` — passed.
- `/home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pip check` — no broken requirements.
- `git diff --check 404d76112a9d53282b3f55698b4decb705a5601c..22c81a3dcdcbe26c71dc99f22383a095ffc7860c` — passed.
- T3 architecture accounting passed in the full suite; the retirement manifest remains byte-identical. BOX-01/02/03/04, TOWN-01/02, Mathematical/Town Inspectors, and T0–T5 classifier-teardown protections are included in the passing full suite.
- `main`, `origin/main`, and live GitHub `main` remain `404d76112a9d53282b3f55698b4decb705a5601c`; no integration/merge was performed.

## Limitations and handoff

The revised cross-device comparison was verified on one GPU/software stack. Wider dimensions/configurations retain CP1 exploratory status. Exact-boundary Boolean membership can differ across device-specific physical endpoints even when the revised endpoint limits pass; each device’s Boolean evaluation remains exact for its own geometry. No learning performance or scientific learning claim is made.

**Independent review is pending.** This report does not accept or merge the work. The implementation source SHA and report-only follow-up SHA, complete changed-file inventory, numerical author-signoff links, verification evidence, and review request are recorded in the final issue handoff.
