# BOX-05 CP1 — Exploratory Numerical Characterization

**Status: CP1 evidence only. This is not formal numerical feature acceptance, an approved operating envelope, or a scientific claim.**

The implementation and probe are on the isolated BOX-05 worktree. This report is the CP1 stop handoff for [issue #24](https://github.com/midhunpradeep/town_model/issues/24). The author’s first approval is recorded in [comment 6811877358](https://github.com/midhunpradeep/town_model/issues/24#issuecomment-6811877358). CP2, CP3, and formal numerical acceptance remain unauthorized until the author posts a separate numerical signoff on #24.

## 1. Authority and exact CP1 implementation

The mathematical source is original A1 **F7.1–F7.8** and the **grouped F12.1–F12.8** geometry approval in `docs/reconstruction/A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md` §2. F12.1–F12.8 remain a grouped block; this report makes no per-ID mapping. Repository documentation acceptance is not proof of a numerical envelope.

For finite float32 raw values `a,b`, domain `[-1,1]^d`, and configured `0 < w < 2`, the implementation applies exactly:

```text
h = [w + (2 - w) sigmoid(a)] / 2
c = (1 - h) tanh(b)
lower = c - h
upper = c + h
```

The ordinary policy is `w=epsilon`; the hosting-frame width policy is `w=epsilon+2*delta`. The configured starting values `epsilon=delta=1e-4` were not changed. No clipping, physical repair, alternative transform, inverse API, initializer, parameter module, optimizer, loss, or training path was added.

Files added:

- `src/town_model/models/bounded_trainable_box_geometry.py` — pure decoder, named minimum-width policies, independent physical validator, and analytic endpoint Jacobian.
- `tests/test_box05_cp1_geometry.py` — CP1 correctness/derivative smoke tests, explicitly not feature-acceptance tests.
- `tools/box05_cp1_exploratory_probe.py` — deterministic exploratory probe and private diagnostic-only inverse calculations. The inverse helper is not production code or a verified inverse API.
- `docs/reconstruction/BOX05_CP1_PROBE_RESULTS.json.gz` — exact raw inputs, computed physical outputs, per-device diagnostics, boundary records, and inverse samples.
- `tests/test_t3_architecture.py` — narrowly adds separate accounting for 16 CP1 smoke-test nodes. The T3 retirement manifest and prior BOX/TOWN/historical assertions are unchanged.

The validator checks the exact real values represented by stored float32 centers, half-widths, and corners against the configured Python `w` by promoting only for validation. It independently checks finiteness, corner/center/half-width consistency, ordered corners, domain bounds, `h >= w/2`, and exact represented endpoint width `upper-lower >= w`. It uses no tolerance and performs no repair. Saturation/weak gradients are diagnostic state, not a physical-invalidity criterion.

## 2. Reproduction, coverage, and environment

From the CP1 worktree:

```bash
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python \
  tools/box05_cp1_exploratory_probe.py \
  --output docs/reconstruction/BOX05_CP1_PROBE_RESULTS.json.gz

gzip -t docs/reconstruction/BOX05_CP1_PROBE_RESULTS.json.gz
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python \
  -m pytest -q -rs tests/test_box05_cp1_geometry.py
```

The probe records exact inputs and actual centers, half-widths, corners, reserves, derivative/conditioning summaries, and higher-precision-reference differences. Its predeclared schedule is **sparse, not Cartesian**:

- `d ∈ {1,2,4,8,16,32,64}`; representative `C ∈ {0,1,2,16,60}`.
- Ordinary `epsilon ∈ {1e-5,1e-4,1e-3,1e-2}`; hosting `(epsilon,delta) ∈ {(1e-4,1e-4),(1e-5,1e-4),(1e-4,1e-5),(1e-3,1e-3),(1e-3,1e-2)}`.
- 17 required raw probes `{-40,-24,-20,-16,-12,-8,-4,-1,0,1,4,8,12,16,20,24,40}` are rotated through every coordinate in mixed/asymmetric `C=2` cases for each dimension under both default policies.
- All five class counts are measured at `d=2` under both default policies. The seven remaining width configurations are assigned fixed representative `(d,C)` cases. A near-capacity `w=nextafter(2,0)` case and explicit invalid scalar policies are separate.
- The fixed characterization has **256 core cases**, followed by 240 minimum-width nextafter probes, 144 sigmoid/tanh saturation-boundary geometry probes, 14 seeded occupancy/anisotropy probes (`C=60` per mode/dimension), and 1,206 physical-target inverse diagnostics (nine width configurations × 128 seeded random targets plus six fixed cases each).
- The float64 reference, inverse diagnostics, and random inputs are exploratory oracles only. The seeded probe uses private RNG state and records seed/version; no cross-version bitwise RNG guarantee is claimed.

Environment: Python 3.12.0; PyTorch 2.5.1+cu124; CUDA 12.4; NumPy 2.5.4; NVIDIA GeForce RTX 2060, compute capability 7.5, driver 580.126.09. CUDA was available and actually measured; this is one GPU/software stack, not a universal device guarantee.

The executable result artifact can be inspected with:

```bash
gzip -dc docs/reconstruction/BOX05_CP1_PROBE_RESULTS.json.gz | python -m json.tool
```

## 3. Measured physical outcomes and finite-precision reserves

**Core grid:** on both CPU and CUDA, 69 of 256 batched cases contained at least one physically invalid row (72 invalid rows total per device). Every such row was rejected for the strict half-width floor, primarily the ordinary `epsilon=1e-4` case when float32 rounded the minimum half-width below the exact configured value. No core case violated the physical domain/order constraints. The minimum observed domain slack was zero (exact boundary touch), not negative.

A representative actual result at ordinary `w=1e-4`, `a=-40`, `b=0` is:

```text
stored h = 4.999999873689376e-5 < w/2 = 5e-5
stored endpoint width = 9.999999747378752e-5 < w = 1e-4
```

The exact real-valued equation has the width floor; this actual float32 state does not. The decoder leaves it untouched and the independent validator rejects it. The current report does not silently replace `epsilon`, round the configured floor down, or repair the corners.

The adjacent-float32 search for `b=0` found these first valid raw-width values in the bracket `a ∈ [-40,0]`; “valid at -40” means no invalid-to-valid transition was present in that bracket:

| Policy/configuration | `w` | CPU first valid `a` | CUDA first valid `a` |
|---|---:|---:|---:|
| Ordinary `epsilon=1e-5` | `1e-5` | `-29.1121749878` | `-29.1121749878` |
| Ordinary `epsilon=1e-4` | `1e-4` | `-27.0326900482` | `-27.0326881409` |
| Ordinary `epsilon=1e-3` | `1e-3` | valid at `-40` | valid at `-40` |
| Ordinary `epsilon=1e-2` | `1e-2` | `-22.1756954193` | `-22.1756954193` |
| Hosting `(1e-4,1e-4)` | `0.00030000000000000003` | valid at `-40` | valid at `-40` |
| Hosting `(1e-5,1e-4)` | `0.00021` | `-26.3394870758` | `-26.3394870758` |
| Hosting `(1e-4,1e-5)` | `0.00012` | `-27.0326786041` | `-27.0326786041` |
| Hosting `(1e-3,1e-3)` | `0.003` | valid at `-40` | valid at `-40` |
| Hosting `(1e-3,1e-2)` | `0.021` | `-21.4770069122` | `-21.4770069122` |

The boundary refinement also varied `b ∈ {0,±4,±8}`. It found **8 stored endpoint-width floor failures per device** even where the half-width floor passed; e.g. ordinary `w=0.01`, `a=-22.1756954193`, `b=-4` had endpoint reserve slack `-9.5367431643e-9`. Across the 120 probes/device, 38 were invalid: 30 half-width failures and 8 endpoint-width failures. This demonstrates that a half-width-only check is insufficient.

The most negative endpoint-width reserve slack in the core default ordinary grid was `-2.5262124884e-12` at `w=1e-4`. The target/decoder report keeps the exact physical slack separate from the boolean validation result. No domain or width inequality is relaxed.

**Decision implication:** the starting constants remain as approved, but `a=-40` and other low-`a` values are not universally valid stored geometry for all listed `w`. The author must decide which finite-precision operating envelope is supported and how out-of-envelope raw states are classified. No clamp or substitute transform is proposed.

## 4. Sigmoid/tanh saturation and CPU/CUDA behavior

Targeted ordered-float32 binary searches plus adjacent `nextafter` checks measured:

| Transition | CPU raw | CUDA raw | Observed activation/derivative behavior |
|---|---:|---:|---|
| Sigmoid first nonzero | `-88.7228317261` | same | Predecessor output/derivative `0`; first nonzero output about `2.93876e-39`. |
| Sigmoid first exactly `1` | `16.6355323792` | same | Predecessor `p=0.9999998808`, derivative `1.19209e-7`; transition `p=1`, derivative `0`. |
| Tanh first above `-1` | `-9.0108394623` | `-9.0109128952` | Predecessor `u=-1`, derivative `0`; transition `u=-0.9999999404`, derivative `1.19209e-7`. |
| Tanh first exactly `+1` | `9.01084041595` | `9.0109138489` | Predecessor `u=0.9999999404`, derivative `1.19209e-7`; transition `u=1`, derivative `0`. |

The CPU/CUDA tanh transition differed by 77 ordered-float32 ULPs on this stack. Fifteen of 72 activation-boundary geometry probes/device were physically invalid due to the same minimum-half-width issue; there were no domain-bound violations. Saturated/zero derivatives were recorded separately from physical validity.

Across all 256 core CPU/CUDA comparisons, maximum absolute differences were `5.9604645e-8` for centers/half-widths and `1.1920929e-7` for endpoints; maximum endpoint distance was 4 ULP (center maximum 5 ULP). There were no observed endpoint zero-status or sign-bit mismatches. These are observations, not accepted cross-device limits.

## 5. Inverse representability, endpoint errors, ULP, and zero crossing

No production inverse was implemented. For the requested exploratory evidence only, the probe evaluates the issue’s inverse equations in float64 against 1,206 independently specified physical endpoint targets, casts candidate raw values to float32, decodes on CPU/CUDA, and independently checks the actual reconstructed physical state. It does not clamp, repair, substitute large finite raw values, or apply a reconstruction pass/fail budget.

Observed target categories:

- 6 fixed minimum-width endpoint targets were **invalid physical targets** when their exact stored-float32 width was below the configured real `w`.
- 27 physically valid fixed targets were finite-real unattainable: full-domain capacity (`t=1`) and exact domain-touching centers (`|s|=1`).
- 1,173 targets had finite-real candidates and finite float32 candidate raws. All resulting CPU/CUDA reconstructions in that category passed the strict physical validator in this sample.
- No float32 inverse candidate was nonfinite; no reconstructed hard-invariant failure was observed. These counts are sampled diagnostic outcomes, not completeness proof.

For the **1,152 random interior targets** (128 per width configuration), endpoint `E_inf` quantiles were:

| Device | p50 | p95 | p99 | maximum |
|---|---:|---:|---:|---:|
| CPU | `2.9802322e-8` | `8.9406967e-8` | `1.1920929e-7` | `1.7881393e-7` |
| CUDA | `4.4703484e-8` | `8.9406967e-8` | `1.1920929e-7` | `1.7881393e-7` |

The maximum observed `E_inf/epsilon` was `0.0163913`; the maximum `E_inf/delta` for hosting frames was `0.0178814`. No author-approved budget was applied. For endpoints with `|target| > 0.01`, maximum target/reconstruction distance was 68 ULP in the random samples; for `|target| > 0.05`, it was 17 ULP. CPU/CUDA inverse endpoint outputs differed by at most `2.3841858e-7` absolute; the maximum ULP distance occurred near zero (256 ULP), with no observed sign or zero-status mismatch.

An exact zero target endpoint sometimes reconstructed as a small nonzero endpoint (`about 4.47e-8` to `7.45e-8` in examples). Its ULP distance from zero was hundreds of millions because ULP counts are not informative at zero. Some other fixed configurations reconstructed exact zero. Therefore signed-zero/zero-crossing treatment needs an explicit near-zero absolute rule; endpoint approximation does not establish exact set membership. All such cases remain budget-unassessed.

## 6. Analytic Jacobian, independent derivative checks, and conditioning

The analytic per-coordinate Jacobian was implemented in issue-defined order:

```text
k = (2-w)/2; p=sigmoid(a); u=tanh(b)
h_a = k p(1-p)
c_a = -u h_a
c_b = (1-h)(1-u^2)
J = [[-(1+u)h_a, c_b], [(1-u)h_a, c_b]]
det(J) = -2 h_a c_b
```

Across the core and seeded occupancy probes, maximum analytic/autograd absolute discrepancy was `8.9406967e-8` on both devices; maximum determinant/formula discrepancy was `2.40e-8`. Independent float64 central differences over moderate `a,b ∈ {-4,-1,0,1,4}` had maximum absolute error `3.2729e-9` at step `1e-4` and `3.2729e-7` at step `1e-3`. The CP1 smoke test currently asserts an exploratory `atol=2e-6, rtol=0` autograd comparison; that test bound is **not** an approved acceptance budget.

Conditioning convention: singular values of the local `(a,b) → (lower,upper)` Jacobian, with raw coordinates unscaled and endpoint coordinates in normalized domain units. Among 27,281 measured local maps (core plus seeded occupancy), 6,370 had `sigma_min=0` and infinite condition number, primarily saturated probes. Overall `sigma_min` p50 was `0.08160`; the condition-number p50 was `2.8043`, while the long tail reached infinity. This is valid-but-weak/saturated differential behavior, not automatically invalid geometry.

Within the sampled `|a|≤3, |b|≤3` subset (1,108 local maps), minimum observed `sigma_min` was `0.01912` and maximum observed condition number was `20.01`. Separately, a conservative analytic bound can be derived for the exact Jacobian over this raw rectangle and all tested `w≤0.021`: `sigma_min ≥ 2.91e-5`. This uses `sigma_min=|det J|/sigma_max`, lower bounds on `h_a,c_b`, and `sigma_max≤||J||_F`; it is an analytic argument for the real formula, not a float32 implementation proof.

## 7. Seeded occupancy and anisotropy observations

These are **raw-normal exploratory probes, not initializers**: 60 rows per dimension/mode, private CPU `torch.Generator`, seed derived from SHA-256 of base seed `20261112` and case ID, then measured on CPU/CUDA. All 14 batches passed the physical validator. Occupancy is `q=prod_j h_j`, summarized through `log(q)=sum_j log(h_j)` to avoid dimension-driven underflow.

| `d` | Median `log(q)`, ordinary | Median `log(q)`, hosting | Anisotropy p95, ordinary | Anisotropy p95, hosting |
|---:|---:|---:|---:|---:|
| 1 | -0.753 | -0.555 | 1.00 | 1.00 |
| 2 | -1.277 | -1.349 | 3.56 | 4.09 |
| 4 | -2.976 | -3.134 | 5.29 | 7.60 |
| 8 | -6.193 | -6.144 | 9.57 | 9.63 |
| 16 | -12.715 | -13.259 | 12.26 | 11.09 |
| 32 | -26.453 | -25.817 | 13.99 | 20.32 |
| 64 | -52.440 | -51.365 | 27.59 | 16.75 |

Maximum observed anisotropy ratio was 45.55 (hosting, `d=32`). The probe is not a feasibility or sampler-success study; it shows that fixed raw-normal draws produce sharply dimension-dependent occupancy and potentially high anisotropy. It does not select scientific occupancy bands.

## 8. Proposed numeric choices for author review — none are approved yet

These are engineering recommendations based only on the measurements and the stated analytic bound. They are **proposals**, not accepted limits. The author may revise or reject every value.

| Decision requiring explicit signoff | CP1 recommendation |
|---|---|
| Discrete tested dimensions/counts/configurations | Initially support only `d ∈ {1,2,4,8,16,32,64}`, `C ∈ {0,1,2,16,60}` and the exact nine epsilon/frame settings enumerated in §2. Treat all other values as uncharacterized until separately confirmed; do not claim a continuous envelope from the grid. |
| Initialization raw range / `U_max` | Candidate initialization rectangle `a,b ∈ [-3,3]`, `U_max=3`, for the local Jacobian gate only. This is **not clipping** and does not imply runtime states outside are physically invalid. The author must say whether `U_max` is an initializer-only or broader supported-operating-range rule, and specify out-of-range disposition. |
| Endpoint ULP and near-zero policy | Candidate near-zero threshold `z=0.01`; outside it, endpoint reconstruction `U_max=128` ULP. If either target or reconstruction is within `z`, use the strict absolute endpoint error below; continue enforcing reserve-relative error everywhere. Candidate CPU/CUDA forward endpoint comparison is at most 4 ULP outside the zone, with exact zero-status/sign agreement. Probe maxima were 4 ULP forward and 68 ULP target/reconstruction outside `z`. |
| Signed zero / zero crossing | Ignore only the sign bit of exact zero as a physical mismatch; record it diagnostically. Inside the near-zero zone use the absolute endpoint limit and reserve-relative limit, not ULP count; crossing zero is not independently accepted as exact membership. |
| Near-zero absolute endpoint error | Candidate `E_inf ≤ 2^-20 = 9.5367431640625e-7` in the near-zero zone, in addition to reserve-relative limits. The strict reserve rule may be tighter. |
| Reserve-relative reconstruction budgets | Candidate `kappa_epsilon=kappa_delta=1/32=0.03125`, so `E_inf≤kappa_epsilon*epsilon` and additionally `E_inf≤kappa_delta*delta` for hosting frames, everywhere. The maximum observed ratios were about 0.0164 and 0.0179 respectively; sampled evidence is not validation. |
| Autograd/gradient numerical budgets | Candidate analytic/autograd and CPU/CUDA local-gradient max absolute difference `≤1e-6`, `rtol=0`; candidate meaningful central-difference check uses float64 step `1e-4` and absolute error `≤1e-7`. Current exploratory maxima were `8.94e-8` autograd and `3.28e-9` finite difference at that step. |
| Conditioning definition and `gamma` | Use the unscaled local Jacobian convention in §6 and candidate initialization gate `sigma_min(J)≥gamma=1e-5` over `a,b∈[-3,3]`. The conservative formula bound was `2.91e-5` for tested `w` values; the author must approve both scaling and gate. This is an initialization gate only. |
| Occupancy bands | Candidate target occupancies and bands `[q/2,2q]` (clipped only at the mathematical `q≤1` endpoint), preflighted for feasibility under the approved raw range: `d=1:{0.1}`; `d=2:{0.01,0.1}`; `d=4,8,16,32:{0.001,0.01,0.1}`; `d=64:{0.001,0.01}`. The unlisted exploratory `{0.001,0.01,0.1}` targets are not asserted feasible for every dimension/range. |
| Anisotropy / center distribution | Candidate maximum aspect ratio `max(h)/min(h)≤4`; center fraction `s=tanh(b)` sampled independently and label-agnostically from `Uniform[-0.5,0.5]` per coordinate. Candidate anisotropy generation uses zero-mean log-width perturbations with max/min ratio `≤4`, then exact feasibility preflight; no silent adaptation. |
| RNG / retries | Candidate private per-row/per-attempt RNG stream seeded from the first 64 bits of SHA-256 over `(master_seed,class_id,attempt,field)` and NumPy `PCG64`, recording algorithm/software/seed. Claim determinism only for pinned software; no cross-version bitwise guarantee. Candidate bounded retry budget: 64 attempts/class, with impossible-target preflight distinct from exhaustion. Neither policy is approved. |
| Confirmation sample | Candidate fixed confirmation rerun: all dimensions/counts/config points in §2; each of the nine width settings with 1,024 seeded physical endpoint targets; exact minimum-width, full-domain, face-touch, zero-endpoint, signed-zero, and nextafter neighbors; CPU and actual CUDA; gradient/conditioning cases on and immediately outside proposed limits. This is a proposed plan, not an executed result. |
| Failure behavior | Keep exact stored-geometry validity independent from gradient conditioning. Reject/report any actual invariant violation before publication; do not clamp or repair. Keep finite-real-unattainable, float32 nonrepresentability, reconstruction-budget miss, hard-invariant failure, and unsupported-envelope outcomes distinct. The author must approve what future runtime/integration does for finite raw values outside the supported envelope. |

## 9. Unresolved decisions and required second author signoff

The author’s second issue comment must approve or revise **each** of the numeric choices in §8, including:

1. Exact discrete/continuous supported dimensions, capacities, epsilon/delta settings, raw range, `U_max`, and whether those raw limits govern only initialization or all supported runtime states.
2. Near-zero threshold `z`, absolute endpoint limit, `U_max` ULP gate, signed-zero and zero-crossing policy, endpoint/center/half-width reporting, CPU/CUDA comparison rules, and `kappa_epsilon` / `kappa_delta`.
3. Analytic/autograd/finite-difference/CPU-CUDA gradient budgets, Jacobian scaling convention, conditioning definition, and `gamma`.
4. Dimension-specific occupancy targets/bands, aspect-ratio law, center distribution, seed derivation/RNG/version guarantee, retry count, preflight rules, and complete-set failure/publication behavior.
5. The confirmation cases and sample counts, plus disposition for the measured ordinary `epsilon=1e-4` low-`a` floor failures and hosting/ordinary endpoint-width failures. The report offers no alternate transform or default change.

The previously floated `1/8 ULP`, `1% reserve`, and `32 retries/class` values are **not** adopted as approved tolerances. No numerical recommendation in this report becomes policy through implementation, test assertions, or silence.

## 10. Limitations and CP1 stop boundary

- A finite grid and finite random sample do not prove continuous validity, a universal operating range, mathematical superiority, or learning behavior.
- CPU/CUDA evidence covers one RTX 2060/PyTorch stack only. The float64 reference and diagnostic inverse are not production algorithms.
- The focused test’s `2e-6` autograd assertion is an exploratory smoke bound, not the requested author-approved budget.
- No production inverse, initializer, trainable `Parameter` module, BOX-04 geometry integration, Inspector extension, optimizer/loss, training, or formal numerical feature-acceptance test was implemented or run.
- `docs/reconstruction/T3_TEST_RETIREMENT_MANIFEST.md` was not changed. TOWN-02 #22 source paths and remote branches were not modified. CP1 probes were measured from the claim base `c01dc2124b6523bbdb07e4a8f97c23395996eea0`; TOWN-02 was integrated to `main` during this task, and the CP1 branch’s narrow T3 accounting reconciliation is recorded in the issue journal. The final CP1 commit parent is reported in the handoff, not inferred from this probe report.

**Mandatory stop:** report the measured CP1 evidence and proposed numeric decisions on issue #24, keep #24 open and this worktree intact, request the author’s second numerical signoff, and do not begin CP2/CP3 or formal feature acceptance before that exact-issue approval.
