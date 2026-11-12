# TOWN-03 CP2 input-independence remediation report

> **Status: implementation submitted for independent fixed-SHA re-review.** This report does not claim acceptance. Preserve the open author visual-acceptance and integration gates. Do not merge, close issue #23, or start TOWN-L01.

## 1. Authorization, finding, and revisions

- Direct author remediation authorization: [issue #23 comment 6830101628](https://github.com/midhunpradeep/town_model/issues/23#issuecomment-6830101628).
- Independent review finding R1: [issue #23 comment 6815233836](https://github.com/midhunpradeep/town_model/issues/23#issuecomment-6815233836), which found 50 exact input repeats in the original frozen CP2 set.
- Remediation implementation SHA: `ff7717796226a5cb130785ef81001b576d773da1`.
- Pre-evaluation supplement-freeze commit: `9f69ac9e55ca1bae64e06dadd6d89fc522e1aa60`, on `remediation/town03-cp2-independent-inputs`.
- Worktree: `/tmp/town-model-town03-r1-remediation`.
- Reviewed implementation/evidence base: `9c2106274a3bd4af80397a110b6e4981c3654aab`.

This remediation changes evidence generation, verification, tests, and T3 node accounting only. It does **not** change the mathematical clearance functions, Boolean membership functions, or Town Mathematical Inspector.

## 2. Frozen supplement and independent disjointness audit

The prior CP2 artifacts remain unchanged:

| Artifact | SHA-256 |
|---|---|
| Original CP2 manifest | `acb1f9c43110ac839530066ef2aa3311755003321b945b65345f3445b4548f3e` |
| Original CP2 numerical results | `2adc4a00ae12fc26a40306bf5dbe05a121813a93d38d8b25f438f959c7cd1188` |
| CP1 exploratory results | `7e31ffeaac15b704e3b582f84e4ef1fd5903591380a54dc98895f36a626457c2` |

New, separately versioned inputs were generated without calling a production evaluator and pushed **before** numerical evaluation:

- Supplement: [`TOWN03_CP2_INDEPENDENCE_SUPPLEMENT_V1.json`](TOWN03_CP2_INDEPENDENCE_SUPPLEMENT_V1.json), SHA-256 `b53625dcc7eb428ef02cca5dbba48831b0c8ca9fff5182e98df558d48867eafa`.
- Pre-evaluation audit: [`TOWN03_CP2_INPUT_INDEPENDENCE_AUDIT_V1.json`](TOWN03_CP2_INPUT_INDEPENDENCE_AUDIT_V1.json), SHA-256 `4bec527d803f3c8c017c4fee6a8d3d6e533cda2a965a8d418b2877bdef78b344`.

The audit compares complete `(d,H,W,N)` inputs and every point, frame-corner, and window-corner float32 bit pattern, preserving tensor boundaries. It checks the 128 stored CP1 comparison inputs and reconstructs the 18 CP1 literal inputs from the CP1 fixture-data factory; no clearance or Boolean evaluator is called by the audit.

The audit independently reproduces the original overlap: 36/36 nextafter rows and 14/18 literal rows repeat CP1 inputs; none of the 192 seeded rows do. It therefore retains exactly **196** original CP2 cases (192 generated + four independent literals). The supplement has **36 new nextafter cases + 14 new hand-specified literal fixtures**. All 50 supplement inputs are byte-distinct from every CP1 input and every retained original CP2 input. The resulting confirmation set has **246 distinct full inputs**.

Coverage is retained for every `d∈{1,2}`, `H∈{1,2}`, `W∈{0,1,2}` configuration. Each of the 12 configurations has three supplemental lower-face nextafter probes (below/on/above); the literal fixtures cover closed frame/window faces, strict window interiors, degenerate frames/windows, overlapping and protruding windows, finite outside-domain queries, and multi-house composition/rescue. The 192 original generated cases provide 16 cases per configuration. Combined counts range from 19 to 22 per configuration, with all 12 represented.

The original retirement manifest remains unchanged at SHA-256 `b3b283a4b7849cd5dd5d08b9aa9ecf66888f8435954479a6f3a7823a0802ded3`.

## 3. Numerical verification — all previously approved limits unchanged

The new verifier preflights the original manifest, supplement, audit artifact, and regenerated complete-input audit before its first production evaluator call. It then evaluates the 196 retained cases plus all 50 supplemental cases on CPU and actual CUDA. It also independently checks direct Boolean expectations and `H=1/W=0` equality to the BOX clearance kernel.

**Result: PASS.** Every previously author-approved numerical budget was applied unchanged:

- Forward: ≤1 float32 ULP and scaled absolute error ≤`1e-7`.
- Near-zero (`|oracle|≤1e-6`): absolute error ≤`1e-7`; exact sign/zero status, with no crossings.
- CPU/CUDA: ≤1 ULP with exact sign/zero status.
- Unique-branch and CPU/CUDA gradient discrepancies: ≤`1e-6`.
- Native tied-gradient discrepancy: ≤`1e-6`.
- Non-tie central finite differences: ≤`2e-5`, float32 step `1e-3`.

| Measurement | House | Town |
|---|---:|---:|
| CPU/CUDA outputs | 246 each | 246 each |
| Max absolute oracle error | `5.960464477539063e-08` | `5.960464477539063e-08` |
| Max scaled oracle error | `5.893974124628436e-08` | `4.606144476972525e-08` |
| Max ULP from correctly rounded oracle | 0 | 0 |
| Near-zero output count | 44 | 45 |
| Near-zero max absolute error | 0 | 0 |
| Sign crossings / zero mismatches | 0 / 0 | 0 / 0 |
| CPU/CUDA max absolute / ULP difference | 0 / 0 | 0 / 0 |
| Direct Boolean mismatches | 0 | 0 |

Unique-branch gradient discrepancy was 0; tied-gradient discrepancy was 0; CPU/CUDA gradient discrepancy was 0. The maximum finite-difference discrepancy was `1.2934207916259766e-05`, within `2e-5`. `H=1/W=0` BOX-equivalence checks ran **84 times** (42 applicable cases on each device) and passed exactly.

Environment: Python 3.12.0; PyTorch 2.5.1+cu124; actual synchronized CUDA on NVIDIA GeForce RTX 2060, capability 7.5, driver 580.126.09, CUDA 12.4.

Machine-readable results: [`TOWN03_CP2_INDEPENDENCE_RESULTS_V1.json`](TOWN03_CP2_INDEPENDENCE_RESULTS_V1.json), SHA-256 `fa77c9079c3c8a6bdcc53dab6049a820a7643e2a1420313c0728d8d72cf64ac9`. The canonical combined-input-record hash is `924eb187b9b3dee64f2bf185b1e830fa9d1bef9fcdb13ce107b0920e95dd7ec9`.

## 4. Regression, architecture, and protected-boundary evidence

- Focused clearance/CP2/Inspector/T3 selection — **57 passed, 0 failed, 0 skipped**.
- Full repository suite at implementation SHA `ff771779…` — **634 passed, 0 failed, 0 skipped**, eight existing Streamlit/Pandas deprecation warnings.
- `tests/test_t3_architecture.py` — **4 passed**.
- Full collection — **634 collected**. T3 accounting now records eight CP2 confirmation-test nodes (six existing + two R1 checks); the original retirement manifest and its historical disposition counts are unchanged.
- `compileall` passed; `pip check` reported no broken requirements; `git diff --check 9c2106274a3bd4af80397a110b6e4981c3654aab..ff7717796226a5cb130785ef81001b576d773da1` passed.

Exact commands (run from the remediation worktree):

```bash
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python \
  tools/town03_cp2_freeze_independence_supplement.py \
  /tmp/TOWN03_CP2_INDEPENDENCE_SUPPLEMENT_V1.json
sha256sum /tmp/TOWN03_CP2_INDEPENDENCE_SUPPLEMENT_V1.json
cmp /tmp/TOWN03_CP2_INDEPENDENCE_SUPPLEMENT_V1.json \
  docs/reconstruction/TOWN03_CP2_INDEPENDENCE_SUPPLEMENT_V1.json

PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python \
  tools/town03_cp2_audit_input_independence.py \
  --supplement docs/reconstruction/TOWN03_CP2_INDEPENDENCE_SUPPLEMENT_V1.json

PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python \
  tools/town03_cp2_verify_independence_remediation.py

PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python \
  -m pytest -q -rs \
  tests/test_town_signed_clearance.py \
  tests/test_town03_cp2_confirmation.py \
  tests/test_town_math_inspector.py \
  tests/test_t3_architecture.py

PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python \
  -m pytest -q -rs
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python \
  -m pytest -q -rs tests/test_t3_architecture.py
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python \
  -m pytest --collect-only -q -p no:cacheprovider tests
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python \
  -m compileall -q src/town_model tests tools
/home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pip check
git diff --check 9c2106274a3bd4af80397a110b6e4981c3654aab..ff7717796226a5cb130785ef81001b576d773da1
```

## 5. Scope, limitations, and review request

No numerical limit or equation changed. The original CP2 manifest, original CP2 numerical results/report, CP1 evidence, clearance functions, Inspector, Boolean semantics, and retirement manifest remain untouched. No trainable geometry, loss, optimizer, classifier integration, learning, merge, issue closure, or TOWN-L01 work was started.

Please independently review the fixed implementation SHA `ff7717796226a5cb130785ef81001b576d773da1` and this report/results against the authorization and R1 acceptance criteria. Verify the freeze-before-evaluation provenance, byte-level disjointness, unchanged numerical budgets, actual CPU/CUDA results, regressions, and preserved boundaries. The implementer does not self-accept. The issue remains open; no integration is authorized by this handoff.
