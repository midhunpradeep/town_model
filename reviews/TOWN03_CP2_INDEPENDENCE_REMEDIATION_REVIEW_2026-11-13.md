# TOWN-03 CP2 R1 Remediation — Independent Fixed-SHA Review

**Verdict: PASS — R1 is resolved; no additional technical blocker was found.** This is technical acceptance of the bounded input-independence remediation only. It does **not** accept/close all of TOWN-03: issue #23 remains open for the thesis author's visual Inspector acceptance and separate integration approval.

## 1. Scope, authority, and exact revisions

- Repository: `midhunpradeep/town_model`; issue [#23](https://github.com/midhunpradeep/town_model/issues/23).
- Original finding: issue comment [`6815233836`](https://github.com/midhunpradeep/town_model/issues/23#issuecomment-6815233836), CHANGES REQUIRED because 50 CP2 inputs repeated CP1 inputs.
- Author remediation approval: comment [`6830101628`](https://github.com/midhunpradeep/town_model/issues/23#issuecomment-6830101628).
- Prior CP2 evidence baseline: `9c2106274a3bd4af80397a110b6e4981c3654aab`.
- Supplement/audit freeze commit: `9f69ac9e55ca1bae64e06dadd6d89fc522e1aa60`.
- **Reviewed remediation implementation:** `ff7717796226a5cb130785ef81001b576d773da1` (parent `9f69ac9e55ca1bae64e06dadd6d89fc522e1aa60`).
- **Final evidence SHA:** `190792adccdf781392bb5259f062e31e13974e37` (parent is the implementation SHA; adds only the remediation report and numerical results JSON).
- Current `main` and `origin/main`: `3e62681f05d2f4ee6144cd0b8362e9e0371bf5b0`. BOX-05 #24 is CLOSED / `workflow:accepted`.

The review ran in `/tmp/town-model-town03-cp2-r1-review`, a separate worktree/branch directly based on the exact final evidence SHA. No implementation branch, `main`, BOX-05 worktree, or concurrent work was altered, rebased, or merged.

## 2. Independent review method and process limitation

I independently compared full tensor inputs using exact binary32 encodings and independently replayed the clearance equations with high-precision `Decimal` arithmetic plus direct Boolean set comparisons. The review script is versioned at [`TOWN03_CP2_INDEPENDENCE_REVIEWER_CHECKS_2026-11-13.py`](TOWN03_CP2_INDEPENDENCE_REVIEWER_CHECKS_2026-11-13.py), SHA-256 `266fd8b51018ccf4200a0010d5837be7ee59f3e7b579cd9281b7e5d31299c5ad`. It does not import the implementation agent's independence-audit or numerical-verifier modules. It reconstructs CP1's 18 fixed input fixtures by calling only the original CP1 fixture-data factory; it does not run its clearance evaluator.

The harness cannot launch the two parallel fresh-context subagents required by the installed `code-review` skill. I do not claim that two-axis subagent workflow ran. Per the repository execution protocol's fallback, this was an explicitly assigned independent reviewer, performing direct fixed-SHA technical review and independent checks. This is a separate Pi session (`01a1d5fc-0e0a-7449-bbe7-d3d16e3395a0`) and worktree; the GitHub CLI uses the same repository-owner identity as the implementer, so independence is session/worktree-level, not a separate authenticated GitHub account.

## 3. R1 — full-input disjointness and coverage

The comparison key included `(d,H,W,N)`, each tensor's implied shape, and every float32 bit pattern in `points`, `frame_lower`, `frame_upper`, `window_lower`, and `window_upper`. This preserves tensor boundaries and distinguishes signed/adjacent float32 values; case IDs and expected scores were not used as input identity.

| Set | Independently observed inventory |
|---|---:|
| CP1 inputs reconstructed from result artifact and fixture factory | 146 rows / 145 unique complete inputs |
| Original CP2 manifest | 246 rows; its only duplicate pair was inside the already-overlapping CP1 subset |
| Original CP2 rows overlapping CP1 | 50: 36 nextafter + 14 literal; 192 generated rows do not overlap |
| Retained original CP2 inputs | 196 unique: 192 generated + 4 literals |
| R1 supplement | 50 unique: 36 nextafter + 14 literal |
| Supplement overlap with CP1 / retained original CP2 | 0 / 0 |
| Combined retained-original + supplement set | 246 unique inputs; zero intersection with CP1 |

The duplicate CP1 rows are the identical complete geometry inputs `house-1d-window-interior` and `town-1d-window-interior`. The duplicate original-CP2 rows are `literal-house-1d-window-interior` and `literal-town-1d-hole-interior`; both are within the 50 removed overlaps. These source duplicates do not enter the retained 196 or final 246-case set. This detail is why the source inventories are described as rows and the retained/combined inventories as unique complete inputs.

The 36 nextafter cases contain exactly one below/on/above triplet for each approved `(d,H,W)` configuration, `d∈{1,2}`, `H∈{1,2}`, `W∈{0,1,2}`. I independently checked each below/above point is the immediate float32 neighbor of the first frame-lower face and each on-point equals the face. The 14 literal inputs were individually re-evaluated against exact stored-float32 Decimal score oracles and direct Boolean set equations. Their actual inputs demonstrate closed frame/window faces, strict window interiors, degenerate frames/windows, overlapping and protruding windows, finite outside-domain queries, and multi-house union/rescue. All 12 approved configurations remain present; final per-configuration counts range from 19 to 22:

| `d,H,W` | Count | `d,H,W` | Count | `d,H,W` | Count |
|---|---:|---|---:|---|---:|
| 1,1,0 | 22 | 1,1,1 | 21 | 1,1,2 | 21 |
| 1,2,0 | 21 | 1,2,1 | 21 | 1,2,2 | 19 |
| 2,1,0 | 20 | 2,1,1 | 20 | 2,1,2 | 20 |
| 2,2,0 | 20 | 2,2,1 | 20 | 2,2,2 | 21 |

**Result: PASS.** This independently reproduces the exact original overlap and verifies every supplement row and all 246 final rows, not merely the submitted audit summary.

## 4. Freeze-before-evaluation provenance

The frozen supplement and input audit first appear in commit `9f69ac9e55ca1bae64e06dadd6d89fc522e1aa60`, timestamped 2026-11-12 18:07:11 +0100. At that revision their hashes are:

- `TOWN03_CP2_INDEPENDENCE_SUPPLEMENT_V1.json`: `b53625dcc7eb428ef02cca5dbba48831b0c8ca9fff5182e98df558d48867eafa`.
- `TOWN03_CP2_INPUT_INDEPENDENCE_AUDIT_V1.json`: `4bec527d803f3c8c017c4fee6a8d3d6e533cda2a965a8d418b2877bdef78b344`.

The author published those hashes and the pre-evaluation audit in issue comment `6830480378` at 17:09:56Z. The numerical-run checkpoint was posted later, at 17:16:52Z (`6830656006`). The R1 numerical verifier source was added in the later implementation commit `ff771779…` (17:17:55Z). The freeze commit itself contains no R1 evaluator runner; the freeze generator imports no TOWN evaluator, and the audit calls only CP1's fixture-data constructor, not a clearance or membership evaluator. The final result artifact was added later at `190792ad…`.

This establishes the published hash/freeze and issue-journal ordering before the recorded evaluator run. As usual, repository evidence cannot prove the absence of an unlogged out-of-band execution; no such execution is indicated by the immutable commits or issue journal.

## 5. Reproduction against every unchanged author-approved limit

I replayed the combined 246 inputs against the production functions on actual CPU and CUDA, using exact Decimal values derived from the stored float32 inputs, a separately derived correctly rounded float32 oracle, and direct Boolean expectations. The approved limits remain those in author signoff comment `6814259456`; no tolerance or equation was changed.

| Approved check | Independent result | Limit | Verdict |
|---|---:|---:|---|
| Forward max scaled absolute error, house / town | `5.893974124628436e-8` / `4.606144476972525e-8` | `≤1e-7` | PASS |
| Maximum absolute error, house / town | `5.9604644775390625e-8` / same | derived from scaled budget | PASS |
| ULP from correctly rounded oracle, house / town | 0 / 0 | `≤1` | PASS |
| Near-zero output count, house / town | 44 / 45 per device; maximum near-zero error 0 | if `|oracle|≤1e-6`, error `≤1e-7` | PASS |
| Sign crossings / exact-zero-status mismatches | 0 / 0 | exactly 0 | PASS |
| Direct Boolean membership mismatches | 0 | exactly 0 | PASS |
| CPU/CUDA forward agreement | 246 house + 246 town outputs; 0 absolute / 0 ULP difference, no sign/zero mismatch | `≤1` ULP; exact sign/zero | PASS |
| Unique-branch gradient error vs analytic and CPU/CUDA | 0 / 0 | `≤1e-6` | PASS |
| Native tied-gradient error vs independent tie expectations | 0 | `≤1e-6` | PASS |
| Central finite-difference maximum error (float32 step `1e-3`) | `1.2934207916259766e-5` | `≤2e-5` | PASS |
| H=1/W=0 BOX equivalence | 84 exact comparisons (42 cases × 2 devices) | exact equality | PASS |

Unique and tied probes exercised gradients through query points and physical corners, including native `amin` ties, a house-branch tie with point-gradient cancellation, and a town `amax` tie. The independently measured results match the submitted CP2 report. Environment: Python 3.12.0; PyTorch 2.5.1+cu124; CUDA 12.4; NVIDIA GeForce RTX 2060, capability 7.5; driver 580.126.09.

## 6. Original evidence, equations, limits, Inspector, and retirement integrity

I compared the protected paths in the submitted evidence tree with the reviewed CP2 baseline `9c210627…`; the paths are byte-identical. Representative SHA-256 values:

| Protected path | SHA-256 |
|---|---|
| `src/town_model/models/town_geometry.py` (equations and Boolean APIs) | `fd50bbf924eb32b86efad236ecd57ed6292a44a4810e02a939088dadd2887541` |
| `src/town_model/dashboard/town_math_inspector.py` | `12657a1c2996dd570846db6315fa53d6072792d376e8a38108f71f0e5ff107c7` |
| `tools/town03_cp2_verify_confirmation_set.py` (original locked limits) | `ec01727f8c82c4429c591977f0cf1584ce24ff9c8905fdd85b6fc07e64964f00` |
| Original CP1 results / report | `7e31ffeaac15b704e3b582f84e4ef1fd5903591380a54dc98895f36a626457c2` / `cad18fa6f42b4cd04ec09511fb9c83f1b4e52c08ea66f70edd0f647c501a9999` |
| Original CP2 manifest / results / report | `acb1f9c43110ac839530066ef2aa3311755003321b945b65345f3445b4548f3e` / `2adc4a00ae12fc26a40306bf5dbe05a121813a93d38d8b25f438f959c7cd1188` / `1aa500326b7417d7de82e148f12d895ceef06d59f7f333c92fa38a465b22dc37` |
| `docs/reconstruction/T3_TEST_RETIREMENT_MANIFEST.md` | `b3b283a4b7849cd5dd5d08b9aa9ecf66888f8435954479a6f3a7823a0802ded3` |

The implementation diff from `9c210627…` to `ff771779…` contains only the two new frozen input/audit JSONs, their generator/auditor and numerical-verification tools, two R1 regression tests, and the T3 confirmation-node count update. It changes no production model, Boolean evaluator, equation, limit, or Inspector file. The `ff771779…` to `190792ad…` diff adds only the remediation report and numerical results JSON. This review's own branch adds only this review report and its review-only script; neither is production runtime code.

T3 architecture accounting now expects eight CP2 confirmation test nodes (six prior + two R1 checks). The full test collection is 634; the T3 architecture test passes; the retirement manifest remains byte-identical and its historical disposition counts were not rewritten.

## 7. Regression commands and results

Run in `/tmp/town-model-town03-cp2-r1-review`, pinned to final evidence SHA `190792adccdf781392bb5259f062e31e13974e37`; Python executable is `/home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python`.

```bash
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python \
  reviews/TOWN03_CP2_INDEPENDENCE_REVIEWER_CHECKS_2026-11-13.py
# PASS: independent full-input audit, exact Decimal/direct-Boolean CPU+CUDA replay,
#       all locked budgets, gradient probes, BOX equivalence, frozen-artifact hashes

PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python \
  -m pytest -q -rs tests/test_town_signed_clearance.py \
  tests/test_town03_cp2_confirmation.py tests/test_town_math_inspector.py \
  tests/test_t3_architecture.py
# 57 passed, 0 failed, 0 skipped

PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python \
  -m pytest -q -rs
# 634 passed, 0 failed, 0 skipped; 8 existing Pandas/Streamlit deprecation warnings

PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python \
  -m pytest -q -rs tests/test_t3_architecture.py
# 4 passed

PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python \
  -m pytest --collect-only -q -p no:cacheprovider tests
# 634 collected

PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python \
  -m compileall -q src/town_model tests tools
# passed
/home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pip check
# No broken requirements found

git diff --check 9c2106274a3bd4af80397a110b6e4981c3654aab..ff7717796226a5cb130785ef81001b576d773da1
git diff --check ff7717796226a5cb130785ef81001b576d773da1..190792adccdf781392bb5259f062e31e13974e37
# both passed
```

## 8. Acceptance mapping and handoff

| Required check | Finding |
|---|---|
| Original CP1/CP2 artifacts, equations, limits, and Inspector unchanged | PASS — protected hashes and baseline diff verified |
| Supplement has 36 boundary probes + 14 literal fixtures | PASS — independently counted and inspected |
| 50 complete float32 inputs disjoint from CP1 and retained 196 | PASS — independently compared every field/bit pattern; both intersections zero |
| Final 246 disjoint from CP1; approved dimensional/house/window/boundary coverage retained | PASS — 246 unique; all 12 configurations, 36 below/on/above probes and literals verified |
| Supplement/audit frozen, hashed, published before recorded evaluation | PASS — freeze commit, hashes, and issue-journal chronology verified |
| Numerical limits reproduced on actual CPU and CUDA | PASS — every unchanged limit passes on RTX 2060 and CPU |
| Gradients, BOX equivalence, regressions, T3, retirement integrity | PASS — all numerical probes and suite/accounting checks pass; manifest unchanged |
| Unauthorized changes and evidence/code distinction | PASS — no production source/Inspector changes; review evidence is separate |

**R1 is resolved. No additional technical blocker remains for this bounded remediation.** The scope remains limited to the frozen case set and recorded environment; this is not a universal hardware or performance guarantee.

Issue #23 remains OPEN / `workflow:in-review`. I do not close it, mark the full TOWN-03 milestone accepted, authorize integration, or start TOWN-L01. **Next gates:** thesis author's visual acceptance of the unchanged standalone Inspector, then separate integration approval. No implementation changes were made by this reviewer.
