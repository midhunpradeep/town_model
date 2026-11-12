# TOWN-03 CP2 Independent Technical Review — v1

**Verdict: CHANGES REQUIRED**

This is a versioned independent review of the submitted TOWN-03 CP2 work. The mathematical implementation, the author-locked numerical limits on the frozen inputs, the Inspector, and repository protections pass the checks below. The confirmation manifest is **not fully separated from CP1 exploratory inputs**, however: 50 of its 246 cases exactly repeat CP1 inputs. Because this review was explicitly required to verify separation, I cannot give an overall PASS. No implementation fixes were made.

## 1. Exact revisions and authority

- Repository: `midhunpradeep/town_model`, issue [#23](https://github.com/midhunpradeep/town_model/issues/23).
- Starting / live `main`: `404d76112a9d53282b3f55698b4decb705a5601c`.
- **Implementation reviewed:** `258aafa6399adff0d106f8a2dfd58db58f862ad5`.
- **Final evidence reviewed:** `9c2106274a3bd4af80397a110b6e4981c3654aab`; its parent is exactly the implementation SHA.
- Frozen-set build commit: `af58270e643dacf2110ce5932fb6533c0e8aee55`.
- Direct author CP1 contract confirmation: issue #23 comment `6813474135`.
- Direct author CP2 numerical signoff and authorization: issue #23 comment `6814259456`. It locks the six reported budgets and the 246-case plan; I verified the live OWNER comment and did not infer signoff from the implementation report.
- Accepted prerequisite TOWN-02 #22 is closed/accepted and integrated at the starting `main` revision.
- At review, #23 was OPEN / `workflow:in-review`, assigned to the implementer.

I used a separate review worktree at `/tmp/town-model-town03-independent-review`, created at the exact final evidence SHA. The implementation worktree was not edited. The concurrent BOX-05 worktree remains at `3e62681f05d2f4ee6144cd0b8362e9e0371bf5b0`, clean and untouched. `main` remains at `404d761…`; no merge or integration was attempted.

## 2. Revision/evidence diff

`git diff --name-status 258aafa6399adff0d106f8a2dfd58db58f862ad5 9c2106274a3bd4af80397a110b6e4981c3654aab` contains only:

- `A docs/reconstruction/TOWN03_CP2_CONFIRMATION_REPORT.md`
- `A docs/reconstruction/TOWN03_CP2_CONFIRMATION_RESULTS.json`

Thus later evidence commits add no executable changes after the reviewed implementation SHA. The implementation's base-to-target source changes are limited to the CP1/CP2 clearance functions, fixed Inspector fixtures/presentation, scoped tests, CP2 manifest/generator/verifier, and separate T3 accounting. Existing `house_contains` and `town_contains` implementations are not changed by the clearance additions.

## 3. Mathematical and architectural findings

The implementation matches the approved raw-coordinate equations:

\[
s_B(x)=\min_j(x_j-l_j,r_j-x_j),\quad
s_H(x)=\min(s_F(x),-s_{W_1}(x),\ldots),\quad
s_T(x)=\max_h s_{H_h}(x).
\]

`house_signed_clearance` uses `torch.amin` for each box and the house branches; `town_signed_clearance` uses those minima followed by `torch.amax` over houses. I found no smoothing, rescaling, clipping, custom backward, or Boolean threshold conversion.

- **House/town composition:** independent literal and exact-rational checks confirm the frame term and negated window-clearance terms combine through `amin`, and houses combine through `amax`.
- **H=1/W=0:** the code reduces to frame clearance; the focused tests pass exact equality to the BOX kernel. My independent replay also confirmed exact BOX equivalence for 43 matching manifest cases on each device (86 calls total).
- **Autograd:** gradients flow through query points and physical corners. Native PyTorch `amin`/`amax` equal-split tie behavior is preserved. Independent analytic checks confirmed the frame-face tie, house-branch tie/cancellation, and town-house `amax` tie on CPU and CUDA.
- **Geometry edge cases:** tests and independent fixture checks cover `N=0`, `W=0`, collapsed frames, one-axis-degenerate windows, overlapping windows, protruding windows, boundary/coincident faces, multiple houses, outside-domain finite points, and invalid inputs. The empty query result is `[0]` float32; town `H=0` rejects.
- **Boolean independence:** clearance functions do not call Boolean membership. `house_contains`/`town_contains` retain direct closed-frame / strict-window-interior comparisons. The Inspector invokes membership and clearance separately. Independent direct set-equation membership calculations matched the Boolean APIs on all 246 frozen cases; zero mismatches.
- **Sign/Boolean diagnostic:** the report correctly notes that a `s >= 0` versus Boolean-membership disagreement is impossible under the approved equations. The Inspector instead marks `s > 0` as a strict-positive diagnostic, displaying included zero-clearance boundary points without using clearance to generate membership.

No mathematical or implementation defect was found in these reviewed areas.

## 4. Frozen confirmation set and numerical verification

### Freeze, hash, construction, and oracle

- Frozen manifest: `docs/reconstruction/TOWN03_CP2_CONFIRMATION_SET.json`.
- Recorded and independently recomputed SHA-256: `acb1f9c43110ac839530066ef2aa3311755003321b945b65345f3445b4548f3e`.
- CP2 result artifact SHA-256 independently recomputed as `2adc4a00ae12fc26a40306bf5dbe05a121813a93d38d8b25f438f959c7cd1188`.
- Generator: `tools/town03_cp2_freeze_confirmation_set.py`; it imports no TOWN evaluator and computes Decimal clearance plus direct Boolean set expectations.
- I reran `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python tools/town03_cp2_freeze_confirmation_set.py /tmp/town03-cp2-regenerated.json`; `cmp -s /tmp/town03-cp2-regenerated.json docs/reconstruction/TOWN03_CP2_CONFIRMATION_SET.json` confirmed byte-for-byte identity, and `sha256sum` returned the locked hash for both.
- Inventory is 192 seed-`20261113` generated cases, 36 nextafter probes, and 18 literal cases, across all 12 `(d,H,W)` configurations for `d∈{1,2}`, `H∈{1,2}`, `W∈{0,1,2}`.
- The set commit and issue journal record the manifest hash before the evaluator run; the verifier/Inspector implementation commit follows the set commit. This supports the claimed freeze-before-evaluation order. As with any repository audit, I cannot independently prove that no unlogged out-of-band run occurred before the commit.

### Finding R1 — confirmation inputs overlap CP1 exploratory inputs

I canonicalized full `(d,H,W, points, frame corners, window corners)` float32 inputs from the CP1 exploratory artifact/source against the frozen CP2 manifest:

- All **36/36** CP2 nextafter cases are exact geometry/input repeats of the 36 CP1 nextafter probes.
- **14/18** CP2 literal cases are exact repeats of CP1 literal fixtures.
- The **192/192** seed-generated CP2 cases have no exact input overlap with the CP1 comparison cases.
- Total exact repeated CP2 manifest rows: **50/246**. Only four CP2 literal cases are new relative to CP1 literals.

The frozen set is separately versioned and its seeded generated cases are distinct, but the full confirmation set is not input-disjoint from exploratory characterization. Therefore the reported PASS establishes the locked limits on the 246 listed rows, but does not establish a wholly independent 246-case confirmation relative to CP1. This is an evidence-independence deficiency, not a numerical-limit failure. The implementer/author must disposition the overlap; I did not alter or re-freeze the author-approved manifest. If the intended criterion requires a disjoint confirmation set, any replacement set must be authorized, frozen and hashed before evaluation, then rerun against the unchanged approved limits.

### Independent reproduction against every author-approved budget

I independently replayed all cases using a separate exact `Fraction` oracle over the stored binary32 inputs (not the verifier's Decimal helper), a separately derived nearest-binary32 value, and direct Boolean set comparisons. Production outputs were executed on actual CPU and CUDA. Environment observed: Python 3.12.0, PyTorch 2.5.1+cu124, CUDA 12.4, NVIDIA GeForce RTX 2060, capability 7.5; explicit CUDA synchronization succeeded.

| Approved criterion | Independent observation | Result |
|---|---:|---|
| Forward: ≤1 ULP from correctly rounded oracle and scaled absolute error ≤`1e-7` | House/town max ULP 0; max scaled error `5.893974124628436e-08` / `4.606144476972525e-08` | Pass |
| Near-zero (`|oracle|≤1e-6`): absolute error ≤`1e-7`; exact zero/sign | 31 outputs per evaluator/device; maximum near-zero error 0; zero crossings/status mismatches 0 | Pass |
| CPU/CUDA: ≤1 ULP and exact zero/sign agreement | 246 house + 246 town outputs per device; max difference 0 ULP / 0 absolute; no sign/zero mismatches | Pass |
| Unique-branch gradients: ≤`1e-6` vs analytic and CPU/CUDA | Maximum gradient discrepancy 0; CPU/CUDA gradient discrepancy 0 | Pass |
| Tied gradients: ≤`1e-6` vs native PyTorch tie expectations | Maximum discrepancy 0 | Pass |
| Non-tie central finite difference: ≤`2e-5`, float32 step `1e-3` | Maximum discrepancy `1.2934207916259766e-05` | Pass |

Maximum forward absolute error was `5.960464477539063e-08` for house and town. No numerical threshold was relaxed. These results apply to the frozen inputs and stated hardware/software, not arbitrary hardware or untested capacities.

## 5. Inspector and regression/retirement review

The standalone Inspector contains fixed read-only fixtures for frame/window boundaries, a one-axis-degenerate window, a collapsed frame, a two-house town, and overlapping windows. Tables show expected/actual Boolean membership and expected/actual clearance separately, plus zero-boundary diagnostics. It calls the established Boolean APIs separately from clearance. Source inspection found no controls, model/training path, mutable geometry, dataset loading, or artifact-writing operation. AppTests exercise all five clearance fixtures, verify exact table values and `artifacts/` non-mutation, reject incorrect evaluator outputs with point/scenario diagnostics, and assert no buttons or input controls. A live standalone Streamlit launch returned health HTTP 200 (`ok`) and root HTTP 200. This is technical rendering evidence, not the thesis author's visual acceptance.

T3 architecture/accounting and retirement integrity passed:

- TOWN-03 separate accounting: 36 mathematical nodes, 6 confirmation nodes, 3 Inspector nodes.
- `tests/test_t3_architecture.py`: **4 passed**; it retains historical retirement assertions and validates T3 manifest replacements against collected nodes.
- Retirement manifest SHA-256: `b3b283a4b7849cd5dd5d08b9aa9ecf66888f8435954479a6f3a7823a0802ded3`, unchanged from the accepted baseline.
- No historical classifier implementation was restored or made reachable.

## 6. Commands and observed results

All commands ran in `/tmp/town-model-town03-independent-review`, pinned at final evidence SHA `9c2106274a3bd4af80397a110b6e4981c3654aab`; Python was `/home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python`.

- `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q -rs tests/test_town_signed_clearance.py tests/test_town03_cp2_confirmation.py tests/test_town_math_inspector.py tests/test_t3_architecture.py` — **55 passed, 0 failed, 0 skipped**; includes actual CUDA-parametrized tests and the locked confirmation runner.
- `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q -rs` — **632 passed, 0 failed, 0 skipped**, 8 existing Pandas/Streamlit deprecation warnings.
- `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q -rs tests/test_t3_architecture.py` — **4 passed**.
- `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest --collect-only -q -p no:cacheprovider tests` — **632 collected**.
- `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m compileall -q src/town_model tests tools` — passed.
- `/home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pip check` — no broken requirements.
- `git diff --check 404d76112a9d53282b3f55698b4decb705a5601c 258aafa6399adff0d106f8a2dfd58db58f862ad5` and `git diff --check 258aafa6399adff0d106f8a2dfd58db58f862ad5 9c2106274a3bd4af80397a110b6e4981c3654aab` — passed.
- `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python reviews/TOWN03_CP2_INDEPENDENT_ORACLE_2026-11-12.py` — independently recomputed exact-rational forward oracle, direct Boolean oracle, gradients, CPU/CUDA comparisons, and BOX equivalence; all approved numerical limits passed (summary in §4). This review-only script does not call the implementation team's Decimal verifier.
- Live Inspector: launched `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m streamlit run src/town_model/dashboard/town_math_inspector.py --server.headless=true --server.address=127.0.0.1 --server.port=19323 --browser.gatherUsageStats=false` in the background; `curl --silent --fail http://127.0.0.1:19323/_stcore/health` returned `ok`, and `curl --silent --output /tmp/town03-inspector-root.html --write-out '%{http_code}\n' http://127.0.0.1:19323/` returned `200`. The process was stopped after the check.
- `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m py_compile reviews/TOWN03_CP2_INDEPENDENT_ORACLE_2026-11-12.py` — passed.

The eight warnings are non-failing existing Pandas4Warning deprecations. No skips or CUDA unavailability occurred.

## 7. Limits and handoff

- CP2 numerical coverage is bounded to the 246 manifest rows, `d≤2`, `H≤2`, `W≤2`, and the recorded CPU/RTX 2060 environment. No large-capacity performance test or universal cross-device guarantee was established.
- The frozen manifest's generated block is independent of CP1 random cases, but the 50 exact structural/literal repeats prevent the requested whole-set separation claim.
- The author visual-acceptance gate remains open. Because the overall technical verdict is CHANGES REQUIRED, I am **not** recording a final technical pass or asking that TOWN-03 be accepted. After the overlap is dispositioned and any required verification is reviewed, the thesis author must separately inspect the Inspector visually; integration still requires separate authorization.
- Issue #23 must remain open. No TOWN-03 acceptance, issue closure, merge, correction, or TOWN-L01 work was performed by this reviewer.

The review used a separate Pi agent session. The harness did not provide the two parallel subagents required by the installed code-review skill, so that workflow is not claimed. GitHub CLI authenticates through the shared repository-owner account; independence is session-level, not a claim of a separate GitHub identity.
