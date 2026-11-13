# TOWN-03 Combined Integration — Independent Fixed-SHA Review

**Verdict: PASS.** Candidate `679cb332b9fc1fde0243c961f4d203973b9ebe36` is technically eligible for the already author-authorized integration in issue comment `6860940484`. This review does not merge, close #23, or start TOWN-L01.

## 1. Reviewed revisions and independence

- Repository / issue: `midhunpradeep/town_model`, [#23](https://github.com/midhunpradeep/town_model/issues/23).
- **Exact combined integration candidate reviewed:** `679cb332b9fc1fde0243c961f4d203973b9ebe36`, branch `integration/town03-accepted-20261113`.
- **Accepted base:** `3e62681f05d2f4ee6144cd0b8362e9e0371bf5b0`.
- Author's controlled integration authorization: [comment 6860940484](https://github.com/midhunpradeep/town_model/issues/23#issuecomment-6860940484).
- Accepted TOWN-03 evidence SHA: `190792adccdf781392bb5259f062e31e13974e37`; accepted remediation implementation SHA: `ff7717796226a5cb130785ef81001b576d773da1`; final independent review SHA: `6df07dc6197c9e60639afabf316110c7deefc83c`.
- Ancestry checks pass for the exact evidence and final review SHAs. I byte-compared all 21 accepted TOWN-03 implementation, test, tool, fixture, and evidence paths from `190792…` to the candidate: **all identical**. The candidate additionally carries the final review report and reviewer-check script; their SHA-256 values are `6ffff56e0073d21a070c0a12eeacde152c0aef722eb39f6b2830bbc0fe56fc22` and `266fd8b51018ccf4200a0010d5837be7ee59f3e7b579cd9281b7e5d31299c5ad` respectively.
- Review worktree: `/tmp/town-model-town03-integration-review-679cb33`, initially detached at the exact candidate, separate from implementation, remediation, prior-review, integration, and main worktrees. The report is the only new file on this review branch.

The first R1 review's CHANGES REQUIRED finding is preserved in issue comment `6815233836` and is fully dispositioned in the accepted R1 remediation report/re-review. The candidate carries the author-designated accepted evidence and final PASS review; it does not carry the earlier non-acceptance review worktree as a separate report file.

## 2. Combined diff, BOX-05 preservation, and T3 accounting

The candidate-to-base diff contains 24 TOWN-03-scoped paths: clearance implementation/fixture and Town Inspector additions; scoped tests and T3 accounting; CP1/CP2/R1 reports, frozen inputs/results, verification tools, and final independent review artifacts. No BOX-05 source, test, evidence, or numerical-contract path differs from the accepted base. Targeted `git diff --exit-code` checks over the BOX-05 implementation, Inspector, tests, and reports returned 0. BOX-05 issue #24 remains accepted/closed.

The only shared accounting conflict is resolved additively in `tests/test_t3_architecture.py`:

- Existing BOX-05 accounting remains unchanged: 16 CP1 nodes and 22 CP2/CP3 nodes; the two BOX-05 Inspector assertions remain members of the 22-node CP2/CP3 set and are not double-counted.
- TOWN-03 adds 36 mathematical tests, 8 confirmation tests, and 3 distinct Inspector tests (47 total).
- Existing BOX/TOWN cohorts and fixed T3 regression nodes remain present. The feature-node unions and legacy-node complement partition the 672 collected nodes; the test's cardinality equality rejects overlap, and its fixed node/group disposition assertions still guard omissions and retirement weakening.
- `tests/test_t3_architecture.py`: **4 passed**. Full collection: **672**.

No conflict resolution changed geometry, numerical limits, or Inspector code. The protected `docs/reconstruction/T3_TEST_RETIREMENT_MANIFEST.md` is byte-identical to the accepted base and candidate; SHA-256 on both is `b3b283a4b7849cd5dd5d08b9aa9ecf66888f8435954479a6f3a7823a0802ded3`.

## 3. Mathematical and Boolean semantics

The added house score is the approved raw-coordinate composition `s_H = min(s_F, -s_W1, …, -s_WW)`, with each box score the minimum of its lower/upper face clearances. The town score is `s_T = max_h s_Hh`. The implementation uses native `torch.amin`/`torch.amax`, preserving framework tie gradients; `W=0` reduces to frame clearance, and `H=1,W=0` is exactly the BOX kernel.

I compared the candidate and accepted-base AST source for `_validate_house_inputs`, `_validate_town_inputs`, `box_strict_interior`, `house_contains`, and `town_contains`: all five are **source-identical**. Existing TOWN-01/02 fixtures are byte-identical. Boolean membership remains direct comparison/set logic and is evaluated separately by the Town Inspector; no clearance threshold is used to manufacture membership. The documented boundary limitation is handled honestly: under these approved equations, a nonnegative-clearance/Boolean sign disagreement is impossible for valid geometry; the Inspector shows strict-positive boundary differences only as a labeled diagnostic.

BOX-05 source, Inspector, tests, CP1/CP2/CP3 evidence, and its numerical contracts remain byte-identical to the accepted base. No tolerance or acceptance threshold is changed in the integration.

## 4. Independent numerical confirmation

I independently ran both the accepted R1 reviewer replay and the remediation verifier against the candidate, on the actual CPU and CUDA devices. The frozen set is 246 distinct complete float32 inputs: 192 original seeded cases + 4 retained original independent literals + 36 new nextafter probes + 14 new literal fixtures. It is disjoint from the CP1 inputs and covers all 12 `(d,H,W)` configurations. The original 50-row CP1 overlap was removed without changing the original manifest/results or approved thresholds.

| Observation | Result | Approved limit |
|---|---:|---:|
| Max scaled forward error, house / town | `5.893974124628436e-8` / `4.606144476972525e-8` | `≤1e-7` |
| Max absolute forward error, house / town | `5.960464477539063e-8` / same | also within scaled limit |
| Max ULP from correctly rounded oracle | 0 | `≤1` |
| Near-zero cases, house / town; max error | 44 / 45; 0 | `|oracle|≤1e-6`, error `≤1e-7` |
| Sign crossings / exact-zero mismatches | 0 / 0 | 0 / 0 |
| Direct Boolean mismatches | 0 | 0 |
| CPU/CUDA forward agreement | exact; 0 ULP and 0 sign/zero mismatches | `≤1` ULP; exact sign/zero |
| Unique/tied gradient discrepancy; cross-device gradient discrepancy | 0 / 0 | each `≤1e-6` |
| Non-tie float32 central-difference max error (step `1e-3`) | `1.2934207916259766e-5` | `≤2e-5` |
| Exact H=1/W=0 BOX equivalence calls | 84, all exact | exact |

Actual environment: Python 3.12.0, PyTorch 2.5.1+cu124, CUDA 12.4, RTX 2060 (compute capability 7.5; driver 580.126.09). No numerical threshold was relaxed. This is evidence for the locked finite set and recorded device/software stack, not a universal hardware guarantee.

## 5. Inspectors and regression verification

Both Inspector paths retain their accepted behavior in candidate tests:

- BOX-05 read-only Inspector fixture and mismatch checks, plus the existing BOX Mathematical Inspector navigation/presentation checks.
- Town Inspector TOWN-01/02 sections and the additive fixed-fixture TOWN-03 section, with independent Boolean and clearance values and no writes/controls.
- The focused BOX/TOWN/Inspector/T3 selection passed **479 tests, 0 failures, 0 skips** (6 Pandas4Warning deprecations). It includes BOX-01–05, TOWN-01–03, both Inspector test sets, the shared dashboard tests, and T3 architecture.
- I also launched the standalone Town Inspector: `/_stcore/health` returned `ok`; `/` returned HTTP 200. This is not a substitute for visual review; author visual acceptance is already recorded at comment `6860895544`.

Required verification on the exact candidate worktree:

```bash
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q -rs
# 672 passed, 0 failed, 0 skipped; 8 Pandas/Streamlit deprecation warnings

PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q -rs \
  tests/test_box05_cp1_geometry.py tests/test_box05_cp2_cp3.py \
  tests/test_bounded_box.py tests/test_bounded_multi_box.py \
  tests/test_bound_box_inference.py tests/test_town_geometry.py \
  tests/test_town_composition.py tests/test_town_signed_clearance.py \
  tests/test_town03_cp2_confirmation.py tests/test_town_math_inspector.py \
  tests/test_dashboard_app.py tests/test_t3_architecture.py
# 479 passed, 0 failed, 0 skipped; 6 Pandas4Warning deprecations

PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q -rs tests/test_t3_architecture.py
# 4 passed

PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest --collect-only -q -p no:cacheprovider tests
# 672 collected

PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m compileall -q src/town_model tests tools reviews
# passed

/home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pip check
# No broken requirements found

git diff --check 3e62681f05d2f4ee6144cd0b8362e9e0371bf5b0 679cb332b9fc1fde0243c961f4d203973b9ebe36
# passed

PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python reviews/TOWN03_CP2_INDEPENDENCE_REVIEWER_CHECKS_2026-11-13.py
# PASS; independent input audit, Decimal/direct-Boolean CPU+CUDA replay, gradients, BOX equivalence, and artifact hashes

PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python tools/town03_cp2_verify_independence_remediation.py --output /tmp/town03-integration-remediation-replay.json
# PASS; 246 distinct inputs; CPU/CUDA and every locked numerical/Boolean/gradient limit passed
```

## 6. Diff boundary, concurrent state, and limitations

The full candidate diff has no edits under training, experiment, vision, loss, optimizer, or classifier paths; no BOX-05 path changed; no historical classifier definition or retired module was restored. The T3 forbidden-definition/import architecture checks and full suite pass. `git diff --check` is clean.

Immediately before this verdict, local `main`, live `origin/main`, and GitHub `main` all resolved to the accepted base `3e62681f05d2f4ee6144cd0b8362e9e0371bf5b0`; the candidate remote ref remained exactly `679cb332b9fc1fde0243c961f4d203973b9ebe36`. No new concurrent main/candidate change was observed. Other implementation, remediation, prior-review, and integration worktrees remained at their recorded SHAs and clean; this review used only its separate worktree.

Review-process limitations: this harness cannot dispatch the installed code-review skill's two parallel fresh-context subagents, so I do not claim that workflow ran. This was a direct fixed-SHA review in a separate Pi session/worktree; GitHub CLI uses the shared repository-owner identity, so the separation is session/worktree-level rather than a separate authenticated account. I did not perform a human visual inspection (author visual acceptance is already recorded). The eight non-failing deprecation warnings remain.

## 7. Acceptance-criteria disposition

| Criterion | Disposition | Evidence |
|---|---|---|
| Exact independently reviewed TOWN-03 implementation, remediation evidence, and final review records present | **PASS** | Exact accepted evidence and final review are ancestors; 21 accepted TOWN paths byte-identical; final report/check script present. |
| Accepted BOX-05 code and numerical contracts unchanged | **PASS** | No BOX-05 path differs from accepted base; focused CPU/CUDA tests pass. |
| BOX-05 and TOWN-03 T3 accounting coexist without omission/overlap or weakened retirement checks | **PASS** | Fixed BOX counts retained; TOWN sets added disjointly; 4 architecture tests pass; 672-node accounting passes. |
| Mathematical and Boolean semantics unchanged from accepted sources | **PASS** | Town equations match approved min/max composition; all existing Boolean/validation function bodies and prior fixtures are source/byte-identical. |
| BOX and Town Inspectors retain accepted behavior | **PASS** | Read-only/mismatch AppTests pass; standalone Town Inspector health/root pass; author visual acceptance remains separately recorded. |
| Full suite, focused BOX/TOWN tests, architecture, compilation, dependency checks | **PASS** | 672 full; 479 focused; 4 architecture; 672 collected; compileall and pip check pass. |
| Approved Town numerical confirmation and actual CPU/CUDA checks rerun independently | **PASS** | Independent replay and remediation verifier pass all 246 inputs and locked thresholds on CPU and actual RTX 2060 CUDA. |
| Protected retirement manifest byte-identical | **PASS** | SHA-256 `b3b283a4…0802ded3`, equal to accepted base. |
| No unauthorized changes, threshold changes, or classifier restoration | **PASS** | Diff scope is bounded; BOX files unchanged; no training/loss/optimizer/classifier paths; thresholds match author signoff. |
| Current main and concurrent changes revalidated | **PASS** | Local/live main equal approved base; candidate ref unchanged; unrelated worktrees preserved. |

## Final verdict

**PASS — `679cb332b9fc1fde0243c961f4d203973b9ebe36` is technically eligible for the already author-authorized integration.** No technical blocker remains within this fixed-SHA review scope. The integration agent may proceed with the authorized merge and post-merge verification. This reviewer did not merge, close #23, alter implementation, or start TOWN-L01.
