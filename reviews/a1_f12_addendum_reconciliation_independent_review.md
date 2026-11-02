# Independent Review — A1 F12 Addendum Reconciliation

## Verdict

**ACCEPT F12**

All AC1–AC10 pass. No blocking violation or materially insufficient evidence was found. Acceptance is limited to faithful documentation/tracker reconciliation of the fixed F12 revision; it does not validate the underlying mathematics, implement the approved policies, or authorize geometry, optimizer, classifier, detector, training, or R2 work.

## Review identity, exact target, and independence

- **Review date:** 2026-11-02.
- **Repository / issue:** `midhunpradeep/town_model`, [#13 — F12: Reconcile trainable-geometry and optimizer approvals](https://github.com/midhunpradeep/town_model/issues/13).
- **Base:** `a050234ae490d650e28391f180670d1c38183192`.
- **Exact implementation reviewed:** `a9c988ffd80b16a5ebcffa7a022f0aa86fa7490e` on `docs/a1-f12-scientific-addendum-reconciliation`.
- **Ancestry:** the implementation is a direct child of the stated base. The exact remote implementation branch resolves to that SHA; remote `main` remains at the base.
- **Review worktree/branch:** clean isolated worktree `/tmp/town-model-f12-independent-review`, branch `review/a1-f12-independent-review`, checked out at the exact implementation SHA before this report was added. The implementation commit and submitted worktree were not modified.
- **Reviewer independence:** this is a separate Pi agent session (`01a199a0-f9d9-7682-a778-09dd649b3f54`) and isolated review worktree; I did not implement F12. GitHub authentication uses the same repository-owner account as the implementation journal, so this is not a claim of a separately authenticated human identity. The review/report is committed on the separate review branch.
- **Environment:** Linux x86_64, Git 2.34.1, Python 3.10.12, GitHub CLI 2.55.0.
- **Issue state at review start:** #13 was OPEN with its sole lifecycle label `workflow:in-review`; its native parent is #2 and it has no open native blockers.

## Baseline, source integrity, and provenance

Independently computed hashes and byte sizes in the exact target worktree:

| Scientific source | SHA-256 | Bytes | Verification against accepted source |
|---|---|---:|---|
| `docs/reconstruction/A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md` | `f552b60f10f96d9e76585331b99e8543d8b7a81535c261386a93aa697b0489b9` | 34,910 | Git blob is identical at accepted R1 implementation `be4bbc067d235cbd896665ba33f6585b2cabc4e6`, R1 review/integration `100d04d4ecae4f58f525bbdac76b87bf28026d5c`, base, and target. |
| `docs/reconstruction/A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md` | `3a417f74dbfdbee0771dbdf353c9348801a1c35467c64cdd6b9975caabf91c86` | 21,547 | Git blob is identical at accepted F11 implementation `96caedd03bf8c970d1592ce2e82c1e1506756dd5`, F11 review/integration/base `a050234ae490d650e28391f180670d1c38183192`, and target. |
| `docs/reconstruction/A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md` | `a4fd095dd92143c2b70ef6491b97b74a8c47f3e9a874ec71e4ca085115de30cc` | 15,631 | Independently recomputed; matches the required addendum identity. |

I executed the original completeness-audit Python script from `A1_DECISION_COMPLETENESS_AUDIT.md` against the exact target. It reports **70 unique active approvals**, with counts F0=1, F1=1, F2=8, F3=11, F4=4, F5=6, F6=5, F7=8, F8=5, F9=5, F10=16; F3.6-R active and F3.6 superseded; 12 numbered open gates; and A–J. Thus the source-level frontier is **70 + 12 + 12 = 94**, not a claim that all 94 were reviewed together or implemented. The original ledger/audit and accepted F11 source remain separate and unchanged; F12 is a new separate addendum.

The supplied handoff identifies F12.1–F12.8 collectively as one trainable-geometry block. The addendum preserves that entire block without assigning any formula or requirement to an individual ID, explicitly discloses the traceability gap, and does not claim an unavailable transcript, source URL, proof, or external source was inspected. F12.9–F12.12 are separately identifiable. The author-reported approval date is 2026-11-01.

## Formula and numerical-policy verification

I compared the addendum directly with the approved F12 handoff in issue #13 and ran an independent literal assertion check of 37 formula, policy, and default items.

- **Bounded domain:** `[-1,1]^d`; physical endpoints `l = c - h`, `r = c + h`; width-first `h = [w_min + (2 - w_min) sigmoid(a)] / 2`, `c = (1 - h) tanh(b)`. Ordinary boxes use `w_min = epsilon`; hosting frames with windows use `w_min = epsilon + 2delta`. The addendum does not substitute lower-corner parameterization, clipping, or a different transform.
- **Unbounded domain:** `c = b`, `h = w_min/2 + softplus(a)` for `R^d`; no artificial finite domain is introduced.
- **Contained windows:** `a_available = h_F - delta - epsilon/2`; `h_W = epsilon/2 + a_available sigmoid(u)`; `c_W = c_F + (h_F - delta - h_W) tanh(v)`. The physical parent-frame relationship and reserves are retained.
- **Independent windows:** ordinary domain-specific transform; may extend beyond the frame; same F3.6-R Boolean membership semantics, with out-of-frame window interiors not removing points outside a frame.
- **Initialization:** seeded, label-agnostic feasible physical geometry → inverse transform → forward decode → validate actual decoded physical geometry → compare physically matched starts, not raw parameters → record physical corners, matching errors, occupancy, and dimension-aware window-exclusion diagnostics. No inverse algorithm, sampling distribution, or matching budget is invented as approved.
- **Numerical policy:** configurable initial `epsilon=1e-4` and `delta=1e-4`; float32 transforms/losses/evaluation; independent validation of stored corners, widths, domain bounds, and contained-window clearances; hard constraints without tolerance relaxation; declared/tested operating envelopes; invalid geometry distinguished from saturation/conditioning; invalid states rejected without automatic repair/clamping; predeclared reserve-constrained ULP inverse/forward matching; record actual initial geometry, matching error, and occupancy. The addendum explicitly leaves envelopes, ULP budgets, inverse algorithms, gradient behavior, and numerical extremes for measured engineering verification and does not claim they have been empirically validated.

## F12.9–F12.12 verification

- **F12.9:** all-or-nothing effective update includes model parameters, optimizer moments/state, scheduler advancement, and relevant counters/progress. Staged commits, complete rollback, and fail-closed recovery are mechanisms for later engineering; no partial effective step is accepted.
- **F12.10:** confirmed numerical invalidity rejects the update and terminates the run; records failure reason, affected parameters, and last complete valid checkpoint; does not count the step, advance the scheduler, skip-and-continue, auto-reduce LR, clamp/repair geometry, or reset optimizer state and continue; resume requires explicit action from a validated complete checkpoint. Saturated/poorly conditioned gradients alone are kept distinct from invalid geometry.
- **F12.11:** a new stage transfers selected model parameters/required module state and constructs a fresh optimizer and scheduler with fresh AdamW moments and stage monitoring. Same-stage resume restores complete model, optimizer, scheduler, RNG, and progress/counters after checkpoint-completeness validation. The two operations are not conflated.
- **F12.12:** all six LRs are present: backbone/FPN `1e-5`; detector heads `1e-4`; point projection, composition/fusion, affine classifier, and raw box/town geometry each `1e-3`. Other defaults: AdamW betas `(0.9,0.999)`, epsilon `1e-8`, neural decay `0.01` excluding bias and normalization parameters, raw geometry decay `0`, constant initial schedule, global clipping norm `1.0`, float32, and accumulation `1`. They are described as configurable engineering starts, not optimal values; no detector development is authorized.

## Readiness and scientific invariants

The exact target reclassifies:

- **P0-A:** **Scientifically specified; engineering implementation and numerical validation required.** Measured operating envelopes, ULP matching budgets, initialization robustness, gradients, and numerical extremes remain obligations.
- **P0-B:** **Scientifically specified; implementation and adversarial fault-injection verification required.** End-to-end atomicity must cover optimizer moments/state, scheduler, counters/progress, and complete/incomplete/corrupted checkpoint recovery.
- **P0-C:** **Scientifically specified for an initial training lifecycle; engineering implementation and review required.** Stage transitions, complete state restoration, and training verification remain; final stage-specific monitors, detector/ROI policy, and final statistical/comparison protocol remain separately scoped.

The readiness report retains unresolved classifier boundaries (flat point input, class-order ownership, typed outputs/`DomainThing`, empty batches, bounded/unbounded point-domain behavior, zero-window integration), says pure evaluators are not blocked by unrelated optimizer requirements, and leaves candidate work planning-only—not `workflow:ready` or independently accepted. Optional reference-loss conformance, compactness, point organization, fixed-point/higher-order benchmarks, broad capacity/topology work, optional affine diagnostics, and final RQ2 aggregation remain separate from initial pure evaluator/reconstruction scope.

The exact five primary conditions remain: (1) affine + BCE-with-logits; (2) box + signed logistic; (3) box + signed hinge; (4) town + signed logistic; (5) town + signed hinge. Primary objectives remain unregularized. No compactness, auxiliary point-attraction, ontology-axiom, or supplementary reference-loss penalty is introduced. F3.6-R remains authoritative: a house is a closed frame minus the union of ambient open window interiors; a town is a union of houses. Hard Boolean membership remains separate from signed coordinate clearance and exact min/max score compositions. Detector development, optional studies, final RQ2 protocols, and R2 are not implemented or authorized.

## GitHub issue state and #11 boundary

Live tracker checks independently confirmed:

- **#2:** OPEN, `wayfinder:map`; current body links the exact submitted F12 addendum and implementation SHA `a9c988ffd80b16a5ebcffa7a022f0aa86fa7490e`, reports 94 source-level approvals, and preserves F12 review-pending status.
- **#3:** OPEN, `wayfinder:grilling`; explicit author confirmation remains pending. Not closed or resolved by this review.
- **#10:** CLOSED, `workflow:accepted` (R1 only).
- **#12:** CLOSED, `workflow:accepted` (F11 only).
- **#11:** OPEN, unassigned, `workflow:defined`, zero open blockers. F12 changes overlap the seven current-facing paths named by #11: `AGENTS.md`, `README.md`, `docs/ACCEPTANCE_INDEX.md`, `docs/CURRENT_PROJECT_STATUS.md`, `docs/agents/DECISION_AUTHORITY.md`, `docs/agents/EXECUTION_PROTOCOL.md`, and `docs/reconstruction/A1_IMPLEMENTATION_READINESS.md`. The overlap is disclosed in the F12 issue/report and in #11 coordination comments. #11's body, labels, and lifecycle were not changed or accepted by F12; remaining lifecycle-status reconciliation can proceed separately.
- **#13:** OPEN, `workflow:in-review` at review start, assigned to the implementer, native child of #2, no open native blockers.

The accepted R1 and F11 independent reports were read. They establish only their own scopes; neither is treated as an F12 review.

## Exact scope and executed checks

The complete base-to-target diff contains exactly these nine documentation paths:

1. `AGENTS.md`
2. `README.md`
3. `docs/ACCEPTANCE_INDEX.md`
4. `docs/CURRENT_PROJECT_STATUS.md`
5. `docs/agents/DECISION_AUTHORITY.md`
6. `docs/agents/EXECUTION_PROTOCOL.md`
7. `docs/reconstruction/A1_IMPLEMENTATION_READINESS.md`
8. `docs/reconstruction/A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md`
9. `docs/reconstruction/A1_F12_RECONCILIATION_REPORT_2026-11-01.md`

The changed-path comparison is exact. No original scientific source/audit, accepted review report, executable source/tests, dataset, checkpoint, experiment result, historical specification, or retirement manifest changed. No implementation or runtime test result is claimed.

Checks actually executed in `/tmp/town-model-f12-independent-review`:

- `sha256sum` and `wc -c` for all three scientific sources — required hashes and sizes match.
- Embedded original completeness-audit Python — PASS; 70 active IDs and F3.6-R supersession confirmed.
- Git blob identity comparisons for original A1 across accepted R1/base/target and F11 across accepted F11/integration/base/target — PASS.
- Independent F12 grouped/individual heading check, 37 literal formula/policy/default assertions, and F12 hash/size check — PASS.
- `git diff --name-status`/exact expected inventory and protected-path comparison — PASS; exactly nine documentation files, no protected-path changes.
- `git diff --check a050234ae490d650e28391f180670d1c38183192 a9c988ffd80b16a5ebcffa7a022f0aa86fa7490e` — PASS.
- Local Markdown inline-link/fragment scan over the nine changed documents — **120 checked, zero failures**.
- `git merge-base --is-ancestor` and parent check — PASS; target is a direct child of base.
- `git ls-remote` — submitted branch matches target; remote `main` remains at base.
- Live `gh issue view` and GraphQL checks for #2, #3, #10, #11, #12, #13 — statuses, labels, parent, and blocker state as stated above.

No runtime tests were run because the revision is documentation-only. No model, geometry, optimizer, proof, GPU, or scientific-generalization claim is made.

## AC1–AC10 verdict matrix

| Criterion | Verdict | Evidence |
|---|---|---|
| **AC1 — Original A1/F11 integrity** | **PASS** | Independently recomputed all three hashes/sizes; original audit reproduces 70 IDs/F3.6-R; accepted A1/F11 Git blobs are unchanged through target. |
| **AC2 — F12 provenance fidelity** | **PASS** | 2026-11-01 author-reported date; complete grouped F12.1–F12.8 block; no individual mapping invented; limitation explicit; F12.9–F12.12 separate; no unavailable transcript/source/proof claimed. |
| **AC3 — Geometry and initialization** | **PASS** | All bounded/unbounded and contained-window equations, reserve distinction, independent-window semantics, and six initialization steps match the handoff. |
| **AC4 — Numerical policy** | **PASS** | Epsilon/delta, float32, hard checks, stored physical-state validation, reserve-constrained ULP policy, invalid-state disposition, and future measured-validation limits are preserved without invented constants or validated-envelope claims. |
| **AC5 — Optimizer, lifecycle, defaults** | **PASS** | F12.9–F12.12 policies and all six LRs plus AdamW/decay/schedule/clipping/dtype/accumulation settings are present and correctly scoped as starting defaults. |
| **AC6 — P0-A/B/C readiness** | **PASS** | Required statuses and continuing engineering/fault-injection/training obligations are explicit; detector/statistics/monitor boundaries remain separate; pure evaluators are not blocked by unrelated optimizer work. |
| **AC7 — Scientific invariants** | **PASS** | Exact five unregularized conditions, F3.6-R, Boolean-membership/clearance distinction, and optional-work boundaries are preserved. |
| **AC8 — Authority and workflow** | **PASS** | Current documents and #2 point to the exact F12 revision; live #2/#3/#10/#11/#12/#13 statuses and separate roles match the requested lifecycle. #11 remains separate. |
| **AC9 — Scope isolation** | **PASS** | Exact nine-file documentation-only diff; protected authority, review/history, executable, data, model, and experiment paths unchanged. |
| **AC10 — Reproducibility** | **PASS** | Exact SHA/ancestry, source hashes/counts, formulas/defaults, 120 local references, issue states, inventory, and `git diff --check` independently reproduced. |

## Limitations and final disposition

The supplied geometry provenance does not provide individual F12.1–F12.8 statements; the addendum correctly preserves that source limitation rather than manufacturing traceability. The review verifies faithful reconciliation, not the mathematical proof of the formulas or their empirical numerical behavior. Operating envelopes, ULP budgets, inverses, gradients, fault-injection mechanisms, stage implementation, and training remain future engineering work. Final detector/ROI policy, final statistical evaluation, final stage-specific scientific monitors, unresolved classifier interfaces, and RQ2 protocols remain separately scoped/author-controlled. Issue #11 remains OPEN/`workflow:defined` and is not accepted or closed by this review. No integration into `main` is performed.

**Final verdict: ACCEPT F12.**
