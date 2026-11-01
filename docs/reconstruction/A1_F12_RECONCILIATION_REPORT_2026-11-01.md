# A1-F12 Scientific Addendum Reconciliation Report

- **Issue:** [#13 — F12: Reconcile trainable-geometry and optimizer approvals](https://github.com/midhunpradeep/town_model/issues/13), native child of planning map [#2](https://github.com/midhunpradeep/town_model/issues/2)
- **Starting `main` / implementation base:** `a050234ae490d650e28391f180670d1c38183192`
- **Report date:** 2026-11-01
- **Scope:** Documentation and GitHub tracker reconciliation only; no scientific implementation.

## 1. Source provenance and authority

The thesis author approved the F12 reconciliation contract on 2026-11-01. The F12 handoff available to this task reports twelve approvals. This repository record preserves the supplied handoff; it does not claim that a separate mathematics-session transcript, external source URL, or unavailable individually numbered F12.1–F12.8 statements were inspected.

**Grouped-ID limitation:** F12.1–F12.8 are associated collectively with the complete trainable-geometry block in the addendum. The source does not identify one distinct mathematical statement per ID. The addendum deliberately contains no one-to-one ID mapping. F12.9–F12.12 are separately and individually recorded. If original individually numbered F12.1–F12.8 session decisions become available, incorporate their traceability only through a separately reviewed revision.

| Source | Identity | Status and scope |
|---|---|---|
| Original A1 ledger | `docs/reconstruction/A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md`; SHA-256 `f552b60f10f96d9e76585331b99e8543d8b7a81535c261386a93aa697b0489b9`; 34,910 bytes | Immutable authority for 70 F0–F10 decisions; F3.6-R supersedes F3.6. Its audit still covers only this file. |
| F11 addendum | `docs/reconstruction/A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md`; SHA-256 `3a417f74dbfdbee0771dbdf353c9348801a1c35467c64cdd6b9975caabf91c86`; 21,547 bytes | Twelve author-reported approvals; independently accepted/integrated under #12 at `a050234ae490d650e28391f180670d1c38183192`. The F11 review did not review F12. |
| F12 addendum | `docs/reconstruction/A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md`; SHA-256 `a4fd095dd92143c2b70ef6491b97b74a8c47f3e9a874ec71e4ca085115de30cc`; **15,631 bytes** | Twelve author-reported approvals; F12.1–F12.8 grouped, F12.9–F12.12 individual. Submitted under #13 for its own independent review/integration; not implemented or self-accepted. |

The supplied-source frontier is **70 + 12 + 12 = 94 active approvals**. This count does not imply that F12 was implemented, independently reviewed, or scientifically evaluated. The three source records remain separately versioned; neither accepted source was modified or superseded.

## 2. F12 content and scientific boundaries

The addendum preserves all supplied formulas and policies:

- **F12.1–F12.8 grouped geometry block:** bounded `[-1,1]^d` width-first center/half-width construction, physical endpoints, ordinary and hosting-frame width reserves; unbounded `R^d` center/half-width transform; contained-window availability, half-width and center equations; independent-window ordinary transform and shared F3.6-R semantics; seeded label-agnostic physical initialization and round-trip/diagnostic sequence; configurable `epsilon = 1e-4` / `delta = 1e-4`, float32, hard validation, invalid-state rejection, and predeclared ULP-based matching requirement. See addendum §§2.1–2.6, especially [bounded geometry](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md#L40), [contained windows](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md#L70), [initialization](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md#L88), and [numerical policy](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md#L101).
- **F12.9:** all-or-nothing effective optimizer update across model, optimizer moments/state, scheduler, and relevant progress/counters; mechanism and fault injection deferred to implementation.
- **F12.10:** reject and terminate on confirmed invalidity; record failure and last complete checkpoint; prohibit counting, scheduler advancement, skip-and-continue, automatic LR reduction, geometry repair/clamping, optimizer reset-and-continue; resume only explicitly from a validated complete checkpoint. Keep separate from gradient saturation/conditioning.
- **F12.11:** fresh optimizer, scheduler, and AdamW moments for a new stage with stage monitoring; complete model/optimizer/scheduler/RNG/progress restoration and checkpoint-completeness validation for same-stage resume.
- **F12.12:** all six starting learning rates and other defaults (AdamW `(0.9, 0.999)`, epsilon `1e-8`, neural decay `0.01` with bias/norm exclusions, raw-geometry decay `0`, constant schedule, clip norm `1.0`, float32, accumulation `1`). These are configurable starting points, not established optima, and do not specify actual stage-specific trainability or authorize detector development. See addendum §§3–6 and [the defaults tables](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md#L165).

## 3. Readiness reconciliation

| P0 item | State at starting `main` | Reconciled state | Remaining boundary |
|---|---|---|---|
| **P0-A — Trainable geometry** | Exact bounded/unbounded transforms, window parameterizations, and initialization were unresolved scientific inputs. | **Scientifically specified; engineering implementation and numerical validation required.** | Measure declared operating envelopes, ULP matching budget, initialization robustness, gradients, and numerical extremes. Do not treat epsilon/delta starts as universal validation. |
| **P0-B — Atomic optimizer lifecycle** | Invalid-update rejection was approved in principle, but transaction, failure, and recovery details were unsettled. | **Scientifically specified; implementation and adversarial fault-injection verification required.** | Prove all-or-nothing model/optimizer/scheduler/progress state; verify terminal disposition, fault injection, and recovery from incomplete/corrupted as well as last complete checkpoints. |
| **P0-C — Stage transitions and defaults** | Stage optimizer/scheduler transitions and initial numeric policy were unresolved. | **Scientifically specified for an initial training lifecycle; engineering implementation and review required.** | Implement fresh state at stage boundaries and exact complete same-stage restore; final stage-specific monitors, detector policies, and comparison-quality experiments remain separately scoped. |

The [readiness frontier](A1_IMPLEMENTATION_READINESS.md) remains derived planning. It preserves unresolved classifier interfaces: flat point input, class-order ownership, typed outputs, `DomainThing`, empty batches, bounded/unbounded point-domain behavior, and zero-window integration. These were not answered by F12.

### Future engineering acceptance frontier

Later implementation/test tasks must establish: (1) valid geometry over declared tested ranges; (2) bounded/unbounded and contained/independent invariants; (3) inverse/forward correctness and physically matched initialization; (4) finite corrective point/geometry gradients; (5) exact hard membership separate from signed clearance; (6) atomic invalid-update handling including moments/state and complete, incomplete, and corrupted-checkpoint recovery; (7) fresh transition state versus complete same-stage continuation; and (8) controlled-fixture loss decrease in the five primary conditions. Mathematical correctness, numerical validity, optimization success, and scientific generalization remain distinct evidence categories.

## 4. Primary scientific invariants and optional boundaries

The five primary conditions remain exactly, with **unregularized** primary objectives:

1. Affine baseline + BCE-with-logits.
2. Box + signed logistic.
3. Box + signed hinge.
4. Town + signed logistic.
5. Town + signed hinge.

No compactness penalty, auxiliary point attraction, ontology-axiom loss, or supplementary reference-loss penalty is added. F3.6-R remains authoritative: a house is a closed frame minus ambient open window interiors; a town is a union of houses. Exact Boolean membership remains distinct from signed coordinate clearance and its exact min/max numerical compositions.

Supplementary reference-loss conformance, compactness, auxiliary point organization, fixed-point/higher-order ontology benchmarks, broad capacity/topology theorems, final supervisor-dependent RQ2 aggregation, and optional affine diagnostics remain non-blocking for initial pure classifier reconstruction. Detector/ROI policy, statistical methodology, and final comparison-quality evaluation remain separate dependencies for the stages and claims they affect. No R2, detector, classifier, or training implementation is authorized or performed here.

## 5. Repository and tracker reconciliation

The readiness document now states the three required P0 statuses and their outstanding engineering scope. Current authority/navigation documents expose the separate A1/F11/F12 source identities, 94-approval source-level count, grouped F12-ID limitation, and distinct F12 review status. The current #2 planning map is updated after the implementation SHA is fixed; it remains open and is not implementation authorization.

Issue roles remain separate: #2 is the open Wayfinder map; #3 remains open pending explicit author confirmation; #10 is accepted R1; #12 is accepted F11; #11 remains a separate lifecycle-status follow-up and is not implemented by #13; #13 is the F12 documentation/review task. A coordination comment about shared readiness/navigation paths is recorded on #11; its body, state, and labels are not changed by this work.

## 6. Changed-file inventory

Exactly these nine Markdown/documentation paths are in scope:

1. `AGENTS.md`
2. `README.md`
3. `docs/ACCEPTANCE_INDEX.md`
4. `docs/CURRENT_PROJECT_STATUS.md`
5. `docs/agents/DECISION_AUTHORITY.md`
6. `docs/agents/EXECUTION_PROTOCOL.md`
7. `docs/reconstruction/A1_IMPLEMENTATION_READINESS.md`
8. `docs/reconstruction/A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md`
9. `docs/reconstruction/A1_F12_RECONCILIATION_REPORT_2026-11-01.md`

The original A1 ledger, F11 addendum, completeness audit, both accepted independent reports, all historical reviews, executable source/tests, datasets, snapshots, models, and experiment artifacts are protected and unchanged. Runtime tests are not required for this documentation-only change and were not run or claimed.

## 7. Independent verification instructions

Review the exact implementation SHA linked in the final #13 handoff, not a moving branch. Reproduce and inspect:

1. `sha256sum docs/reconstruction/A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md docs/reconstruction/A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md docs/reconstruction/A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md` and `wc -c` for all three files; compare with §1.
2. F12 heading inventory: one grouped `F12.1–F12.8` geometry section and separate F12.9, F12.10, F12.11, F12.12 sections. Verify there is no individual F12.1–F12.8 mapping.
3. Compare each equation, reserve, initialization step, numerical invariant, optimizer disposition, stage transition rule, and numeric default in addendum §§2–6 against the supplied #13 contract. Check especially both domain transforms, both contained-window equations, all six LR values, AdamW settings, weight-decay exclusions, clipping, dtype, and accumulation.
4. Verify P0-A/B/C previous/current states against the readiness changes and check its other interface/optional boundaries for contradictions.
5. Inspect the exact base-to-implementation diff inventory; verify no source, history, code, test, data, experiment, or model paths changed. Run `git diff --check <base>..<implementation>`.
6. Resolve local Markdown links and inspect the pinned current #2 source/status map. Query live #2/#3/#10/#11/#12/#13 state, labels, ownership, and native blockers; #11 must remain a separate task.
7. Read the accepted R1 and F11 independent reports for their scope limits. Neither is evidence that F12 itself was reviewed.

## 8. Acceptance-criteria evidence

- **AC1 — Original authority integrity:** original A1 and F11 hashes/sizes are recorded in §1; the implementation diff excludes both files and all accepted reports. Recompute hashes independently.
- **AC2 — F12 provenance fidelity:** §§1–2 preserve author-reported approval/date, handoff limitation, grouped F12.1–F12.8 block, absent per-ID traceability, and individually identifiable F12.9–F12.12. No unavailable source is claimed as inspected.
- **AC3 — Geometry correctness:** §2 of the addendum transcribes bounded/unbounded, contained/independent, reserve, and initialization requirements; formula references are in §2 of this report.
- **AC4 — Numerical policy:** addendum §2.6 preserves epsilon/delta starts, float32, hard constraints, stored-geometry validation, reserve-constrained ULP matching, and records without inventing tolerances or operating envelopes.
- **AC5 — Optimizer policy:** addendum §§3–6 record F12.9–F12.12 separately, including atomicity, terminal failure, stage/resume distinction, and every numeric default.
- **AC6 — Readiness reconciliation:** §3 documents previous versus current P0-A/B/C states and remaining engineering verification; the derived readiness file matches those statuses.
- **AC7 — Scientific invariants:** §4 preserves all five unregularized conditions, F3.6-R, Boolean membership, and separate signed clearance/min-max composition; optional additions remain out of the primary objective.
- **AC8 — Authority/workflow:** §§1 and 5 distinguish original A1, accepted F11, F12 source approval, repository-review state, and #2/#3/#10/#11/#12/#13 roles. The live tracker is checked again at submission.
- **AC9 — Scope isolation:** §6 lists only documentation paths; the exact diff is checked for protected files and absence of executable/code/data/experiment/model/history changes.
- **AC10 — Reproducibility:** §§1, 2, 6, and 7 provide exact hashes/sizes, formula/default references, inventory, source/issue pointers, and independent-verification instructions. The final #13 handoff pins the implementation SHA and verification results.

## 9. Verification record and limitations

The final #13 submission comment records commands and actual results for source hashes and byte counts, formula/default checks, grouped-ID audit, readiness consistency, links, exact changed-path/protected-path inventory, `git diff --check`, and live issue/conflict checks. No runtime regression, model, optimization, proof, or scientific generalization test is claimed. The principal provenance limitation remains the unavailable per-ID association for F12.1–F12.8; numerical operating envelopes and budgets remain unmeasured engineering obligations, and unresolved classifier interfaces and later-stage scientific policies remain author-controlled.
