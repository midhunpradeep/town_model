# A1 Implementation Readiness and Scientific Dependency Frontier

**Role:** Derived planning frontier, not a scientific specification or implementation authorization. Scientific authority is versioned: the immutable [original A1 ledger](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md) is canonical for F0–F10; the separate [F11 addendum](A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md) records twelve post-R1 approvals and was independently accepted/integrated under #12; the separate [F12 addendum](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md) records twelve author-reported approvals, with F12.1–F12.8 grouped. F12 documentation was independently accepted under [#13](https://github.com/midhunpradeep/town_model/issues/13) and integrated at [`5686d60bdb3e309924d37eb600aa75fb93855340`](https://github.com/midhunpradeep/town_model/commit/5686d60bdb3e309924d37eb600aa75fb93855340). Candidate scopes below are not `workflow:ready` tickets.

## Authority, provenance and status dimensions

The original A1 ledger contains 70 active approvals and remains byte-identical (SHA-256 `f552b60f10f96d9e76585331b99e8543d8b7a81535c261386a93aa697b0489b9`, 34,910 bytes). The author-reported F11 and F12 handoffs dated 2026-11-01 each supply twelve additional approvals. The supplied-source scientific frontier is **70 + 12 + 12 = 94 active approvals**. R1/#10 was independently accepted and integrated at `100d04d4ecae4f58f525bbdac76b87bf28026d5c`; its review did not cover F11 or F12. F11/#12 was independently accepted and integrated at `a050234ae490d650e28391f180670d1c38183192`. F12 is recorded separately in the [F12 addendum](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md), independently accepted under [#13](https://github.com/midhunpradeep/town_model/issues/13) and integrated at [`5686d60bdb3e309924d37eb600aa75fb93855340`](https://github.com/midhunpradeep/town_model/commit/5686d60bdb3e309924d37eb600aa75fb93855340). F12.1–F12.8 are reported collectively; no per-ID formula mapping is asserted. The original [completeness audit](A1_DECISION_COMPLETENESS_AUDIT.md) still audits only the original 70-decision source and is not rewritten.

| Dimension | Current state |
|---|---|
| **Scientific** | Author-reported approval: 70 original F0–F10 decisions plus twelve F11 and twelve F12 approvals (94 total). F3.6 remains superseded by F3.6-R; F11 and F12 supersede no original decisions. F12.1–F12.8 are grouped without a per-ID mapping. Audit A–J and other omitted formal/experimental details remain open where stated. |
| **Implementation** | No reconstructed classifier, geometry transform/inverse, objective, optimizer transaction, detector, or F12 evaluator has been implemented. Candidate mathematical slices below are planning only. The accepted T0–T5 classifier-free baseline remains operational. |
| **Independent review** | R1 (#10) is accepted for its bounded documentation scope; it did not review F11/F12. F11 (#12) and F12 (#13) each have a separately accepted/integrated documentation record, with F12 integrated at `5686d60bdb3e309924d37eb600aa75fb93855340`. Scientific approval, implementation completion, and independent acceptance are distinct. |

The F11 source handoff contains concise approval summaries, not the original mathematics-session transcript or formal derivations. The F12 source handoff reports the geometry block collectively for F12.1–F12.8 without per-ID traceability. Neither gap authorizes filling in missing definitions, theorem hypotheses, algorithms, or experimental protocols by inference.

## Primary comparison and semantic guardrails

The five primary conditions remain exactly:

1. Affine baseline + BCE-with-logits.
2. Box + signed logistic.
3. Box + signed hinge.
4. Town + signed logistic.
5. Town + signed hinge.

Primary objectives remain unregularized. Do not add compactness penalties, auxiliary point attraction, ontology-axiom penalties, or supplementary reference losses to the primary comparison.

F3.6-R remains authoritative: a house is a closed frame minus the union of ambient open window interiors; a town is a union of houses. Exact Boolean membership determines geometric predictions. Signed coordinate clearance is a separate numerical training measurement, composed through the approved exact min/max rules without hidden normalization. Do not infer implementation mechanics from retired classifiers.

F11 reinforces, but does not fully operationalize, distinctions among OWL entailment, verified generator interpretations and finite-sample evidence; emergent geometry and explicit relation penalties; structural invariants and raw coordinates; relation classes; exact certificates and tolerance estimates; learning questions and feasibility levels; and held-out image, compositional and structural generalization. See the [F11 addendum and its source-linked A–J crosswalk](A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md#L98).

## Implementation-critical

These decisions are required for the corresponding trainable modules to exhibit mathematically correct, numerically valid, or trainable behavior. They are **not** prerequisites for unrelated pure evaluators whose contracts explicitly take supplied physical endpoints/points and exclude trainable parameterization, optimization, and optimizer state.

### P0-A — Trainable geometry

**Scientifically specified; engineering implementation and numerical validation required.** F12.1–F12.8 collectively approve the bounded width-first transform, unbounded center/half-width transform, contained and independent window modes, and seeded label-agnostic physical initialization sequence. The exact equations and `epsilon`/`delta` starting values are in the [F12 addendum](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md#L40). Actual numerical operating envelopes, predeclared ULP matching budgets, initialization robustness, corrective gradients, and numerical extremes still require measured verification. The configurable starting constants are not validated universally across dimensions/capacities. These requirements block trainable parameterization/initialization and their numerical validation, but not physical-corner Boolean membership or other pure evaluators with explicit valid-input contracts.

### P0-B — Atomic optimizer lifecycle

**Scientifically specified; implementation and adversarial fault-injection verification required.** F12.9–F12.10 approve all-or-nothing effective updates across model parameters, optimizer moments/state, scheduler advancement, and relevant counters/progress, plus terminal disposition on confirmed numerical invalidity. Later engineering must prove the transaction across all these state components and verify failure recording and explicit recovery from the last complete valid checkpoint. Verification must include incomplete/corrupted-checkpoint recovery and adversarial fault injection. The mechanism is not implemented by the addendum. These requirements block corresponding training/update modules, not pure evaluators that do not perform optimizer updates.

### P0-C — Stage transitions and initial defaults

**Scientifically specified for an initial training lifecycle; engineering implementation and review required.** F12.11–F12.12 establish fresh optimizer/scheduler/AdamW state at a new stage, complete model/optimizer/scheduler/RNG/progress restoration for same-stage resumption, and configurable initial learning-rate/optimizer/dtype/clipping/accumulation policies. The exact defaults are in the [F12 addendum](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md#L165). Implementations must keep parameter-group policy distinct from stage-specific trainability. Final stage-specific scientific monitors, detector policies, and comparison-quality experiments remain separately scoped. This blocks corresponding stage orchestration/configuration, not unrelated deterministic mathematical evaluators.

### Classifier-facing integration edges — separate unresolved boundary

Do not merge these integration questions into P0-A/B/C or silently derive answers from retired classifiers:

- the classifier adapter's flat point input contract (`[N,d]`);
- ownership and ordering of class outputs/class IDs;
- typed output boundaries, including `DomainThing`;
- point-domain behavior, including bounded/unbounded (`Δ=ℝ^d`) integration;
- empty-batch behavior;
- zero-window behavior at classifier/integration boundaries (while F3.11's mathematical `W=0` and single-box equivalence remain approved).

A pure evaluator or a point-only affine slice may be planned only with these edges explicitly scoped out or supplied by an independently approved caller contract. This frontier does not implement or ready that integration work.

## Future engineering acceptance frontier

These are later implementation/test obligations, not acceptance evidence for this documentation task:

1. Valid physical geometry across declared, measured operating ranges.
2. Bounded/unbounded and contained/independent-window invariants.
3. Inverse/forward correctness and physically matched initialization.
4. Finite, corrective gradients through point and geometry parameters.
5. Exact hard membership distinct from signed-clearance scores.
6. Atomic invalid-update handling across optimizer moments/state, scheduler, counters/progress, and checkpoint recovery, including incomplete/corrupted checkpoints and adversarial fault injection.
7. Fresh optimizer/scheduler/moments at stage transitions versus exact complete same-stage continuation, including RNG and progress.
8. Controlled-fixture loss decrease in the five primary classifier conditions.

Report mathematical correctness, numerical validity, optimization success, and scientific generalization as distinct evidence categories.

## Evaluation-critical

These items are required before defensible RQ1/RQ2 conclusions or comparison-quality generalization claims. An approved foundational principle does not mean the complete evaluation protocol is approved.

- **Relations and references:** formal conditions and population for the F11.5 taxonomy; typed pair population; evidence mapping/provenance across OWL entailment, verified generator interpretations and finite samples; unknown handling; relation labels and aggregation. F11.6 preserves evidence distinctions but does not supply the protocol.
- **Exact certificates:** sound algorithms and arithmetic assumptions for sets defined by finite stored physical endpoints; resource/unknown policy; separately labeled tolerance estimates. F11.8 does not choose the certificate algorithm or tolerance method.
- **Statistical units and experiment populations:** independent image/recipe/asset and relation-pair units, predeclared denominators, seed handling, uncertainty, split/selection rules, and the populations used for each claim.
- **Generalization protocols:** separately define held-out image, compositional and structural generalization populations, measurements and selection controls. F11.7 names the distinct questions but not their operational tests.
- **RQ1/RQ2 comparison scope:** the primary five conditions remain fixed, but final dimensions/capacities, populations, analysis, relation references, metrics and claim thresholds are not supplied by this addendum. Do not decide final methodology here.
- **Feasibility evidence:** F11.3/F11.11 require separate learning and simultaneous-feasibility questions; operational criteria and diagnostics remain to be specified before using them to explain outcomes.

F11.12 authorizes optional affine structural diagnostics while preserving box–town as the main RQ2 comparison. Those diagnostics do not replace or preempt the remaining protocol.

## Optional experiments — not primary prerequisites

Keep these supplementary and separate from the five primary classifier conditions:

- compactness studies;
- auxiliary point organization;
- fixed-point diagnostic benchmarks;
- higher-order intersection and capacity benchmarks;
- supplementary affine structural diagnostics;
- additional reference losses and ablations.

Approving a theorem, diagnostic category, or optional study does not make its experimental protocol complete or turn it into a primary prerequisite. In particular, the F11.9 higher-order intersection benchmark, F11.10 capacity witnesses, F11.12 affine diagnostics, F10.3 fixed-point benchmark, compactness work, and reference losses must not add hidden terms or conditions to the primary comparison.

Supplementary reference-loss conformance, compactness experiments, auxiliary point organization, fixed-point and higher-order ontology benchmarks, broad capacity/topology theorems, final supervisor-dependent RQ2 aggregation, and optional affine structural diagnostics are not blockers for initial pure classifier reconstruction. Detector/ROI policy, statistical methodology, and final comparison-quality evaluation remain separate dependencies for the stages and claims they affect.

## Deferred theory

Broad topology characterizations and general ontology satisfiability investigations may remain outside the immediate reconstruction path when they are not needed for the specified experiments. They remain in the scientific record (including the original A1 F10.5 and foundational audit A–J) and are not resolved, deleted, or converted into implementation prerequisites by this frontier.

## Narrow mathematical slices suitable for separately scoped planning

The following have sufficient scientific definitions for a future bounded task to be proposed, subject to its own contract, dependency check, and independent review. **They are planning candidates only; none is implemented, automatically authorized, or made `workflow:ready` here.** A pure-evaluator contract must explicitly exclude trainable transforms/parameterization, optimization, and optimizer state; do not block such an evaluator on P0-A/B/C if those items are outside its inputs and outputs.

| Candidate slice | Scientific basis | Boundary that its own contract must state |
|---|---|---|
| Physical-corner box Boolean membership | F3.2, F3.4, F7.4 | Supplied physical endpoints and point-domain preconditions; exact closed faces; no raw-to-physical transform. |
| House/window membership and town union | F3.2, F3.5, F3.6-R | Supplied physical corners; strict ambient open-window interiors; exact Boolean membership; no parameterization or optimization. |
| Zero-window and single-box equivalence | F3.11 | `W=0` and `H=1,W=0` on pure supplied geometry; do not infer integration behavior for empty batches or classifier adapters. |
| Signed coordinate-clearance evaluation | F5.1, F5.6 | Coordinate clearance in raw physical units, separate from membership and not Euclidean distance; explicit finite input assumptions. |
| Exact min/max composition | F5.3, F5.5 | Approved composition and framework AD tie behavior/diagnostics only where gradients are in scope; no soft replacement or hidden normalization. |
| Signed logistic and hinge penalty kernels | F6.1–F6.3 | Pure supplied signed-clearance/label inputs and explicit temperature/margin parameters; no implicit regularization, reducer, optimizer, or default selection. |
| Flat point-only affine classifier behavior | F4.1–F4.4, F11.1 | Keep class-order ownership, point-domain behavior, empty inputs and `W=0` integration edges explicitly scoped; do not couple to image infrastructure or infer unresolved adapters. |
| Deterministic boundary, degeneracy and fixed-point fixtures | F3.2, F3.4, F3.6-R, F3.10, F10.3 | Fixed physical-corner/point fixtures only; degenerate evaluator fixtures do not relax positive trainable widths or decide comparison-quality experiments. |

## Foundational audit A–J status

All ten original foundational topics remain in the scientific record. F11 addresses portions of them but does not make all ten resolved. For each topic, the linked [F11 crosswalk](A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md#L98) records the approved principle, missing details, consequence, and whether a dependent scope is actually blocked.

| Topic | F11 contribution | Current status / actual dependency |
|---|---|---|
| A — Semantic universe and typing | F11.1 | Partly addressed; typed universe/relation scope remains open. Pure explicit-input evaluators are not blocked. |
| B — Meaning of discovery | F11.2, F11.6 | Meaning/provenance principles approved; leakage and RQ2 evidence protocol remain open. |
| C — Realizability versus optimization | F11.3, F11.11 | Questions separated; formal criteria remain open. Pure evaluators are not blocked. |
| D — Coordinate identifiability | F11.4 | Structural-invariant priority approved; invariant set/alignment/scale protocol remains open. |
| E — Relation vocabulary | F11.5, F11.6 | Taxonomy/provenance distinctions approved; formal conditions/population remain open. |
| F — Certificate soundness | F11.8 | Endpoint/tolerance distinction approved; sound arithmetic/algorithms remain open. |
| G — Geometry capacity | F11.9, F11.10 | Theorem/benchmark summaries approved; hypotheses, construction and reachability detail remain open. |
| H — Scale/margin identifiability | F11.4 where applicable | Qualified structural comparison approved; scale protocol remains open. |
| I — Generalization/statistics | F11.7 where applicable | Generalization categories separated; populations/statistical protocols remain open. |
| J — Visual composition and ontology claims | F11.2, F11.7 where applicable | Emergence/generalization principles approved; ontology-role interpretation and claim evidence remain open. |

No row above is a new approval or an assertion that the foundational question is fully resolved.

## Source and hold boundaries

- The [original A1 ledger](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md) is canonical for F0–F10; the separate [F11 addendum](A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md) is the accepted/integrated F11 record; the separate [F12 addendum](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md) records the F12-only handoff and was independently accepted under #13 and integrated at `5686d60bdb3e309924d37eb600aa75fb93855340`. The [completeness audit](A1_DECISION_COMPLETENESS_AUDIT.md) audits only the immutable original source.
- The expanded v1.1 specification remains a historical review draft and cannot settle F11 omissions or override A1.
- The accepted T0–T5 teardown remains the classifier-free execution baseline. Phase 4, detector development, scientific training, and R2 remain on hold/unimplemented.
- Issues [#2](https://github.com/midhunpradeep/town_model/issues/2) and [#3](https://github.com/midhunpradeep/town_model/issues/3) retain their separate planning/author-decision identities. Issue [#10](https://github.com/midhunpradeep/town_model/issues/10) is accepted R1; [#12](https://github.com/midhunpradeep/town_model/issues/12) is accepted F11; [#13](https://github.com/midhunpradeep/town_model/issues/13) tracks F12 and its independent review. Issue [#11](https://github.com/midhunpradeep/town_model/issues/11) is narrowed to reconciling the remaining current-facing F12 lifecycle status.
