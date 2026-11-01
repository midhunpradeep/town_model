# A1 Implementation Readiness and Scientific Dependency Frontier

**Role:** Derived planning frontier, not a scientific specification or implementation authorization. Scientific authority is split by version: the immutable [original A1 ledger](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md) is canonical for F0–F10; the separate [F11 addendum](A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md) records twelve post-R1 approvals and becomes repository authority for F11 only after independent review and integration. Candidate scopes below are not `workflow:ready` tickets.

## Authority, provenance and status dimensions

The original A1 ledger contains 70 active approvals and remains byte-identical (SHA-256 `f552b60f10f96d9e76585331b99e8543d8b7a81535c261386a93aa697b0489b9`, 34,910 bytes). The author-reported mathematics-session handoff dated 2026-11-01 supplies twelve additional approved summaries, F11.1–F11.12. Together the current supplied-source scientific frontier is **70 + 12 = 82 active approvals**. This does not mean R1 independently reviewed all 82: R1 issue [#10](https://github.com/midhunpradeep/town_model/issues/10) was accepted and integrated at `100d04d4ecae4f58f525bbdac76b87bf28026d5c`, but its review did not cover F11. The F11 repository addendum is submitted separately under issue [#12](https://github.com/midhunpradeep/town_model/issues/12); its fixed repository revision awaits independent review/integration. The original [completeness audit](A1_DECISION_COMPLETENESS_AUDIT.md) still audits only the original 70-decision source and is not rewritten.

| Dimension | Current state |
|---|---|
| **Scientific** | Author-reported approval: original 70 A1 decisions plus twelve F11 summaries. F3.6 remains superseded by F3.6-R; no original decision is superseded by F11. Audit A–J and other omitted formal/experimental details remain open where stated. |
| **Implementation** | No reconstructed classifier, geometry, objective, optimizer, detector, or F11 evaluator has been implemented. Candidate mathematical slices below are planning only. The accepted T0–T5 classifier-free baseline remains operational. |
| **Independent review** | R1 (#10) is accepted for its own bounded documentation scope; it did not review F11. This F11 repository addendum is not self-accepted and requires its own independent review. Scientific approval, implementation completion and independent acceptance are distinct. |

The handoff source contains concise approval summaries, not the original mathematics-session transcript or formal derivations. Do not fill missing definitions, theorem hypotheses, algorithms, or experimental protocols by inference.

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

### P0-A — Raw-to-physical transforms and numerical feasibility

Still open: exact bounded/unbounded raw-to-physical parameterization/transforms and initialization; finite-precision reserves and validated operating range; geometry feasibility and `ε`/`δ` constraints/defaults; matched physical initialization. These block trainable parameterization/initialization and its numerical validation. They do not block physical-corner Boolean membership or other pure evaluators with explicit valid-input contracts.

### P0-B — Transactional invalid optimizer-update rejection

Still open: atomic rejection, rollback and recovery for invalid/nonfinite proposed updates, including model parameters, optimizer state (including AdamW moments), scheduler state and the recorded run/checkpoint disposition. F7.8 approves rejecting invalid updates but does not settle the transaction. This blocks corresponding training/update modules, not pure geometry or objective evaluators that perform no optimizer update.

### P0-C — Training stages and optimizer policy

Still open: training-stage transitions, optimizer-state policy at transitions, numerical optimizer defaults, and gradient-failure handling. This blocks the corresponding trainable-stage orchestration/configuration; it does not block unrelated deterministic mathematical evaluators.

### Classifier-facing integration edges — separate unresolved boundary

Do not merge these integration questions into P0-A/B/C or silently derive answers from retired classifiers:

- ownership and ordering of class outputs/class IDs;
- point-domain behavior, including `Δ=ℝ^d` integration;
- empty-input behavior;
- zero-window behavior at classifier/integration boundaries (while F3.11's mathematical `W=0` and single-box equivalence remain approved).

A pure evaluator or a point-only affine slice may be planned only with these edges explicitly scoped out or supplied by an independently approved caller contract. This frontier does not implement or ready that integration work.

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

- The [original A1 ledger](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md) is canonical for F0–F10; the separate [F11 addendum](A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md) is the F11-only record, subject to its independent-review/integration condition. The [completeness audit](A1_DECISION_COMPLETENESS_AUDIT.md) audits only the immutable original source.
- The expanded v1.1 specification remains a historical review draft and cannot settle F11 omissions or override A1.
- The accepted T0–T5 teardown remains the classifier-free execution baseline. Phase 4, detector development, scientific training, and R2 remain on hold/unimplemented.
- Issues [#2](https://github.com/midhunpradeep/town_model/issues/2) and [#3](https://github.com/midhunpradeep/town_model/issues/3) retain their separate planning/author-decision identities. Issue [#10](https://github.com/midhunpradeep/town_model/issues/10) is accepted R1. Issue [#11](https://github.com/midhunpradeep/town_model/issues/11) remains a separate post-R1 lifecycle-status task; issue [#12](https://github.com/midhunpradeep/town_model/issues/12) tracks this F11 reconciliation.
