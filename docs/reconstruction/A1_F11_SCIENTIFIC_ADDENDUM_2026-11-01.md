# A1 F11 Scientific Addendum — Twelve Post-R1 Approvals

**Thesis:** Representing Ontologies Using Town Embeddings
**Handoff date:** 2026-11-01
**Scope:** F11.1–F11.12 only; separately versioned addition to the original A1 source.

## Provenance and status

The thesis author supplied the twelve approval summaries below in a post-R1 mathematics-session handoff dated **2026-11-01** and explicitly approved this addendum-reconciliation task on that date. The summaries are preserved verbatim as supplied approval-summary text. This record reflects the author-reported approval status; it is not a transcript, a verbatim quotation from the mathematics session, or a formal derivation.

The original mathematics-session transcript, detailed proofs, formulas, algorithms, experimental protocols, and a source URL were not supplied for this task. None is claimed or reconstructed here. Where an approval summary names a theorem, taxonomy, or diagnostic without all formal conditions, the approved statement is retained and the omitted technical detail is identified below as unresolved.

| Status dimension | State recorded by this addendum |
|---|---|
| **Scientific** | The supplied handoff reports F11.1–F11.12 as author-approved. The exact approved summaries are those in §2; no further definition is implied. |
| **Implementation** | No model, evaluator, classifier, objective, experiment, or training code is implemented by this addendum. Candidate scopes in the readiness report remain planning candidates. |
| **Independent review** | The R1 review under issue [#10](https://github.com/midhunpradeep/town_model/issues/10) accepted the R1 documentation revision, but did not independently review these twelve F11 approvals or this addendum. This repository addendum is submitted for its own independent review; it is not accepted by its author or implementer. |

## 1. Relationship to the original A1 source

The [canonical A1 Scientific Decision Ledger](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md) remains unchanged and canonical for F0–F10. Its SHA-256 is `f552b60f10f96d9e76585331b99e8543d8b7a81535c261386a93aa697b0489b9` and its size is 34,910 bytes. The [completeness audit](A1_DECISION_COMPLETENESS_AUDIT.md) continues to index the 70 active approvals in that original source, including F3.6-R superseding F3.6.

This is a separately versioned F11 addendum, not a replacement or rewritten 82-decision ledger. **No original A1 decision is superseded by this addendum:** the supplied handoff names no supersession. The original F3.6 → F3.6-R supersession remains unchanged.

At the supplied-source level, the combined scientific frontier is **70 original active approvals + 12 new F11 approvals = 82 active approvals**. This count does not mean R1 independently reviewed all 82. The F11 summaries are author-reported approved source; the repository addendum becomes the separately versioned repository authority for F11 only after this addendum implementation is independently reviewed and integrated. Until then, this submitted repository record remains pending that review/integration condition.

## 2. Exact supplied F11 approval summaries

The following twelve statements are reproduced exactly from the supplied approval summaries. Their concise form is intentional: do not fill gaps with inferred definitions or implementation mechanics.

### F11.1 — Global predicates and supervision

Global class predicates in one latent domain; evidence-supported known/unknown masks; no automatic cross-type negatives.

### F11.2 — Emergent structure

Emergent global geometric structure means structure arising from instance-membership supervision without explicit ontology-relation penalties.

### F11.3 — Distinct learning questions

Separate representational realizability, numerical feasibility, optimization success and generalization.

### F11.4 — Structural invariants

Structural invariants are primary; raw coordinates require qualified comparisons.

### F11.5 — Set-relation taxonomy

Exact set-relation taxonomy including proper inclusion, equality, disjointness, proper overlap, emptiness and boundary-only contact.

### F11.6 — Evidence provenance

Distinguish OWL entailment, verified generator interpretations and finite-sample evidence.

### F11.7 — Generalization

Distinguish held-out image, compositional and structural generalization.

### F11.8 — Exact certificates

Exact geometric certificates refer to sets defined by finite stored physical endpoints; tolerance estimates are separate.

### F11.9 — Helly property

Single boxes have the 2-Helly property; supplementary higher-order intersection benchmark.

### F11.10 — Capacity bounds

Fixed-capacity expressivity bounds and constructive witnesses, including the 1D `H(W+1)` connected-component upper bound.

### F11.11 — Simultaneous feasibility

Separate logical, geometric, instance-level and numerical simultaneous feasibility.

### F11.12 — Affine structural diagnostics

Optional structural diagnostics for the affine halfspace baseline; box–town remains the main RQ2 comparison.

## 3. Crosswalk to prior A1 decisions

This is a navigation crosswalk to the immutable original source, not a restatement or amendment of those decisions. Links point to the original A1 decision lines; the ID references identify context and do not imply that F11 changes an earlier decision.

| F11 addition | Related original A1 decisions (source-linked) | Relationship / boundary |
|---|---|---|
| F11.1 | [F0.1](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L22), [F1.1](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L23), [F2.3](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L29) | Adds a global-predicate/typing and known/unknown-supervision principle; exact typed universe and cross-type applicability still need clarification. |
| F11.2 | [F3.1](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L38), [F3.3](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L40), [F10.1](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L113), [F10.9](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L121) | Clarifies the intended meaning of emergent global geometry while leaving relation evidence and evaluation protocol open. |
| F11.3 | [F3.3](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L40), [F5.2](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L62), [F10.3](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L115), [F10.15](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L127) | Requires distinct questions/evidence for representability, numerical feasibility, optimization, and generalization; supplies no pass criteria or test protocol. |
| F11.4 | [F2.6](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L32), [F2.7](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L33), [F5.6](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L66), [F10.4](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L116) | Makes structural invariants primary and requires qualified coordinate comparisons; it does not specify the invariant set or alignment protocol. |
| F11.5 | [F10.1](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L113), [F10.9](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L121), [F10.10](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L122), [F10.11](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L123) | Names the approved relation categories; exact formal conditions and evaluation populations remain open. |
| F11.6 | [F0.1](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L22), [F1.1](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L23), [F10.6](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L118), [F10.9](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L121), [F10.11](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L123) | Requires evidence-source distinctions; it does not define a final reference-label mapping or aggregation protocol. |
| F11.7 | [F1.1](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L23), [F10.11](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L123), [F10.12](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L124) | Separates three generalization questions; it does not specify populations, split construction, metrics, or statistical units. |
| F11.8 | [F3.2](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L39), [F3.6-R](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L43), [F7.3](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L84), [F10.6](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L118) | Narrows the certificate object to sets from finite stored physical endpoints and separates tolerance estimates; arithmetic soundness and algorithms remain open. |
| F11.9 | [F3.4](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L41), [F3.8](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L45), [F3.9](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L46), [F10.2](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L114), [F10.5](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L117) | Records the single-box 2-Helly approval and a supplementary higher-order benchmark; theorem hypotheses and benchmark protocol are not supplied. |
| F11.10 | [F3.8](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L45), [F3.9](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L46), [F10.2](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L114), [F10.5](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L117) | Adds fixed-capacity expressivity bounds/witnesses and the stated 1D upper bound; the applicable family, assumptions, and witness construction need clarification. |
| F11.11 | [F3.2](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L39), [F3.3](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L40), [F5.2](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L62), [F7.8](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L89), [F10.3](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L115) | Requires separate logical, geometric, instance-level, and numerical feasibility questions; supplies no criterion for classifying a case into each. |
| F11.12 | [F3.1](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L38), [F3.3](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L40), [F4.2](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L55), [F10.1](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L113), [F10.3](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L115) | Allows supplementary affine structural diagnostics while retaining box–town as the main RQ2 comparison; diagnostic metrics and protocol remain unspecified. |

## 4. Foundational audit A–J reconciliation

The source topics below link to the original A1 ledger's unresolved [foundational audit](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L172). The initial F11 mapping is the supplied scope. Each topic remains only partially resolved where stated; an approved principle does not settle every mathematical or experimental detail.

### A — Semantic universe and typing

Source: [A1 audit A](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L176). Initial link: F11.1.

- **Approved foundational principle:** Global class predicates share one latent domain; known/unknown masks must be evidence-supported; cross-type negatives are not automatic.
- **Still missing:** The exact entity universe, typed applicability of each class, cross-type supervision policy beyond the no-automatic-negative rule, and how types enter any relation population.
- **Implementation/evaluation consequence:** Keep point/set membership distinct from evidence about whether a class label applies. Do not synthesize cross-type negative labels.
- **Dependency:** A pure evaluator over explicitly supplied typed points and regions is not blocked. Cross-type supervision, a global typed universe, or relation-pair evaluation is blocked on the missing scope.

### B — Meaning of discovery

Source: [A1 audit B](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L178). Initial links: F11.2 and F11.6.

- **Approved foundational principle:** “Emergent global geometric structure” means structure arising from instance-membership supervision without explicit ontology-relation penalties; evidence provenance distinguishes OWL entailment, verified generator interpretations, and finite-sample evidence.
- **Still missing:** Leakage controls, the exact evidence required to demonstrate emergence, and how inherited generator labels affect the interpretation of a claimed structural pattern.
- **Implementation/evaluation consequence:** Do not add ontology-relation penalties to the primary objective or call an observed pattern independent ontology discovery based on this definition alone. Report evidence provenance separately.
- **Dependency:** Pure geometry evaluators are not blocked. Defensible RQ2 discovery claims and their reference/evaluation protocol remain blocked on these clarifications.

### C — Realizability versus optimization

Source: [A1 audit C](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L180). Initial links: F11.3 and F11.11.

- **Approved foundational principle:** Separate representational realizability, numerical feasibility, optimization success, and generalization; separately distinguish logical, geometric, instance-level, and numerical simultaneous feasibility.
- **Still missing:** Operational definitions, feasibility oracles, diagnostic criteria, and what evidence is sufficient to assign a failure to one layer rather than another.
- **Implementation/evaluation consequence:** Reports and tests must not collapse these questions into a single “works/fails” result. No new objective, solver, or optimizer behavior follows from the classification.
- **Dependency:** Physical-input evaluators and deterministic fixtures are not blocked. Trainable feasibility/optimization validation and claims that distinguish failure modes need the corresponding approved criteria and P0 decisions.

### D — Coordinate identifiability

Source: [A1 audit D](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L182). Initial link: F11.4.

- **Approved foundational principle:** Compare structural invariants primarily; compare raw coordinates only with qualification.
- **Still missing:** The exact invariant set, allowed alignment/equivalence transformations, scale policy, and which coordinate comparisons are scientifically meaningful.
- **Implementation/evaluation consequence:** Do not interpret unaligned coordinates or raw parameter values across runs as comparable structure. Preserve hard set relations separately from score/parameter scale.
- **Dependency:** A single-run membership oracle is not blocked. Cross-run coordinate, scale, or margin comparisons are blocked until a qualified comparison protocol is specified.

### E — Relation vocabulary

Source: [A1 audit E](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L184). Initial links: F11.5 and F11.6.

- **Approved foundational principle:** The exact relation taxonomy includes proper inclusion, equality, disjointness, proper overlap, emptiness, and boundary-only contact; evidence provenance distinguishes OWL entailment, verified generator interpretations, and finite-sample evidence.
- **Still missing:** Formal conditions for each relation, including proper overlap and boundary-only contact; the universe/domain for emptiness; pair selection, reference status mapping, and protocol details.
- **Implementation/evaluation consequence:** Keep these named categories distinct and preserve provenance. Do not infer that a non-entailment is false or that a sample settles a global set relation.
- **Dependency:** Local membership and score evaluators are not blocked. A final relation vocabulary implementation, reference/pair manifest, or RQ2 aggregation remains blocked on the missing formal conditions and evaluation protocol.

### F — Certificate soundness

Source: [A1 audit F](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L186). Initial link: F11.8.

- **Approved foundational principle:** Exact geometric certificates concern sets defined by finite stored physical endpoints; tolerance-based estimates are separate outputs.
- **Still missing:** The precise arithmetic interpretation of stored endpoints, a sound certificate algorithm and resource policy, and the tolerance-estimate method/labels.
- **Implementation/evaluation consequence:** Do not describe a tolerance estimate as an exact certificate or allow a failed search to imply a proof. Boundary contact must follow the still-applicable F3.6-R semantics.
- **Dependency:** Exact Boolean membership for explicitly supplied physical corners is not blocked. Exact relation-certificate implementation and certificate claims are blocked on soundness details.

### G — Geometry capacity

Source: [A1 audit G](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L188). Initial links: F11.9 and F11.10.

- **Approved foundational principle:** A single box has the 2-Helly property; a supplementary higher-order intersection benchmark is approved. Fixed-capacity expressivity bounds and constructive witnesses are approved in principle, including the 1D `H(W+1)` connected-component upper bound.
- **Still missing:** Formal hypotheses and quantifiers for each claim, the exact region family and capacity assumptions, constructive witness details, and supplementary benchmark inputs/metrics. The supplied single-box statement is not a theorem about towns or unions unless separately established.
- **Implementation/evaluation consequence:** Keep higher-order intersection and fixed-capacity studies supplementary. Do not generalize a family-level bound into a trainable-reachability or primary-comparison claim.
- **Dependency:** Evaluating a supplied region is not blocked. Theorem-backed capacity claims, constructive-witness implementations, and supplementary benchmark interpretation require the missing conditions.

### H — Scale and margin identifiability

Source: [A1 audit H](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L190). Initial link: F11.4 where applicable.

- **Approved foundational principle:** Structural invariants are primary; raw coordinates require qualified comparisons.
- **Still missing:** Which invariant metrics are scale-independent, how alignment/scale is controlled or reported, and whether supplementary scale/margin comparisons are required.
- **Implementation/evaluation consequence:** Preserve raw coordinate clearance in its approved physical units where used; do not compare unqualified raw scores across independently scaled runs.
- **Dependency:** Local raw-clearance evaluation is not blocked. Cross-run scale/margin conclusions and scale-controlled studies remain blocked on a comparison protocol.

### I — Generalization and statistics

Source: [A1 audit I](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L192). Initial link: F11.7 where applicable.

- **Approved foundational principle:** Distinguish held-out image, compositional, and structural generalization.
- **Still missing:** Populations and split construction for each category, statistical unit, metric/denominator, uncertainty treatment, and selection/holdout protocol.
- **Implementation/evaluation consequence:** Keep the three generalization claims separate; no one split or metric is presumed to establish all three.
- **Dependency:** Deterministic mathematical fixtures are not blocked. Comparison-quality generalization studies and conclusions are blocked on the missing population and statistical protocol.

### J — Visual composition and ontology claims

Source: [A1 audit J](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L194). Initial links: F11.2 and F11.7 where applicable.

- **Approved foundational principle:** Emergent structure has the F11.2 membership-supervision meaning; held-out image, compositional, and structural generalization are distinct questions.
- **Still missing:** How visual composition evidence relates to ontology role/existential semantics, what would distinguish learned structural relations from generator/recipe regularities, and which evaluation evidence supports an RQ2 interpretation.
- **Implementation/evaluation consequence:** Do not claim full OWL/ALC reasoning or ontology discovery from compositional prediction alone. Keep relation evidence and generalization categories explicit.
- **Dependency:** Component-level membership evaluators are not blocked. Pizza-level RQ2 interpretation and ontology-structure/generalization claims remain blocked on the missing evaluation scope.

**Crosswalk status:** F11 addresses portions of A–J but does not fully resolve any topic whose listed formal or experimental details remain missing. The crosswalk adds no new mathematical approval and does not supersede the original audit.

## 5. Primary scientific invariants and implementation boundaries

The five primary conditions remain exactly:

1. Affine baseline + BCE-with-logits.
2. Box + signed logistic.
3. Box + signed hinge.
4. Town + signed logistic.
5. Town + signed hinge.

Primary objectives remain unregularized. Compactness penalties, auxiliary point attraction, ontology-axiom losses, and supplementary reference losses are not implicit primary terms. F3.6-R remains authoritative: a house is a closed frame minus the union of ambient open window interiors; a town is the union of houses. Exact Boolean membership determines geometric predictions. Signed coordinate clearance is a separate numerical training measurement composed by the approved exact min/max rules, without hidden normalization. No detail is inferred from retired classifiers.

This addendum does not implement models, geometry, objectives, optimizers, classifiers, detectors, or experiments; resolve P0-A/B/C; decide final RQ1/RQ2 methodology; begin R2; change data or historical evidence; or authorize any candidate implementation. Every future task still requires a bounded contract, dependency check, and independent review.
