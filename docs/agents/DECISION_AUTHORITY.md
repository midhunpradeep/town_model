# Decision Authority

## Author-owned decisions

The author is the sole approval authority for:

- Mathematical definitions and classifier/geometric semantics.
- Learning objectives and loss functions.
- Significant architecture, model parameterization, and initialization choices.
- Scientific evaluation methods, experimental comparisons, and interpretation policies.
- Changes to approved scientific requirements.

An agent may inspect, explain, compare, and recommend alternatives. A recommendation is not approval. Until the author explicitly approves a choice, agents must not encode it in production implementation, mark dependent work `workflow:ready`, or present it as an accepted requirement.

Agents may decide internal code organization within approved boundaries, small implementation details, test organization, diagnostic scripts, and routine maintenance that preserves accepted contracts. If one of these choices would change scientific meaning or an approved contract, it is no longer an autonomous implementation detail.

## Current source-of-truth boundaries

- [`CURRENT_PROJECT_STATUS.md`](../CURRENT_PROJECT_STATUS.md) states the current classifier-free operational baseline; [`ACCEPTANCE_INDEX.md`](../ACCEPTANCE_INDEX.md) and linked independent reviews distinguish historical Phase 1–3 implementation acceptance from accepted T0–T5 teardown.
- [`CLASSIFIER_IMPLEMENTATION_INVENTORY.md`](../CLASSIFIER_IMPLEMENTATION_INVENTORY.md) is descriptive historical/reference evidence. It is not a specification for reconstructed models and does not imply its inventoried implementations remain active.
- [`pizzaiolo_thesis_implementation_spec_v1_1_expanded.md`](../../pizzaiolo_thesis_implementation_spec_v1_1_expanded.md) is a historical expanded review draft. It is not the current reconstruction authority or implementation authorization; its former statements cannot supersede explicit later author decisions.
- Scientific authority is versioned by family: the [original canonical A1 Scientific Decision Ledger](../reconstruction/A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md) remains canonical for F0–F10 (70 active decisions; F3.6-R supersedes F3.6), while the separate [F11 Scientific Addendum](../reconstruction/A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md) records twelve author-reported post-R1 approvals and was independently accepted/integrated under #12 at `a050234ae490d650e28391f180670d1c38183192`. The separate [F12 Scientific Addendum](../reconstruction/A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md) records twelve further author-reported approvals; F12.1–F12.8 are grouped without per-ID formula traceability. F12 documentation was independently accepted under #13 and integrated at `5686d60bdb3e309924d37eb600aa75fb93855340`. The three separately versioned sources establish 94 author-reported approvals; acceptance of source documentation does not validate mathematical proofs, executable implementation, or demonstrated learning. The original [completeness audit](../reconstruction/A1_DECISION_COMPLETENESS_AUDIT.md) still verifies only the original 70-decision source. R1 was independently accepted and integrated at `100d04d4ecae4f58f525bbdac76b87bf28026d5c` under issue #10, but that review did not cover F11 or F12. The [readiness frontier](../reconstruction/A1_IMPLEMENTATION_READINESS.md) is derived planning. Do not infer additional approvals from mathematical consequences, historical code, the expanded draft, or implementation recommendations. Open gates and audit questions remain author-controlled; dependent work cannot be marked READY until its specific missing decisions are approved.
- Phase 1, Phase 2, and bounded pre-detector Phase 3 remain accepted for their historical scopes. The later T0–T5 teardown is the accepted operational boundary: historical classifier implementations and model-dependent execution were removed or made explicitly unavailable. The definitive Phase 3 review historically authorized Implementation Phase 4; current author direction keeps it on hold. That historical authorization does not make detector implementation/training or later scientific training ready or independently accepted. A1 approval likewise does not authorize arbitrary implementation.
- The `Classifier Reconstruction — Author-Led Mathematical Design` issue #2 remains an open planning map for the source-linked A1/F11/F12 frontier; it is not an implementation ticket. Issue #3 remains an open author-decision ticket. R1's source-linked proposal is not author confirmation; only explicit author confirmation can make it a final issue-level resolution or close #3. Issues #10 (accepted R1), #11 (narrowed current-facing F12 lifecycle-status reconciliation), #12 (accepted F11 addendum), and #13 (accepted F12 addendum documentation at `5686d60bdb3e309924d37eb600aa75fb93855340`) retain distinct purposes. Neither an agent summary nor scientific approval creates implementation authorization.

If sources appear to conflict, stop and record the precise conflict for the author. Do not select whichever source is easiest to implement.

## What counts as approval

Approval must be explicit, attributable to the author, and scoped to an exact decision. Record it in a GitHub issue comment or another durable author-approved source with:

1. The exact approved definition/requirement and its boundary conditions.
2. Which alternatives were rejected, deferred, or left open.
3. The intended model family, phase, experiment, or other scope.
4. Links to evidence and any superseded decision.
5. The approved specification or ADR location when the decision warrants one.

Silence, lack of objection, issue reactions, a prior agent's recommendation, a historical commit, an implementation report, or a closed review is not renewed author approval. An agent cannot approve its own proposed scientific decision. If the author approves verbally or in a session, capture the decision in the tracker and request confirmation before treating the written record as approved.

## Unresolved-decision protocol

When work depends on a high-level decision:

1. Create a `[AUTHOR DECISION] ...` issue from the decision template. Use a `wayfinder:<type>` label and link it to the classifier-reconstruction map as a child issue.
2. State the question, context, evidence, alternatives and consequences without smuggling in a preferred answer. Mark author approval as **pending**.
3. Add a native GitHub blocking relationship from every dependent implementation issue to the decision issue. Set the implementation issue to `workflow:blocked` and record the precise unblock condition.
4. Keep dependent work out of `workflow:ready`. Do not implement a proposed alternative as a prototype unless the author explicitly authorizes that prototype's scope.
5. Wait for explicit author approval. Record the approved answer and source link in the decision issue; update the approved specification and create/update an ADR when appropriate. Only then may a coordinator revalidate and move dependent work toward `workflow:ready`.

## Durable decisions and ADRs

Small, bounded decisions may be fully recorded in their issue with the author's explicit approval comment. Significant architectural or mathematical choices should also receive a numbered ADR under `docs/adr/`, linked to the decision issue and approval source. ADRs record the approved decision and consequences; they do not independently grant approval. Do not rewrite historical acceptance reports or infer renewed approval from an old ADR, report, or implementation.
