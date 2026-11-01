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
- The author-approved mathematical handoff supplied for R0 reports **50 active A1 decisions**. The full decision ledger and exact approval-source links have not yet been durably reconciled in the repository. **R1 owns that formal reconciliation.** R0 records only the count and limited handoff context; agents must not infer the full ledger from historical code, the expanded draft, or a summary comment. Until R1 makes exact decisions traceable to author-approved sources, unresolved details remain author-controlled and dependent implementation work must not be marked READY.
- Phase 1, Phase 2, and bounded pre-detector Phase 3 remain accepted for their historical scopes. The later T0–T5 teardown is the accepted operational boundary: historical classifier implementations and model-dependent execution were removed or made explicitly unavailable. Neither historical acceptance nor teardown authorizes reconstruction, detector work, or scientific training. Phase 4 remains on hold.
- The `Classifier Reconstruction — Author-Led Mathematical Design` issue #2 remains a planning map. Issue #3 and its provisional status comment do not authorize implementation or finalize the decision record. R1 must reconcile the tracker with the approved A1 sources without converting agent summary into new author approval.

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
