# ADR-0001: Immutable bound wrapper for classifier class-column identity

- **Status:** PROPOSED — author confirmation and independent review pending; not accepted.
- **Date:** 2026-11-03
- **Related decision record:** [#15 — R2 interface selections](https://github.com/midhunpradeep/town_model/issues/15)
- **Related task/draft:** [#14](https://github.com/midhunpradeep/town_model/issues/14), [R2 interface draft at `fb432286113f9d11d433ab9e3499e97c7275dba2`](https://github.com/midhunpradeep/town_model/blob/fb432286113f9d11d433ab9e3499e97c7275dba2/docs/reconstruction/R2_CLASSIFIER_INTERFACE_CONTRACT_DRAFT.md)
- **Reconciliation:** [R2 author-selection reconciliation](../reconstruction/R2_AUTHOR_SELECTION_RECONCILIATION_2026-11-03.md#d11--immutable-class-binding)

> This is a proposed transcription of the author's reported 2026-11-03 session selection. The underlying session transcript was not supplied. This ADR is neither durable author confirmation nor implementation authorization.

## Context

A1 F4.1/F4.4 establishes a flat point classifier interface and ordered trainable outputs. The retained dataset supplies a canonical trainable class order and vocabulary/manifest identity. A mathematical kernel can remain class-ID agnostic, but a persisted checkpoint needs an enforceable identity for the meaning of its parameter rows. The first independent R2 review identified same-width class-order mismatch as a blocker; the corrected draft proposed persistent exact binding but left ownership open.

## Proposed decision

An immutable bound wrapper owns the exact ordered trainable-class ID tuple and canonical vocabulary/manifest identity. Persist this binding with the bound model, run, and checkpoints. Validate exact identity before training, inference, and state restoration. Reject same-width permutations, incompatible manifests, and all mismatches without silently relabeling parameters or mutating restored state. A class-count-only pure mathematical kernel is permitted only behind this enforcing wrapper.

The wrapper choice is an interface/architecture policy, not an additional F0–F12 scientific approval. The source authority remains the [original A1 ledger](../reconstruction/A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L54).

## Alternatives considered

- Let classifier state own the binding — not selected in the reported session.
- Attach the current class IDs only to outputs or validate only class count — rejected because a checkpoint trained under a different same-width ordering would be silently relabeled.
- Permute/relabel trained rows during restore — rejected; mismatch must fail before state mutation.

## Consequences

- The pure mathematical kernel remains independent of dataset/ontology objects.
- A bound model/run/checkpoint must carry durable ordered IDs and canonical manifest identity.
- Checkpoint restore, training, and inference acquire exact preflight checks.
- The serialization schema/version and user-facing mismatch error remain engineering details, but cannot weaken exact equality.
- No implementation is authorized by this proposed ADR.

## Confirmation and verification gate

**Author confirmation:** PENDING in #15. **Independent review:** PENDING for this ADR/reconciliation. **Implementation authorization:** NONE.

A later implementation contract must falsifiably test valid save/restore, same-width permutation rejection, manifest mismatch rejection, no parameter/optimizer-state mutation on rejection, and absence of silent row relabeling. Do not mark a dependent implementation issue READY before the author confirms the decision and a coordinator verifies its exact scope and blockers.
