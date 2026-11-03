# ADR-0002: Domain-specific terminal activations for image point heads

- **Status:** PROPOSED — author confirmation and independent review pending; not accepted.
- **Date:** 2026-11-03
- **Related decision record:** [#15 — R2 interface selections](https://github.com/midhunpradeep/town_model/issues/15)
- **Related task/draft:** [#14](https://github.com/midhunpradeep/town_model/issues/14), [R2 interface draft at `fb432286113f9d11d433ab9e3499e97c7275dba2`](https://github.com/midhunpradeep/town_model/blob/fb432286113f9d11d433ab9e3499e97c7275dba2/docs/reconstruction/R2_CLASSIFIER_INTERFACE_CONTRACT_DRAFT.md)
- **Reconciliation:** [R2 author-selection reconciliation](../reconstruction/R2_AUTHOR_SELECTION_RECONCILIATION_2026-11-03.md#d3--identity-terminal-heads-for-unbounded-image-points)

> This is a proposed transcription of the author's reported 2026-11-03 session selection. The underlying session transcript was not supplied. It specifies a future implementation contract only; it is neither durable author confirmation nor authorization to change the point architecture.

## Context

A1 F2.7 approves bounded `[-1,1]^d` and unbounded `R^d` as distinct scientific conditions. The retained image point-producing paths end in `tanh`, so they emit bounded points. F12's grouped geometry approvals specify trainable geometry transforms, not the image encoder's terminal activations. A bounded checkpoint may have tensor shapes compatible with an unbounded model while carrying different point semantics.

## Proposed decision

Preserve existing bounded `tanh` point-producing pathways. For unbounded-domain runs, use identity terminal activations in the relevant component, global, composition, and hybrid point-producing heads. Do not add clipping, rescaling, or a finite surrogate for `R^d`. Record domain and activation identity in run configuration and checkpoints. Do not silently reinterpret bounded checkpoints as unbounded checkpoints. Preserve the shared point architecture, identities, masks, and composition machinery except for the separately authorized activation behavior.

This is a reported session selection for a future contract, not an F0–F12 mathematical amendment and not permission to implement the change.

## Alternatives considered

- Keep the bounded `tanh` heads and defer unbounded image integration while allowing pure supplied-point evaluators — available in the R2 draft but not selected in the reported session.
- Use a finite rescale, clipping range, or other bounded surrogate for `R^d` — rejected.
- Change other point architecture, identity, mask, or composition behavior at the same time — not selected.

## Consequences

- Bounded point outputs/checkpoint behavior must remain unchanged.
- Unbounded point heads can emit values outside `[-1,1]^d` without a terminal finite bound.
- Domain/activation becomes part of run/checkpoint identity; equal parameter shapes do not imply semantic checkpoint compatibility.
- Future implementation must determine the relevant active heads for each mode and test them individually.
- No implementation is authorized by this proposed ADR.

## Confirmation and verification gate

**Author confirmation:** PENDING in #15. **Independent review:** PENDING for this ADR/reconciliation. **Implementation authorization:** NONE.

A separate implementation task, after durable confirmation, must test bounded compatibility, identity activations/no hidden clip for all selected unbounded paths, gradient/identity/mask/composition preservation, explicit run/checkpoint identity, and mutation-free rejection of bounded/unbounded checkpoint mismatches. Do not mark that task READY from this ADR alone.
