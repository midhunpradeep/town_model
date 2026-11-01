# A1 F12 Scientific Addendum — Twelve Trainable-Geometry and Optimizer Approvals

- **Thesis:** Representing Ontologies Using Town Embeddings
- **Handoff / approval date:** 2026-11-01
- **Scope:** F12.1–F12.12 only; separately versioned addition to the original A1 and F11 sources.

## Provenance and status

The thesis author approved the F12 reconciliation contract on **2026-11-01**. This addendum preserves the F12 mathematics-session handoff supplied for that task. The available provenance record is that author-provided handoff; no separate transcript, source URL, or individually numbered F12.1–F12.8 statements were supplied or inspected. The addendum does not claim otherwise.

The handoff associates **F12.1–F12.8 collectively** with the trainable-geometry block in §2. It does not state which individual F12.1–F12.8 identifier corresponds to which distinct formula or requirement. The entire approved block is preserved together, and no per-ID mapping is inferred. F12.9–F12.12 are individually identifiable and recorded separately in §§3–6. If the original individually numbered mathematics-session decisions become available, their traceability may be added only through a separately reviewed revision.

| Status dimension | State recorded by this addendum |
|---|---|
| **Scientific** | The supplied handoff reports twelve author-approved F12 decisions dated 2026-11-01. F12.1–F12.8 are a grouped approval block with the provenance limitation above; F12.9–F12.12 are separately identifiable. This records author-reported approval, not proof review or scientific evaluation. |
| **Implementation** | No geometry transform/inverse, optimizer transaction, classifier, training stage, detector, loss, or experiment is implemented by this addendum. The readiness document identifies later engineering obligations. |
| **Independent review** | R1/#10 and F11/#12 were independently accepted for their bounded scopes; neither reviewed F12. This F12 repository addendum is submitted under [issue #13](https://github.com/midhunpradeep/town_model/issues/13) and remains pending its own independent review and integration. It is not self-accepted. |

## 1. Relationship to the original A1 and F11 authorities

The [original A1 Scientific Decision Ledger](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md) remains immutable and authoritative for F0–F10: 70 active approvals, with F3.6-R superseding F3.6. Its SHA-256 is `f552b60f10f96d9e76585331b99e8543d8b7a81535c261386a93aa697b0489b9` (34,910 bytes). The [completeness audit](A1_DECISION_COMPLETENESS_AUDIT.md) continues to audit only that original source.

The separate [F11 Scientific Addendum](A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md) records twelve author-reported approvals and is independently accepted/integrated under [issue #12](https://github.com/midhunpradeep/town_model/issues/12) at `a050234ae490d650e28391f180670d1c38183192`. Its SHA-256 is `3a417f74dbfdbee0771dbdf353c9348801a1c35467c64cdd6b9975caabf91c86` (21,547 bytes). Its independent review verifies F11 reconciliation, not F12.

The supplied-source frontier is **70 original active approvals + 12 F11 approvals + 12 author-reported F12 approvals = 94 active approvals**. This is a source-level count, not a claim that all 94 were reviewed together, implemented, or scientifically evaluated. F12 is a separate addendum, not a rewritten 94-decision ledger. It supersedes no original A1 or F11 decision; the F3.6 → F3.6-R supersession remains unchanged. This F12 repository addendum becomes repository authority for F12 only after its own independent review and integration.

### Relationship crosswalk (context, not supersession)

| F12 record | Related accepted A1/F11 context | Boundary |
|---|---|---|
| F12.1–F12.8 grouped geometry block | Original A1 F2.7, F3.6-R, F3.7, F3.10–F3.11, F7.1–F7.8; F11.3, F11.4, F11.8, F11.11 | Provides trainable physical parameterizations, initialization and numerical policy; does not alter F3.6-R set semantics or establish a validated operating envelope. The individual grouped F12 IDs are not mapped to separate statements. |
| F12.9–F12.10 | Original A1 F7.8, F9.2–F9.5; F11.3, F11.11 | Specifies effective-update atomicity and terminal numerical-failure disposition; exact transaction/recovery mechanism and measured verification remain engineering work. |
| F12.11 | Original A1 F9.1–F9.4 | Distinguishes fresh cross-stage optimizer/scheduler state from complete same-stage resume; it does not define all stage-specific monitoring or detector policies. |
| F12.12 | Original A1 F8.5, F9.2–F9.5 | Supplies configurable initial optimizer defaults, not optimal hyperparameters, stage-specific trainability, or detector authorization. |

## 2. F12.1–F12.8 — grouped approval: trainable geometry

The following requirements are preserved as one approved geometry block collectively associated with F12.1–F12.8. No sentence or formula below is assigned to a particular one of those eight IDs.

### 2.1 Bounded latent domain

The bounded latent domain is `[-1,1]^d`. For physical center `c`, positive half-width `h`, and endpoints

`l = c - h`

`r = c + h`,

the approved width-first transform is

`h = [w_min + (2 - w_min) sigmoid(a)] / 2`

`c = (1 - h) tanh(b)`.

`epsilon` is the ordinary minimum full width, so ordinary boxes use `w_min = epsilon`. Hosting frames with windows use the approved reserve

`w_min = epsilon + 2delta`.

Preserve the distinction between ordinary boxes and hosting frames. Do not replace this mapping with a historical lower-corner transform, post-transform clamping, or another bounded parameterization.

### 2.2 Unbounded latent domain

For `R^d`, use

`c = b`

`h = w_min/2 + softplus(a)`.

Do not replace the mathematically unbounded model with an artificial finite clipping domain.

### 2.3 Contained windows

Use the parent frame's physical center and half-width. Define

`a_available = h_F - delta - epsilon/2`,

then use

`h_W = epsilon/2 + a_available sigmoid(u)`

`c_W = c_F + (h_F - delta - h_W) tanh(v)`.

Preserve the intended physical containment and reserve constraints. Do not invent an alternative transform or silently reinterpret the available-width quantity.

### 2.4 Independent windows

Independent windows use the ordinary domain-specific box transform and may extend beyond the parent frame. Contained and independent modes use the same F3.6-R Boolean set semantics: a house is a closed frame minus the union of ambient open window interiors, and a town is the union of houses. Independent windows outside a frame do not remove points outside that frame.

### 2.5 Initialization

The approved initialization sequence is:

1. Sample feasible physical geometry using a seeded, label-agnostic process.
2. Invert the applicable transforms.
3. Forward-decode the initialized raw parameters.
4. Validate the actual resulting physical geometry.
5. Compare physically matched starts, not raw parameter values.
6. Record physical corners, matching error, occupancy, and dimension-aware window-exclusion diagnostics.

Exact inverse algorithms, numerical budgets, and tested envelopes remain engineering obligations; this addendum does not invent them as approved mathematical facts.

### 2.6 Numerical policy

Preserve all of the following:

- Configurable initial `epsilon = 1e-4` and `delta = 1e-4`.
- `float32` geometry transforms, losses, and evaluation.
- Independent validation of actual stored corners, widths, domain bounds, and contained-window clearances.
- Hard constraints without approximate-tolerance relaxation.
- Declared and tested numerical operating envelopes.
- Separation of invalid geometry from saturated or poorly conditioned gradients.
- Rejection of invalid proposed states without automatic clamping or physical repair.
- Inverse/forward validation under a predeclared ULP-based matching budget constrained by available reserves.
- Recording of actual initial corners, matching errors, and occupancy.

The initial epsilon/delta values are configurable starting constants, not universally validated values across dimensions or capacities. Exact inverse algorithms, ULP budgets, tested ranges, initialization robustness, gradients, and numerical extremes require measured engineering verification. An untested numerical envelope must not be presented as accepted.

## 3. F12.9 — Atomic effective optimizer update

An accepted effective optimizer update is all-or-nothing across:

- Model parameters.
- Optimizer moments and optimizer state.
- Scheduler advancement.
- Relevant training counters and progress state.

Permitted engineering approaches include staged commits, complete rollback, and fail-closed recovery from the last complete valid checkpoint. No partially updated state may be accepted as a completed effective step. The actual transaction mechanism and fault-injection verification belong to later implementation tasks, not this documentation addendum.

## 4. F12.10 — Numerical failure disposition

Upon confirmed numerical invalidity:

- Reject the effective update.
- Terminate the current run.
- Record the failure reason, affected parameters, and last complete valid checkpoint.
- Do not count the failed update as completed.
- Do not silently advance the scheduler.
- Do not skip the batch and continue.
- Do not automatically reduce learning rates.
- Do not repair or clamp invalid geometry.
- Do not reset optimizer state and continue.
- Resume only through explicit action from a validated complete checkpoint.

Keep this distinct from ordinary gradient saturation or poor conditioning, which alone does not establish invalid physical geometry. Later implementation must define and fault-injection-test its mechanism without changing this disposition.

## 5. F12.11 — Training-stage transitions

When entering a new scientific training stage:

- Transfer selected model parameters and required module state.
- Construct a fresh optimizer.
- Construct a fresh scheduler.
- Start with fresh AdamW moments.
- Use stage-specific monitoring.

For resumption within the same stage:

- Restore complete saved model state.
- Restore optimizer and scheduler state.
- Restore RNG state.
- Restore relevant training progress and counters.
- Validate checkpoint completeness before resumption.

A new stage and same-stage checkpoint resumption must not be conflated. Do not resurrect historical optimizer-transition assumptions. Specific final stage monitors and detector policies remain separately scoped rather than being invented here.

## 6. F12.12 — Initial numerical defaults

The following configurable starting learning rates are approved:

| Parameter group | Initial learning rate |
|---|---:|
| Pretrained backbone/FPN | `1e-5` |
| Detector heads | `1e-4` |
| Point projection | `1e-3` |
| Composition/fusion | `1e-3` |
| Affine classifier | `1e-3` |
| Raw box/town geometry | `1e-3` |

Other configurable starting defaults:

| Setting | Initial value / policy |
|---|---|
| AdamW betas | `(0.9, 0.999)` |
| AdamW epsilon | `1e-8` |
| Neural weight decay | `0.01` |
| Neural weight-decay exclusions | Bias and normalization parameters |
| Raw-geometry weight decay | `0` |
| Initial learning-rate schedule | Constant |
| Global gradient-norm clipping | `1.0` |
| Initial numerical dtype | `float32` |
| Gradient accumulation | `1` |

These are configurable engineering starting points, not scientifically established optimal hyperparameters. Parameter-group policy is distinct from actual stage-specific trainability. This approval does not authorize detector development.

## 7. Readiness reconciliation and remaining obligations

The derived [implementation-readiness frontier](A1_IMPLEMENTATION_READINESS.md) reclassifies P0-A/B/C as scientifically specified, with engineering implementation and measured validation still required. F12 does not mean geometry, optimizer lifecycle, or training has been implemented or tested.

The following remain future engineering/test obligations:

1. Demonstrate valid physical geometry across declared, measured operating ranges.
2. Verify bounded/unbounded and contained/independent-window invariants.
3. Verify inverse/forward correctness and physically matched initialization.
4. Demonstrate finite, corrective gradients through point and geometry parameters.
5. Keep exact hard membership distinct from signed-clearance scores.
6. Verify atomic invalid-update handling across optimizer moments and state, scheduler, counters/progress, and complete-checkpoint recovery; include incomplete/corrupted-checkpoint recovery and adversarial fault injection.
7. Verify fresh optimizer/scheduler/moments on stage transition versus exact complete same-stage continuation, including RNG and progress.
8. Demonstrate controlled-fixture loss decrease in the five primary classifier conditions.

Separate the evidence categories of **mathematical correctness**, **numerical validity**, **optimization success**, and **scientific generalization**. The above are later implementation/test obligations, not work performed by F12 reconciliation.

### Remaining separate scientific and interface boundaries

F12 does not resolve the classifier-interface boundaries for flat point input, class-order ownership, typed outputs, `DomainThing`, empty batches, bounded/unbounded point-domain behavior, or zero-window integration. It does not supply final stage-specific scientific monitors, detector policies, detector/ROI policy, statistical methodology, comparison-quality experiments, or final supervisor-dependent RQ2 aggregation. Those remain separately scoped for the stages and claims they affect. No answer is inferred here.

## 8. Primary scientific comparison and optional-work boundaries

Retain exactly these five primary conditions, all with unregularized primary objectives:

1. Affine baseline + BCE-with-logits.
2. Box + signed logistic.
3. Box + signed hinge.
4. Town + signed logistic.
5. Town + signed hinge.

Do not add compactness penalties, auxiliary point attraction, ontology-axiom losses, or supplementary reference-loss penalties to these primary objectives. F3.6-R remains authoritative: a house is a closed frame minus the union of ambient open window interiors, and a town is the union of houses. Hard Boolean membership remains distinct from signed coordinate clearance and its exact min/max numerical compositions.

The following must not become blockers for initial pure classifier reconstruction: supplementary reference-loss conformance; compactness experiments; auxiliary point organization; fixed-point and higher-order ontology benchmarks; broad capacity/topology theorems; final supervisor-dependent RQ2 aggregation; and optional affine structural diagnostics. Detector/ROI policy, statistical methodology, and final comparison-quality evaluation remain separate dependencies for stages and claims they affect. This addendum does not authorize R2, detector development, or training.

## 9. Review and authority boundary

This addendum records author-reported F12 source material and provides a versioned repository document for review. Scientific approval, repository documentation implementation, independent review, code implementation, and scientific evaluation are distinct states. The [reconciliation report](A1_F12_RECONCILIATION_REPORT_2026-11-01.md) records source identities, the grouped-ID limitation, readiness changes, reproducibility evidence, and acceptance-criteria mapping. Only an independent reviewer may accept this repository revision; until review and integration, do not treat this addendum as independently accepted repository authority for F12.
