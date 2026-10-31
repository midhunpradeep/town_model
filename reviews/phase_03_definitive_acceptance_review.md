# Definitive Independent Phase 3 Acceptance Review

**Project:** *Representing Ontologies Using Town Embeddings*
**Branch:** `remediation/phase-3-last-two-findings`
**Scope:** Targeted closure of D5a and D5b; no re-audit of previously accepted gates.

## A. Final verdict

**ACCEPT PHASE 3**

Both remaining findings are closed. The targeted fixes introduce no material regression in the reviewed scope.

## B. Reviewed revisions

- Previous independent review: `43fa70fe2b2889451f81d23a1f51443cdb2889a6`
- Corrected implementation: `a098a0a3c66a30200bd85906212cb07ef8bd19eb`
- Separate remediation report: `87a5215624fdae29fec879c83bbcc3dba270e56a`
- Remote branch HEAD before this review: `87a5215624fdae29fec879c83bbcc3dba270e56a` (fetched; equal to local HEAD)
- Remote branch HEAD after this review: the pushed report-only review commit; its SHA is provided in the final handoff. The report cannot embed its own commit SHA.
- Final review commit SHA: provided in the final handoff for the same self-reference reason.

The implementation descends from the previous review SHA; the remediation report is a separate commit after implementation. The complete `43fa70f..a098a0a` implementation diff changes exactly `src/town_model/dataset/ontology.py`, `src/town_model/inspection/relation_diagnostics.py`, and the two existing D5 regression-test modules. No other implementation, test, data, checkpoint, or configuration path changed in that diff. The report commit adds only the remediation report.

## C. D5a — exact ontology IRI identity

I inspected `_resolve_iri()`, `_effective_base_iri()`, `OntologySnapshot.load()`, `ancestors()`, `_supported_subclass_closure()`, `geometric_relation_metadata()`, and `component_class_targets()`. Class declarations and named superclass resources are resolved against effective XML Base scope (including nested bases); absent XML Base uses the document URI. Named parent edges are added only by exact match against the declared-IRI-to-name map.

An independent temporary RDF/XML reproduction changed MushroomTopping's parent to `https://foreign.example/ontology#VegetableTopping`. The parser kept that reference out of `direct_parents`; MushroomTopping→VegetableTopping and MushroomTopping→VegetarianTopping were both `unknown`, and neither class entered MushroomTopping component targets. A nested `xml:base="https://foreign.example/ontology/"` with relative `#VegetableTopping` and an undeclared superclass produced the same unsupported/unknown result. This verifies that the foreign parent cannot contaminate equivalent-class closure.

Positive checks confirmed that the genuine relative `#VegetableTopping` and an absolute IRI exactly matching the declared VegetableTopping IRI are accepted as asserted. On the pinned ontology, MushroomTopping→PizzaTopping is `entailed`; MushroomTopping and the supported vegetable/cheese/fruit classes remain entailed to VegetarianTopping. JalapenoPepperTopping, PeperonataTopping, and PeperoniSausageTopping remain entailed to SpicyTopping. Unsupported or reversed cases—including VegetarianTopping→MushroomTopping, VegetarianTopping→VegetableTopping, MushroomTopping→MeatTopping, and MushroomTopping→SpicyTopping—remain `unknown`.

The pinned ontology SHA-256 is unchanged: `e349737f26aebd5b41c228043cdd35991775ecbacbc68a8f543c76466fe97253`. I independently compared all 22 component-target sets from the corrected parser with the parser loaded from the previous review SHA; all are identical. The inventory remains 66 OWL classes and 60 trainable classes; MushroomTopping ID 26 and VegetarianTopping ID 63 are unchanged. No canonical dataset was regenerated.

**D5a: CLOSED.** Exact-IRI validation prevents the reproduced false relationship while preserving supported ontology semantics and supervised targets.

## D. D5b — post-publication failure truthfulness

`save_relation_diagnostic()` serializes and fsyncs a same-directory temporary file, then `os.link()` atomically creates the complete immutable destination if absent on the tested local filesystem. Publication state is set immediately after successful linking. No `os.replace()` fallback exists. Pre-publication failures remain ordinary errors; post-publication sync/cleanup errors raise `RelationDiagnosticPostPublicationError`, retain the cause through `.cause` and `__cause__`, name the destination, state that complete JSON is visible, and warn against treating it as unpublished. The destination is never rolled back. CLI error handling surfaces the message.

I independently injected each post-publication failure:

- **Initial directory sync:** destination remained complete; error reported `published=True`, stage `initial_directory_sync`, unconfirmed durability and the original cause. The temporary hard link was left intact and matched the destination inode.
- **Temporary unlink:** destination remained complete; error reported `published=True`, stage `temporary_cleanup`, confirmed publication durability, incomplete cleanup and the original cause. Only the exact temporary path was injected; the immutable destination was not removed.
- **Final directory sync:** destination remained complete with no temporary entry; error reported stage `final_directory_sync`, completed cleanup, confirmed publication durability, unconfirmed cleanup durability and the original cause.

Pre-publication serialization, file-fsync and hard-link failures, plus an existing destination, were verified not to be mislabeled as post-publication. The independent two-process contention check on local ext4 had exactly one successful publisher and one `FileExistsError`; the reader observed complete parseable JSON, the final file matched the winner, and the loser could not overwrite it. The focused suite also covers these cases. No claim is made for distributed/network filesystems or untested Windows durability behavior.

**D5b: CLOSED.** Post-publication failures now communicate the actual publication state without weakening atomic no-clobber behavior.

## E. Regression results

Executed on Python 3.12.0:

- Full suite, `.venv/bin/python -m pytest -q`: **292 passed, 0 failed, 0 skipped**; 11 existing Pandas/Streamlit deprecation warnings.
- Focused D5a/D5b/relation tests: **36 passed**.
- Nine-real-image training smoke test: **9 passed**.
- `.venv/bin/python -m compileall -q src tests`: passed.
- `.venv/bin/python -m pip check`: no broken requirements.
- Required implementation-range `git diff --check`: passed.
- Independent ontology fixture, prior-parser target comparison, injected publication failures, and concurrent-process check: passed.

No verification was unavailable or skipped. The sealed test partition was not used; no controlled model-training experiment was required or run.

## F. Finding closure matrix

| Finding | Status | Evidence |
|---|---|---|
| D5a — Exact ontology IRI identity | **CLOSED** | Foreign absolute, nested-base, and undeclared parents stay unknown and do not contaminate targets; legitimate exact IRIs and all canonical targets are preserved. |
| D5b — Post-publication failure reporting | **CLOSED** | Three injected failures report accurate publication/durability/cleanup state and cause while preserving the destination; independent concurrent writers remain atomic and no-clobber. |

## G. Final Phase 3 gates

| Gate | Result | Assessment |
|---|---|---|
| 3A — Dataset, configuration, architecture | **PASS** | Previously accepted; no regression established. Ontology hash, vocabulary and target compatibility reconfirmed. |
| 3B — Visual features and ROI | **PASS** | Previously accepted; no affected code changed and nine real-image smoke cases pass. |
| 3C — Point models and composition | **PASS** | Previously accepted; no affected geometry/model code changed. |
| 3D — Training and experiments | **PASS** | Previously accepted; full suite and training smoke pass; no training contract was changed. |
| 3E — Execution, persistence and diagnostics | **PASS** | D5a and D5b are closed; publication and regression checks pass. |

## H. Remaining blockers

No material blocker remains within the accepted Phase 3 scope. Detector pretraining, predicted-ROI training, joint detector/ontology optimization, complete RQ2 evaluation and final held-out scientific comparisons remain outside this closure review and are not acceptance blockers here. This implementation acceptance is not evidence that town embeddings outperform baseline or box models.

## I. Final authorization

**ACCEPT PHASE 3 — The pre-detector GT-ROI image-to-ontology implementation milestone is independently accepted. All five Phase 3 acceptance gates pass. Implementation Phase 4 is authorized.**
