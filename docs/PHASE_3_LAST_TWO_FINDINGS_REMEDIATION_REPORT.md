# Phase 3 Last Two Findings Remediation Report

**Purpose:** Implementation handoff for a separate independent closure review. This report does not declare Phase 3 accepted or authorize Phase 4.

## 1. Revisions and scope

- Repository: `town_model`
- Working branch: `remediation/phase-3-last-two-findings`
- Exact starting independent-review SHA: `43fa70fe2b2889451f81d23a1f51443cdb2889a6`
- Latest remote branch HEAD at fetch: `43fa70fe2b2889451f81d23a1f51443cdb2889a6`; no newer remote commits were present.
- Final implementation SHA: `a098a0a3c66a30200bd85906212cb07ef8bd19eb`
- Scope: D5a named-subclass IRI identity and D5b post-publication failure reporting only.

The implementation and regression tests were committed separately from this report. No previous report or independent review was modified. No pinned ontology, dataset, checkpoint, generated artifact, or experiment result was changed.

## 2. Findings and original reproductions

### D5a — foreign named superclass accepted by local name

Before the correction, independently loaded a temporary RDF/XML copy with MushroomTopping's parent changed to `https://foreign.example/ontology#VegetableTopping`. The public `OntologySnapshot.load()` API reported:

- MushroomTopping → VegetableTopping containment: `asserted`
- MushroomTopping → VegetarianTopping containment: `entailed`
- Mushroom component targets included VegetableTopping and VegetarianTopping

The root cause was constructing named parent edges from `_local_name(rdf:resource)` and checking that local name against declared classes. It did not establish identity by the referenced class's complete IRI.

### D5b — post-publication exception hides visible result

Before the correction, independently injected `OSError("simulated directory fsync failure")` from `_fsync_directory()` after the hard link succeeded. `save_relation_diagnostic()` raised only that bare `OSError`, while the destination already existed and parsed as the complete JSON result.

The root cause was not tracking or reporting that `os.link()` had succeeded before subsequent directory synchronization and temporary cleanup.

## 3. D5a correction and evidence

### Exact IRI semantics

`OntologySnapshot.load()` now resolves declared class IRIs and named `rdfs:subClassOf` resource IRIs using `urljoin` with the effective in-scope `xml:base` (including nested XML Base values; the document URI is the base when no XML Base is present). It builds an explicit `declared_class_iri → canonical_class_name` reverse lookup. A named parent edge enters `direct_parents` only when its fully resolved IRI exactly matches a declared class IRI.

A foreign or unknown named parent is excluded from the supported named-parent graph. It is not rejected merely for being present in an unpinned fixture, but it cannot be promoted into a local asserted edge or entailment. The supported equivalent-class deductions were not rewritten.

### Positive and adversarial checks

The new regression tests verify:

- Legitimate relative `#VegetableTopping` and an absolute IRI exactly matching the declared VegetableTopping IRI remain asserted.
- A foreign absolute IRI with the same fragment and a foreign nested-`xml:base`/relative-reference variant are unsupported.
- Missing and unknown named parents do not create relations.
- Valid direct and transitive named subclass relations and MushroomTopping → VegetarianTopping entailment remain valid on the pinned ontology.
- Foreign-parent contamination creates neither Mushroom-to-Vegetable nor Mushroom-to-Vegetarian containment metadata and adds neither target to Mushroom component targets.
- Existing unsupported relationships remain `unknown`.

The pinned ontology SHA-256 remains `e349737f26aebd5b41c228043cdd35991775ecbacbc68a8f543c76466fe97253`. Regression coverage preserves the existing 66 class inventory, 60 trainable IDs, MushroomTopping ID 26, VegetarianTopping ID 63, and exact target sets for all 22 detector categories. The supported VegetarianTopping and SpicyTopping consequences remain covered.

## 4. D5b correction and evidence

### Publication-state contract

The existing same-directory atomic hard-link no-clobber primitive is unchanged. Publication state is set immediately after successful `os.link()`. Any later directory-sync or temporary-unlink exception raises `RelationDiagnosticPostPublicationError`, a distinct `RuntimeError` subclass exposing:

- `destination`
- `published`
- `stage`
- `durability_confirmed`
- `cleanup_completed`
- `cleanup_durability_confirmed`
- `cause` (also preserved as Python `__cause__`)

The message states that the complete JSON result is visible, names the destination, warns not to retry as if unpublished, and reports when directory-entry durability is unconfirmed. The destination is never rolled back, and publication is never retried internally.

`_fsync_directory()` now reports whether directory synchronization was available and completed. Unsupported directory synchronization is not claimed as confirmed. A failure at the initial directory sync reports `published=True` and `durability_confirmed=False`; a temporary unlink failure reports cleanup incomplete; a final sync failure reports the already-published entry's earlier sync status separately from cleanup durability. When initial sync failure prevents cleanup, a temporary hard link can remain and the error says so.

The CLI's existing error handling surfaces this actionable exception as a command-line error; a regression test verifies the path, visible-result warning, and complete destination contents. The dashboard already surfaces `RuntimeError` messages through its existing error display; no dashboard code changed.

### Injected failures and no-clobber checks

Focused tests inject and verify all three post-publication failures:

1. Initial directory synchronization failure.
2. Temporary-file unlink failure.
3. Final directory synchronization failure after temporary cleanup.

Each test verifies complete destination JSON, exception path and state attributes, and original cause. Pre-publication serialization, file-fsync, and hard-link failures are also tested; they do not produce the post-publication exception and clean temporary files where possible.

Existing tests continue to verify ordinary publication, immutable existing files and symlinks (including dangling links), and concurrent writers. The synchronized independent-process test still observes exactly one successful publisher, one `FileExistsError`, and complete JSON at first reader observation. No `os.replace()` or unsafe fallback was introduced.

## 5. Verification results

Observed in the repository `.venv`:

- Focused command — `.venv/bin/python -m pytest -q tests/test_phase3_d5a_ontology_entailments.py tests/test_phase3_d5b_relation_publication.py tests/test_phase3_relations.py` — **36 passed**.
- Full suite — `.venv/bin/python -m pytest -q` — **292 passed, 0 failed, 0 skipped**, with 11 existing Pandas/Streamlit deprecation warnings.
- Nine real-RGB classifier × composition smoke cases — `.venv/bin/python -m pytest -q tests/test_phase3_training.py::test_nine_real_image_training_smokes_update_save_reload_and_evaluate` — **9 passed**.
- `.venv/bin/python -m compileall -q src tests` — passed.
- `.venv/bin/python -m pip check` — no broken requirements.
- `git diff --check` — passed.

No controlled image-learning experiment was rerun: training, model geometry, image features, objectives, and checkpoints are outside this change. The sealed test partition was not used.

## 6. Files changed

Implementation commit `a098a0a3c66a30200bd85906212cb07ef8bd19eb` changes:

- `src/town_model/dataset/ontology.py`
- `src/town_model/inspection/relation_diagnostics.py`
- `tests/test_phase3_d5a_ontology_entailments.py`
- `tests/test_phase3_d5b_relation_publication.py`

This separate documentation commit adds only:

- `docs/PHASE_3_LAST_TWO_FINDINGS_REMEDIATION_REPORT.md`

No training, detector, geometry mathematics, dataset, checkpoint, vocabulary, CLI implementation, dashboard architecture, or pinned ontology file was changed.

## 7. Remaining limitations

- The ontology module remains a narrow supported-fragment parser, not a general OWL reasoner. Unsupported named parents are omitted from the supported graph and remain unknown.
- Atomic no-clobber behavior assumes same-filesystem hard-link semantics on a supported local filesystem. No distributed transaction guarantee is made; Windows hard-link/directory-sync and network filesystem behavior are not claimed or tested. There is no non-atomic fallback.
- If directory synchronization is unavailable, durability is not confirmed. An interrupted post-publication operation can leave a hidden temporary hard link; recovery must first verify it is the same inode as the published destination. The immutable destination is never automatically removed.
- No model-training experiment, sealed test partition, CUDA behavior, or unrelated Phase 3 gate was re-evaluated.

## 8. Recommendation

Recommend a narrow **final independent closure review** of D5a and D5b against implementation commit `a098a0a3c66a30200bd85906212cb07ef8bd19eb`, this report, and the passing regression suite. This remediation report makes no implementation or scientific acceptance claim and does not authorize Phase 4.
