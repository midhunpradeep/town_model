# Final Independent Phase 3 Acceptance Closure Review

**Project:** *Representing Ontologies Using Town Embeddings*
**Review scope:** Closure of D5a and D5b for the pre-detector, GT-ROI image-to-ontology milestone.

## A. Final verdict

**REQUEST CHANGES**

The no-clobber hard-link publication prevents concurrent overwrites, but its post-publication failure path does not tell callers that the complete destination may already exist. D5a also accepts a foreign named-subclass IRI when its local name collides with a declared class, producing an unsupported entailment. Both defects were independently reproduced. Phase 3 is therefore not accepted, and Phase 4 is not authorized.

## B. Reviewed revisions

- Previous independent review / starting closure-review commit: `9ed71818dbfffe70b5bcff107a29bd0866278b73`
- Final implementation commit: `4b129099c7562d61c89665e0662f6e27763f14ad`
- Remediation report commit: `d263e2c5a674a5f8fe669e62105af899c427e9db`
- New report-only review commit: reported in the final handoff after commit (a commit cannot embed its own content-derived SHA).
- Branch: `remediation/phase-3-final-closure-findings`
- Fetched `origin` before review. At that point, local and remote remediation-branch HEADs both resolved to `d263e2c5a674a5f8fe669e62105af899c427e9db`; there were no additional remote commits. The implementation commit descends from `9ed7181`, and the report commit descends from the implementation commit. `main` was not substituted.
- Inspected the complete `9ed7181..4b12909` diff. It changes exactly four paths: `src/town_model/dataset/ontology.py`, `src/town_model/inspection/relation_diagnostics.py`, and the two new D5 tests. The `4b12909..d263e2c` handoff diff adds only `docs/PHASE_3_FINAL_CLOSURE_REMEDIATION_REPORT.md`.
- No previous review, source file, test, config, dataset artifact, checkpoint, or experiment result was changed during this review.

## C. D5a independent evidence

### Pinned axioms and positive cases

Inspected the actual RDF/XML asset `src/town_model/dataset/assets/pizzaiolo.xml`; its SHA-256 is `e349737f26aebd5b41c228043cdd35991775ecbacbc68a8f543c76466fe97253`. It declares:

- `MushroomTopping ⊑ VegetableTopping` and `VegetableTopping ⊑ PizzaTopping` as named `rdfs:subClassOf` axioms.
- `VegetarianTopping ≡ PizzaTopping ∩ (CheeseTopping ∪ FruitTopping ∪ VegetableTopping)`.
- `SpicyTopping ≡ PizzaTopping ∩ ∃hasSpiciness.(Medium ∪ Hot)`.
- The relevant named class restrictions include `JalapenoPepperTopping ⊑ ∃hasSpiciness.Hot`, `PeperonataTopping ⊑ ∃hasSpiciness.Medium`, and `PeperoniSausageTopping ⊑ ∃hasSpiciness.Medium`.

On the pinned snapshot, independently observed:

- `MushroomTopping ⊑ VegetableTopping`: **asserted**.
- `MushroomTopping ⊑ PizzaTopping`: **entailed** by named transitive closure.
- `VegetableTopping ⊑ VegetarianTopping` and `MushroomTopping ⊑ VegetarianTopping`: **entailed**.
- The supported detector-category subclasses of `VegetarianTopping` are Garlic, Green Pepper, Jalapeno, Mushroom, Olive, Onion, Parmesan, Peperonata, Rocket, Sliced Tomato, Spinach, and Sultana toppings.
- The supported `SpicyTopping` consequences among detector categories are Jalapeno, Peperonata, and Peperoni Sausage toppings. The source restrictions use the exact declared property and Hot/Medium fillers.
- Existing direct assertions remain `asserted`; named transitive and recognized equivalent-expression consequences are `entailed`.

### Logical and parser checks

For `V ≡ P ∩ (A ∪ B ∪ C)`, the implementation derives `X ⊑ V` only when named closure establishes both `X ⊑ P` and `X ⊑ A` or `X ⊑ B` or `X ⊑ C`. This is the correct direction: membership in one disjunct alone does not imply the whole union's converse. For SpicyTopping, `X ⊑ P` plus `X ⊑ ∃hasSpiciness.Hot` (or Medium) entails the declared existential into the union filler and then the intersection. The implementation does not reverse the union or existential implications.

The equivalent-expression parser checks the declared class/property IRIs and the supported intersection/union/restriction structure. Independent altered-XML checks found that foreign property and filler IRIs in the Spicy definition remain unknown and are not added to component targets. Reordering the supported intersection and union operands preserved the entailments. The new nested-expression test also remains unknown. Reverse directions (`VegetarianTopping ⊑ MushroomTopping`, `VegetarianTopping ⊑ VegetableTopping`) and unrelated pairs remain unknown, not false.

However, the existing named-parent parser in `OntologySnapshot.load()` still maps `rdfs:subClassOf` references by `_local_name()` without verifying the reference IRI against `class_iris`. I independently copied the ontology to a temporary fixture and changed MushroomTopping's parent reference from `#VegetableTopping` to `https://foreign.example/ontology#VegetableTopping`. With that unpinned fixture:

- `geometric_relation_metadata("MushroomTopping", "VegetableTopping", "containment")` returned **asserted**.
- `geometric_relation_metadata("MushroomTopping", "VegetarianTopping", "containment")` returned **entailed**.
- `component_class_targets("MushroomTopping")` included `VegetarianTopping`.

An external class IRI is not an asserted relation to the locally declared class, even when its fragment has the same local name. This violates the requested exact-IRI and external-reference checks and invents a consequence for an altered/unpinned snapshot. The pinned asset itself is correct; the defect is in parser validation and is reproducible through the public snapshot API.

### Dataset and vocabulary compatibility

Independently loaded the pinned ontology and compared target sets for all **22** detector categories against the prior target-construction logic reconstructed from the pre-remediation implementation; all 22 sets are identical. The current vocabulary still contains **66 OWL classes**, **60 trainable classes**, `MushroomTopping` ID **26**, and `VegetarianTopping` ID **63**. The pinned ontology hash is unchanged. Since all component target sets and image target ancestry remain unchanged, target/mask semantics are unchanged. No canonical prepared dataset was regenerated or modified.

**D5a status: OPEN** because the external-IRI adversarial case produces a false asserted/entailed result.

## D. D5b independent evidence

### Publication algorithm and atomicity

The implementation lexically normalizes the destination, creates a unique temporary file in the destination directory, serializes complete JSON, flushes and `fsync`s it, then calls `os.link(temp, destination)`. The same-directory hard link is an atomic create-if-absent publication on the tested local filesystem. There is no `os.replace()` fallback. Directory synchronization is attempted where supported; the temporary link is removed after publication. Existing destination entries, including symbolic links, are not followed or replaced.

On Linux with the local ext4 filesystem used for independent tests, I synchronized two forked processes at the `os.link` boundary while an independent reader waited at the same barrier. The workers had different large JSON documents. Exactly one returned success and the other received `FileExistsError`; the reader's first observed file parsed as complete JSON and equaled the winner; the final destination still equaled that winner. A subsequent writer received `FileExistsError`, with the original destination bytes unchanged. No partial JSON was observed.

Independent checks also confirmed:

- Existing regular file, existing symlink, dangling symlink, and existing directory each fail without replacing the existing entry/bytes or following the dangling link.
- A simulated file-`fsync` failure occurs before publication, leaves no destination, and cleans the temporary file.
- The tested hard-link permission failure in the focused suite fails closed with no replacement fallback.
- A process forcibly exited at the link boundary after serialization left no destination and one harmless hidden temporary file, as expected for a crash that bypasses cleanup.
- The existing post-publication cleanup-failure test leaves the complete destination and does not roll it back.

### Remaining post-publication error defect

I independently injected an error from `_fsync_directory()` after `os.link()` had succeeded. `save_relation_diagnostic()` raised only `OSError("simulated directory fsync failure")`, while the destination already existed and contained the complete JSON. The exception did not say that publication had occurred or might have occurred. The same ambiguity applies to cleanup errors after publication. The code correctly does not delete the destination, but the caller cannot distinguish this state from a pre-publication failure based on the error.

The required failure contract explicitly asks that errors describe the possibility of an already-published result. A caller may otherwise treat the operation as wholly failed and retry, receiving an unrelated destination-exists error or mishandling provenance. The minimal correction is to track publication and raise a distinct, truthful post-publication error/status that names the destination and says the complete result may already be present; never roll back the destination. Add tests for post-link directory-sync and cleanup failures that assert both destination integrity and error clarity.

### Schema and filesystem limits

The result dictionary/schema, JSON content, provenance/checksum references, CLI and dashboard consumer APIs, and exact/sampled relation mathematics are unchanged by this publication-only diff. The tests ran on Linux/local ext4. Windows hard-link/directory-sync behavior, network/distributed filesystems, and CUDA were not tested or claimed. Same-directory temp creation ensures the temp and destination are on the same filesystem under the documented local-filesystem assumption.

**D5b status: OPEN** for the misleading post-publication failure path, although atomic no-clobber behavior itself is independently confirmed on the tested filesystem.

## E. Closure matrix

| Finding | Status | Evidence |
|---|---|---|
| D5a — Ontology entailment metadata | **OPEN** | Equivalent-class reasoning is sound for the pinned axioms and target sets are unchanged, but an external parent IRI sharing a declared class's local name is reported as asserted and propagates a false entailment. |
| D5b — Immutable concurrent publication | **OPEN** | Atomic no-clobber link publication passed synchronized process testing; post-publication sync/cleanup errors do not disclose that the destination may already be published. |

## F. Regression verification

Observed on Python 3.12.0 and PyTorch 2.5.1+cpu; CUDA was unavailable (`torch.cuda.is_available() == False`, CUDA version `None`). No GPU testing is claimed.

- `.venv/bin/python -m pytest -q` — **281 passed, 0 failed, 0 skipped**, 11 Streamlit/Pandas deprecation warnings (147.46 s).
- `.venv/bin/python -m pytest -q tests/test_phase3_d5a_ontology_entailments.py tests/test_phase3_d5b_relation_publication.py` — **20 passed**.
- `.venv/bin/python -m pytest -q tests/test_phase3_d5a_ontology_entailments.py tests/test_phase3_d5b_relation_publication.py tests/test_phase3_relations.py` — **25 passed**. This is the independently observed count for this combined command.
- `.venv/bin/python -m pytest -q tests/test_phase3_training.py::test_nine_real_image_training_smokes_update_save_reload_and_evaluate` — **9 passed**.
- `.venv/bin/python -m compileall -q src tests` — passed.
- `.venv/bin/python -m pip check` — no broken requirements.
- `git diff --check 9ed71818dbfffe70b5bcff107a29bd0866278b73..4b129099c7562d61c89665e0662f6e27763f14ad` — passed.
- Independent adversarial checks reproduced the external-IRI D5a defect and the post-publication D5b error ambiguity. Independent synchronized process publication, destination-entry, reader-integrity, and pre-link crash checks are detailed above.

The full passing suite and nine real-image smoke tests show no observed training, model, snapshot, dashboard, or relation-mathematics regression. No new training experiment was run, and the sealed test partition was not used.

## G. Phase 3 final gate matrix

| Gate | Result | Assessment |
|---|---|---|
| **3A — Dataset, configuration and architecture** | **PASS** | Previously accepted by the prior independent reviewer; this remediation does not alter these modules or contracts. Vocabulary and target compatibility were independently checked. |
| **3B — Visual features and ROI** | **PASS** | Previously accepted; no vision/ROI code changed, and the nine real-image smoke tests pass. |
| **3C — Point models and composition** | **PASS** | Previously accepted; no point-model, geometry, or composition implementation changed. |
| **3D — Training and comparisons** | **PASS** | Previously accepted; no objectives, weighting, training, or checkpoint code changed. Full suite and nine smoke tests pass. |
| **3E — Persistence and dashboard** | **FAIL** | D5a external-IRI entailment validation and D5b post-publication error truthfulness remain open. No dashboard or result-schema regression was found. |

The previous reviewer accepted 3A–3D and limited 3E's failure to D5a/D5b. No concrete regression justifies reopening the first four gates. Mixed exploratory model results and deferred detector/RQ2 work are not blockers in this closure review.

## H. Remaining defects

### D5a — external named-class IRI accepted by local name

- **Severity:** Moderate.
- **Affected module:** `src/town_model/dataset/ontology.py`, named `rdfs:subClassOf` extraction in `OntologySnapshot.load()` and its use by `ancestors()` / `_supported_subclass_closure()`.
- **Minimal reproduction:** In a temporary copy of the pinned RDF/XML, replace MushroomTopping's `rdfs:subClassOf` resource `#VegetableTopping` with `https://foreign.example/ontology#VegetableTopping`; load with `OntologySnapshot.load(fixture)`; query MushroomTopping-to-VegetableTopping and MushroomTopping-to-VegetarianTopping containment metadata.
- **Expected:** The foreign IRI is not the ontology's declared `VegetableTopping`; it must not be reported asserted, used in subclass closure, or add VegetarianTopping to component targets. Unsupported relation remains unknown.
- **Actual:** The local fragment is treated as the declared class: the first query is `asserted`, the second `entailed`, and the component target includes VegetarianTopping.
- **Impact:** Altered or unpinned ontology input can manufacture class targets and ontology relation claims from an external reference.
- **Necessary remediation:** Resolve and compare every named parent resource against the exact declared class IRI before adding a `direct_parents` edge; add an adversarial same-local-name foreign-IRI regression.

### D5b — post-publication errors are indistinguishable from pre-publication failure

- **Severity:** Moderate.
- **Affected module:** `src/town_model/inspection/relation_diagnostics.py::save_relation_diagnostic()`.
- **Minimal reproduction:** Monkeypatch `_fsync_directory()` to raise `OSError("simulated directory fsync failure")`; call `save_relation_diagnostic({"writer": "already-published"}, destination)`. The function raises that bare error after hard-link publication.
- **Expected:** The error must explicitly communicate that the complete immutable result may already be published at `destination`; no rollback of that destination.
- **Actual:** The caller receives only `simulated directory fsync failure`, while the destination exists and parses as the complete JSON. It does not identify the publication state.
- **Impact:** Callers cannot reliably distinguish a pre-publication failure from a published result whose subsequent directory sync/cleanup failed, risking incorrect retry and provenance handling.
- **Necessary remediation:** Track the successful link and surface a distinct, truthful post-publication error/status naming the destination; test directory-sync and cleanup failures while asserting the destination remains complete and intact.

## I. Final recommendation

**REQUEST CHANGES.** Make only the two minimal corrections in H and add their focused regressions. The concurrent no-clobber publication primitive itself is sound under the tested/documented local-filesystem assumptions, and the supported pinned-ontology deductions are logically sound; the blockers are the exact-IRI validation gap and truthful post-publication error reporting.

Phase 3 is not formally accepted on this review. **Phase 4 is not authorized.** This verdict concerns implementation acceptance only; it is not a judgment about predictive or structural superiority of town embeddings.
