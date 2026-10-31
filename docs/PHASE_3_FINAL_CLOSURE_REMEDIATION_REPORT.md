# Phase 3 Final Closure Remediation Report (D5a and D5b)

**Purpose:** Implementation handoff for a separate independent closure review. This report does not declare Phase 3 accepted, authorize Phase 4, or request a merge to `main`.

## 1. Revision and scope

- Repository: `town_model`
- Branch: `remediation/phase-3-final-closure-findings`
- Starting independent closure-review commit: `9ed71818dbfffe70b5bcff107a29bd0866278b73`
- Reviewed implementation ancestor: `e1bdd41722c70160e56b4b1735f7ffd2fafb8684`
- Final implementation (source and test) commit: `4b129099c7562d61c89665e0662f6e27763f14ad`
- Scope: D5a ontology subclass metadata and D5b immutable relation-result publication only.

The starting branch was fetched and verified at the review SHA. The review commit is descended from `e1bdd417`; there were no later implementation commits on the branch to duplicate. Work was performed on a new branch created from `9ed71818`. The accepted D1–D4 implementation was not reopened.

## 2. Finding disposition

| Finding | Implementation status | Result |
|---|---|---|
| **D5a — known subclass entailment reported unknown** | **Corrected; regression-covered** | The supported VegetarianTopping and SpicyTopping equivalent-class consequences now appear as `entailed`, while unsupported expressions and directions remain `unknown`. |
| **D5b — relation result can be overwritten concurrently** | **Corrected; regression-covered** | Publication uses a same-directory atomic hard link that fails if any destination entry exists. Exactly one of two synchronized independent processes publishes. |

These are implementation results pending independent closure review, not an acceptance verdict.

## 3. D5a — ontology subclass metadata

### Reproduction and root cause

The pinned ontology digest is `e349737f26aebd5b41c228043cdd35991775ecbacbc68a8f543c76466fe97253`. Its RDF/XML contains:

- `MushroomTopping rdfs:subClassOf VegetableTopping`.
- `VegetarianTopping owl:equivalentClass` an anonymous intersection with operands `PizzaTopping` and the union `CheeseTopping ∪ FruitTopping ∪ VegetableTopping`.

The previous `geometric_relation_metadata()` only consulted named `rdfs:subClassOf` closure. The dataset target builder separately recognized equivalent-class consequences, so a Mushroom component was already positive for VegetarianTopping while containment metadata for those classes remained `unknown`.

Canonical IDs were resolved from the current vocabulary, not assumed: `MushroomTopping = 26`, `VegetarianTopping = 63`. The reproduced query is therefore `geometric_relation_metadata("MushroomTopping", "VegetarianTopping", "containment")`.

Inspection also confirmed a second supported expression in the ontology model: `SpicyTopping` is equivalent to `PizzaTopping ∩ ∃hasSpiciness.(Hot ∪ Medium)`. Existing component target logic already interprets named `hasSpiciness some Hot/Medium` subclass restrictions.

### Corrected logic and logical justification

`ancestors()` remains the closure of named `rdfs:subClassOf` edges. A separate narrow supported closure is shared by component target construction and relation metadata. It recognizes only the precise equivalent-class fragments already consumed by the dataset model:

1. **VegetarianTopping:** the parser requires the RDF/XML structure `named intersection (named union)`, with the exact required conjunct `{PizzaTopping}` and union members `{CheeseTopping, FruitTopping, VegetableTopping}`. For a source class `A`, it derives `A ⊑ VegetarianTopping` only if the named closure establishes `A ⊑ PizzaTopping` **and** `A` is a subclass of at least one member of that union. In the reproduced case, `MushroomTopping ⊑ VegetableTopping`; `VegetableTopping` is itself below `PizzaTopping` and is one union member. Thus `MushroomTopping` is below both intersection operands and consequently below `VegetarianTopping`.
2. **SpicyTopping:** the parser validates the exact intersection/restriction/union structure and `hasSpiciness` property. It derives `A ⊑ SpicyTopping` only where the named closure establishes `A ⊑ PizzaTopping` and `A` (or a named superclass of `A`) has a named `hasSpiciness some Hot` or `some Medium` restriction. Either named filler is below the union `Hot ∪ Medium`, satisfying the second intersection operand.

A direct named subclass axiom remains `asserted`; named transitive closure and these supported expression consequences are `entailed`. No entailment failure is labeled false. The metadata remains separate from geometric observations: a deliberately non-contained Mushroom region returns a geometric mismatch while its ontology metadata remains `entailed`.

The parser compares resolved IRIs against the ontology's declared class/property IRIs, not just local-name strings. It rejects unknown/nested operands, restrictions and expressions outside these exact supported patterns. In particular, it does not infer `VegetarianTopping ⊑ VegetableTopping` merely because VegetableTopping occurs as one disjunct, nor does a syntactic occurrence inside a nested expression establish subsumption. The implementation is not a general OWL reasoner.

### Negative and compatibility evidence

New ontology tests cover:

- Mushroom, Vegetable, Cheese, Parmesan, Fruit and Sultana subclass entailments to VegetarianTopping.
- Direct named assertion and transitive named closure.
- Unrelated and reversed pairs remaining unknown.
- The supported Jalapeno-to-SpicyTopping entailment and an unsupported Mushroom-to-SpicyTopping pair.
- An adversarial temporary RDF/XML fixture where VegetableTopping appears only inside a nested intersection under a union; the result and component target remain unsupported/unknown.
- Existing asserted and entailed disjointness metadata.
- A geometric counterexample that does not alter ontology metadata.

The vocabulary still has 66 classes and 60 trainable class IDs; the verified IDs remain 26 and 63. The component target set for MushroomTopping is unchanged. A comparison of the new shared closure against the prior component-target construction over all 22 detector categories found identical target sets. Existing dataset target/mask and vocabulary tests also pass. No prepared data, class IDs, trainable vocabulary, or supervision artifacts were rewritten.

## 4. D5b — atomic immutable relation-result publication

### Reproduction and root cause

The prior writer resolved the destination, checked `path.exists()`, wrote and synced a temporary JSON file, and then called `os.replace()`. The existence check and replacement were separate, so two writers could both pass the check and both replace the destination; the later writer silently won. Resolving the final path also followed a dangling destination symlink instead of treating the symlink entry itself as occupied.

### Publication mechanism and assumptions

`save_relation_diagnostic()` now:

1. Normalizes to an absolute **lexical** destination path, without resolving/following the final entry.
2. Creates a uniquely named temporary file in the destination directory.
3. Serializes complete JSON, flushes it and calls `fsync()` on the file.
4. Calls `os.link(temp, destination)`. On supported local filesystems this same-directory, same-filesystem operation atomically creates the destination only if no directory entry exists. An existing regular file or symlink causes `FileExistsError`; competing writers cannot replace the winner.
5. Calls directory `fsync` where supported, removes the temporary directory entry, and syncs the directory again where supported.

There is no `os.replace()` fallback and no process lock as the safety mechanism. Existing-target failures are surfaced as a clear immutable-destination `FileExistsError`; other hard-link failures retain their `OSError` subtype with an atomic-publication explanation. If hard links are not supported or permitted, publication fails closed.

The tested platform is Linux with a local filesystem supporting hard links and directory `fsync`. The contract assumes local filesystem `link(2)` create-if-absent semantics. Network/distributed filesystems are not claimed. On Windows, Python does not expose portable directory-handle `fsync`, so that synchronization is skipped; Windows hard-link publication was not exercised in this work. No non-atomic fallback is used on any platform.

### Concurrency, reader and failure evidence

The new deterministic concurrency test starts two spawned independent processes with different large JSON results. Both are synchronized at the publication primitive after serialization; an observing reader begins at the same barrier. Results:

- Exactly one process reports publication and the other receives `FileExistsError`.
- The destination parses as JSON on the first observation and exactly equals the complete winning input.
- No reader observed partial content; losing and winning temporary files were removed.
- A second test retries publication from a separate process and verifies the original destination bytes remain unchanged.

Additional tests cover a single successful writer; an existing regular destination; existing and dangling symlinks; serialization failure; a pre-publication hard-link permission failure; and a post-publication temporary-unlink failure. Ordinary failures clean temporary files. If a process is killed before publication, a hidden uniquely named temporary file may remain; no unsafe automatic sweeper is used. If interruption/error occurs after the hard link is published but before temporary cleanup, the destination is already complete and immutable; the test verifies the leftover is the same inode and demonstrates safe cleanup only after confirming `samefile(temp, destination)`. Non-matching or potentially active temporary files must not be swept.

A non-unsupported directory-sync error after linking can be reported even though the complete destination is already visible; callers should inspect the immutable path rather than retry with an overwrite assumption. The destination is never removed as rollback after publication.

## 5. Files changed

Implementation commit `4b129099c7562d61c89665e0662f6e27763f14ad` changes only:

- `src/town_model/dataset/ontology.py`
- `src/town_model/inspection/relation_diagnostics.py`
- `tests/test_phase3_d5a_ontology_entailments.py` (new)
- `tests/test_phase3_d5b_relation_publication.py` (new)

This report is a separate documentation change. No geometry algorithms, CLI/dashboard implementation, training objectives, image features, checkpoint calculations, previous reviews, prepared datasets, checkpoints, or experiment results were changed.

## 6. Regression verification

Environment: repository `.venv`, Python 3.12.0, PyTorch 2.5.1+cpu, Torchvision 0.20.1+cpu. CUDA was not used.

- Focused D5a/D5b: `.venv/bin/python -m pytest -q tests/test_phase3_d5a_ontology_entailments.py tests/test_phase3_d5b_relation_publication.py` — **20 passed**.
- All relation diagnostics plus D5 tests: `.venv/bin/python -m pytest -q tests/test_phase3_relations.py tests/test_phase3_d5a_ontology_entailments.py tests/test_phase3_d5b_relation_publication.py` — **24 passed**.
- Relevant snapshot/CLI/dashboard inspection: snapshot-axis round trip, `tests/test_phase3_cli.py`, and read-only saved-run dashboard relation inspection — **3 passed**, one pre-existing Streamlit/Pandas deprecation warning.
- Nine real-RGB classifier-family × composition smoke tests: `.venv/bin/python -m pytest -q tests/test_phase3_training.py::test_nine_real_image_training_smokes_update_save_reload_and_evaluate` — **9 passed**.
- Full suite: `.venv/bin/python -m pytest -q` — **281 passed, 0 failed, 0 skipped**, with 11 existing Streamlit/Pandas deprecation warnings. This includes all nine smoke cells and the previously accepted D1–D4 regression coverage.
- `.venv/bin/python -m compileall -q src tests` — passed.
- `.venv/bin/python -m pip check` — no broken requirements.
- `git diff --check` — passed.

No test-set model outcomes or test-driven geometry inspection were used. The controlled learning experiment was not rerun because neither fix changes training objectives, image features, classifier behavior, or checkpoint calculations.

## 7. Cross-cutting closure and limitations

- **D1–D4 remain intact:** no D1–D4 implementation was edited. Full regression tests passed.
- **Training/vision/checkpoints unaffected:** changes are isolated to ontology inspection/target-closure helpers and relation-result persistence. Canonical component targets were checked against their prior values; no training, vision, or checkpoint source changed.
- **Scientific data protection:** no prepared dataset, split, class ID, label/mask file, saved checkpoint, historical result, sealed test partition, or previous review report was modified.
- **Remaining limitations:** this is not full OWL reasoning. Unsupported equivalent-class shapes remain unknown. Town relation decisions retain their existing sampled/inconclusive semantics. No claim is made about Windows execution, network filesystems, distributed writers, or CUDA. A process crash can leave hidden temporary files; cleanup is intentionally conservative and documented above. Phase 3 acceptance and Phase 4 authorization remain for the independent reviewer only.

## 8. Recommendation

Recommend a narrow independent closure review of D5a and D5b against implementation commit `4b129099c7562d61c89665e0662f6e27763f14ad`, this report, the pinned ontology, and the full passing regression suite. Do not merge this branch into `main` as part of this remediation.
