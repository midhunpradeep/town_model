# Provenance, adaptation, and licensing

## Pinned generator and ontology

- Generator: [SysReIC/Pizzaïolo](https://github.com/SysReIC/pizzaiolo), commit [`6c0bcb221a957c184012beaf103bebcde4b1f793`](https://github.com/SysReIC/pizzaiolo/tree/6c0bcb221a957c184012beaf103bebcde4b1f793).
- Upstream authors: Grégory Bourguin and Arnaud Lewandowski, SysReIC/LISIC/ULCO.
- License declared by upstream: **CC BY-NC 4.0**. This thesis dataset is intended for non-commercial research. Cite Grégory Bourguin and Arnaud Lewandowski, *Pizzaïolo Dataset: Des Images Synthétiques Ontologiquement Explicables*, [HAL record](https://hal.science/hal-04401953). See [`licenses/UPSTREAM_PIZZAIOLO.md`](licenses/UPSTREAM_PIZZAIOLO.md).
- The supplied ontology snapshot `src/town_model/dataset/assets/pizzaiolo.xml` is copied unchanged from that pinned source. SHA-256: `e349737f26aebd5b41c228043cdd35991775ecbacbc68a8f543c76466fe97253`. Prepared datasets preserve the generator's copy at `ontology/pizzaiolo.xml` and record its hash.
- The implementation downloads/uses a clean, detached checkout only during an explicit generation operation. The default managed checkout lives under ignored `artifacts/sources/pizzaiolo`; `PIZZAIOLO_SOURCE_DIR` or `--source-repo` may point to an already pinned, clean checkout. User-provided checkouts are verified and not modified. No upstream generator files are patched or vendored. The isolated worker applies a local deterministic adapter for upstream's seeded random selections over unordered Owlready collections and canonicalizes overlap-check iteration; this stabilizes random-type outputs without editing the pinned checkout.

## Reused preparation work

The dataset label predicates and two-stage, generator-type-stratified splitting behavior were adapted from the maintained-in-this-project source of truth's predecessor, `fourth_attempt/src/pizzaiolo_dataset_prep/{labels.py,splitting.py,config.py,generation.py}` and its tests. The original project remains untouched and is not imported at runtime. Existing recipe predicates, including `GenerousPizza`, `ProteinPizza`, `VerySpicyPizza`, and overlapping labels, are preserved. The split's rounding and seeded shuffle strategy match the tests from that project.

The predecessor did not include a license file. The adaptations in this repository are attributed to that project in code/docs; no independent third-party package is copied from it. Its concrete-component mappings omitted self classes and included `Germany` as if it were a rendered component. The new dataset implementation instead computes named superclass closure from the pinned ontology, preserves only supported equivalent-class-derived memberships, adds each concrete component's own class, and treats country names as individuals.

## Local source and generated artifacts

Generated datasets, downloaded sources, reports, and logs are not Python source and should be kept separate from maintained modules. The repository ignores `artifacts/sources/` and `artifacts/datasets/`; the small reproducibility example is intentionally stored under `examples/pizza-48/`. Raw generator CSV facts are retained under dataset `audit/`, clearly separate from model-facing targets. Pizzaïolo serializes the set-valued `hasTopping` field in process-dependent order, so the audit copy sorts that one list lexicographically; the manifest records this semantic-only canonicalization. Recipe metadata, origin, type, counts, and detector identities are audit/supervision data only and are not predictive input features.
