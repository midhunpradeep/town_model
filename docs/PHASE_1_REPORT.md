# Phase 1 implementation report

**Status: implementation delivered for review; Phase 1 is not declared approved.** This report records the implementation and verification evidence. Approval remains with the thesis author and supervisor.

**Recorded:** 2026-10-28

**Implementation commit:** `99956608ac88fd5ab3011d28d57937c7b5e32983`

**Example dataset:** `examples/pizza-48/` (`pizza-dcc148789aecf8af86a3`)

## Scope delivered

Phase 1 is a model-independent Pizzaïolo dataset-preparation subsystem and the first usable local research dashboard. The maintained Python package is `src/town_model/`; it provides validated TOML/CLI settings, pinned-source generation isolated in a subprocess, ontology-aware preparation, a versioned JSONL wire schema, immutable manifests and checksums, deterministic type-stratified splits, split-aware support statistics, structured validation, shared application operations, and an explicit background generation workflow in Streamlit.

The dashboard includes dataset overview/provenance/support, image and annotation exploration with selectable component occurrences, validation, and explicit generation. Opening it does not generate data. The CLI and dashboard use the same application services. Dataset records and vocabularies do not require a model framework or an OWL reasoner to consume.

No detector, embeddings, geometry, loss, training, evaluation, or experiment-comparison functionality was implemented. The provided implementation specification is committed unchanged as `pizzaiolo_thesis_implementation_spec_v1_1_expanded.md` for traceability.

## Data and provenance decisions

- The example contains 48 generated images: three each for 15 named Pizzaïolo types and `random`; the saved split sizes are 16 train, 16 validation, and 16 test.
- The pinned ontology snapshot is SHA-256 `e349737f26aebd5b41c228043cdd35991775ecbacbc68a8f543c76466fe97253`. Its 66 OWL classes remain distinct from country individuals. The detector vocabulary has 22 foreground identities; ontology targets have 60 trainable classes, one fixed-positive class, and five unsupported classes with unknown masks.
- Image-label inference preserves the audited 32 overlapping source predicates. Component targets include their concrete class, ancestors, and the explicitly supported equivalent-class memberships. Pizza and component supervision-scope flags are distinct. Germany remains an individual, not a detector category.
- The isolated worker verifies the pinned clean upstream checkout and does not edit upstream files. It canonicalizes ontology candidate and overlap-check ordering so seeded random-type generation is repeatable despite unordered upstream collections. The audit copy sorts only the set-valued `hasTopping` list; this is recorded in the manifest. Repeated full generation with the same config produced the same dataset ID and identical immutable-asset checksums.
- Raw recipes, type, ingredient counts, origin, and other generator metadata are audit/supervision data, not predictive input. Incomplete or contradictory raw metadata fails preparation/validation rather than becoming a silent negative. `docs/PROVENANCE.md` and `docs/licenses/UPSTREAM_PIZZAIOLO.md` record attribution and the upstream CC BY-NC 4.0 restriction.

## Verification evidence

| Check | Result |
| --- | --- |
| Editable install: `python -m pip install -e '.[test]'` | Passed |
| Automated suite: `pytest -q` | **37 passed** |
| Python compilation: `python -m compileall -q src tests` | Passed |
| CLI generation, validation, and inspection against the pinned checkout | Passed |
| Saved 48-image example validation | **Pass; 0 errors, 178 low-class-support warnings** |
| Repeat generation using the same config | Dataset ID and all immutable-asset checksums identical |
| JSON Schema Draft 2020-12 check against a saved sample | Passed; pizza and component target vectors each contain 66 entries |
| Streamlit `AppTest` overview and image explorer | Passed |
| `town dashboard --headless` local-server health smoke test | Passed (`/_stcore/health` returned `ok`) |
| Pinned `baseline_model/pizzaiolo` checkout after generation | Clean; no source files changed |

The 178 warnings are expected for this deliberately small data-exploration example: many classes have no observed positive in one or more splits. They are retained in `examples/pizza-48/reports/validation.json` and exposed in the CLI/dashboard. This example is not evidence of adequate training support and the warnings were not “fixed” by changing splits or fabricating positives.

## Artifacts and handoff

- [`README.md`](../README.md): install, CLI, dashboard, and test commands.
- [`dataset.toml`](../dataset.toml): reproducible small all-type configuration.
- [`docs/DATASET_SCHEMA.md`](DATASET_SCHEMA.md) and [`schemas/sample-v1.schema.json`](../schemas/sample-v1.schema.json): dataset wire contract.
- [`examples/pizza-48/manifest.json`](../examples/pizza-48/manifest.json): resolved configuration, source/preparation revision, ontology/vocabulary provenance, split policy, and asset checksums.
- [`examples/pizza-48/reports/validation.json`](../examples/pizza-48/reports/validation.json): machine-readable validation result and support diagnostics.

To use the committed example locally:

```bash
town validate examples/pizza-48
town inspect examples/pizza-48
town dashboard
```

The generator source cache and any larger generated datasets remain outside maintained source under ignored `artifacts/` paths. A next human review should assess the schema/supervision contract and the low-support implications before any later phase is authorized. **No later-phase implementation or Phase 1 approval is implied by this report.**
