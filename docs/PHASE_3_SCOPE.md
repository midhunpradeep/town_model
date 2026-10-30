# Phase 3 scope — real-image-to-ontology embeddings

**Status:** implementation branch created from the independently accepted Phase 2 review commit. This scope note records the sequencing amendment before Phase 3 code changes; it does not alter the scientific specification or claim implementation/acceptance.

## Source and branch provenance

- Repository: `midhunpradeep/town_model`
- Remote source branch verified with `git ls-remote --heads origin remediation/phase-2-review-findings` before implementation: `fbe1063f2451270a89c9e0c110b988d6aafa4a27`
- Exact local source/base commit: `fbe1063f2451270a89c9e0c110b988d6aafa4a27` (`docs: add final Phase 2 acceptance review`)
- New working branch: `implementation/phase-3-image-embeddings`
- The Phase 2 accepted implementation is `087b116c4d06cd08af65d775136986a413cc5eff`; its final independent acceptance review is `fbe1063f2451270a89c9e0c110b988d6aafa4a27`. This branch is based on the latter, not `main`.

## Authority and scientific scope

The unchanged authority is `pizzaiolo_thesis_implementation_spec_v1_1_expanded.md`. This document records implementation sequencing only; it neither edits nor supersedes that specification. Phase 1/2 scope amendments, reports, and independent reviews remain preserved as historical project records.

Phase 3 builds a real-image visual-to-point and ontology-classification path using prepared Pizzaïolo images and ground-truth (GT) component boxes. It is deliberately **pre-detector exploratory work**: the current milestone precedes detector pretraining, so GT-ROI runs do not reproduce the canonical detector-pretrained stage 2, do not demonstrate end-to-end detection, and do not support scientific conclusions about component detection.

The accepted mathematical contracts are retained: bounded point heads; global, DeepSets-compositional, and hybrid pizza composition; shared baseline/box/town class semantics and geometry; explicit known masks; image-aware objective reduction; and exact physical geometry from Phase 2. Existing geometry APIs and public Phase 1/2 workflows are preserved. The detector remains replaceable: Phase 3 uses GT ROI providers behind an interface intended to accept future predicted ROIs and identities.

## Implementation sequencing

Work proceeds through the requested gates and keeps the original training stages distinct:

1. **3A — dataset/configuration/boundaries:** validate canonical manifests and prepared data, create deterministic train/validation subsets from their original partitions, define typed batch/config contracts, and enforce dependency direction with tests. Dataset generation remains an explicit operation; opening a model, run, or dashboard does not generate data.
2. **3B — shared visual path:** implement Torchvision MobileNetV3-Large/FPN and authoritative image/box preprocessing, coordinate transforms, GT ROIAlign and independent global ROI extraction; retain an interchangeable ROI provider.
3. **3C — point models/composition:** implement component/global projections, DeepSets sum and all three composition modes; assemble the existing classifier families through the accepted classifier contract and expose named result fields.
4. **3D — controlled learning:** implement the shared image-aware objective, correct effective-batch accumulation, frozen/trainable feature policies, optimizer/checkpoint/evaluation paths and the nine family×composition CPU-feasible real-image smoke configurations, followed by selected exploratory learning experiments when the environment permits.
5. **3E — run lifecycle/inspection/dashboard:** implement persisted inspection snapshots and shared local run services, then extend (not replace) the dashboard progressively with dataset/image inspection, saved embedding traces, metrics, comparisons and run controls.

Each gate is committed as coherent code/tests/results where resources allow. Tests and reports distinguish a smoke run from comparison-quality science and identify unverified requirements explicitly. Test data remains sealed for model-outcome inspection/checkpoint choice.

## Relationship to the specification's three training stages

The scientific training-stage sequence is unchanged:

- **Training stage 1:** detector pretraining.
- **Training stage 2:** freeze a detector pretrained in stage 1 and train ontology embeddings on GT ROIs.
- **Training stage 3:** joint detector/ontology fine-tuning with predicted ROIs.

This Phase 3 implementation milestone occurs **before stage 1**. It initially trains image embeddings on GT ROIs with ImageNet-pretrained backbone weights only. Such runs are labeled *pre-detector exploratory experiments*. They are not canonical stage-2 runs because no Pizzaïolo detector-pretraining checkpoint exists, and they are not stage-3/end-to-end detector runs. The future detector can supply predicted ROIs through the provider contract without changing point projection, composition or classifier mathematics.

## Limits of this scope note

The full 4,800-image data gate, architecture, experiments and local dashboard controls are acceptance goals, not claims that they have been achieved. No default branch merge is planned. No Phase 4 detector heads, detector matching/training, predicted-ROI joint optimization or detector conclusions are authorized by this scope.
