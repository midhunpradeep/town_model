# A1 Decision Completeness and Provenance Audit

**Role:** Reproducible index of the canonical source, not a second scientific authority. Decision text and mathematical meaning remain solely in the [canonical A1 ledger](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md).

## Source identity and authority categories

- **Canonical source:** [`A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md`](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md), titled *A1 Scientific Decision Ledger — 70 Active Approvals*, author handoff dated 2026-11-01.
- **Expected and verified size:** 34,910 bytes.
- **Expected and verified SHA-256:** `f552b60f10f96d9e76585331b99e8543d8b7a81535c261386a93aa697b0489b9`.
- **Author provenance:** the source identifies the authority as author-approved mathematical decisions F0.1–F10.16, distinguishes them from the historical implementation draft, and records the approved source date. This repository copy was copied byte-for-byte from the supplied local source; no text was added to or removed from it.

The canonical document classifies its contents as follows; these categories must not be merged:

| Source content | Status | Location in canonical source |
|---|---|---|
| All 70 active F decisions | Author-approved scientific requirements | §1, lines 18–128; exact active IDs and source lines are indexed below. |
| F3.6 | Superseded historical choice; **not active** | §0 and F3.6-R at line 43; superseded by F3.6-R. |
| Mathematical consequences | Derived results, not additional approvals | §2, lines 130–138. |
| Accepted dataset, vision, point, artifact and evaluation infrastructure | Historical engineering context; does not revive retired classifiers | §3, lines 140–147. |
| Twelve open gates | Unresolved P0 (1–5), P1 (6–9), and P2 (10–12) topics | §4, lines 149–170. |
| Foundational audit questions A–J | Ten unresolved questions, not approvals | §5, lines 172–193. |
| Future implementation recommendations | Recommendations derived from approved constraints, not stand-alone authorization | §6, lines 196–205. |

## Reproduction command

Run from the repository root with Python 3. The check uses only the standard library, verifies the exact source bytes before parsing, requires the precise expected active ID set and group counts, and checks the supersession, twelve gates, and A–J audit inventory. It does not modify the source.

```python
from collections import Counter
from hashlib import sha256
from pathlib import Path
import re

path = Path("docs/reconstruction/A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md")
raw = path.read_bytes()
assert len(raw) == 34910, len(raw)
assert sha256(raw).hexdigest() == "f552b60f10f96d9e76585331b99e8543d8b7a81535c261386a93aa697b0489b9"
text = raw.decode("utf-8")

active = []
for line_number, line in enumerate(text.splitlines(), 1):
    match = re.match(r"^- \*\*(F(?:10|[0-9])\.\d+(?:-R)?) — (.+?)\*\*", line)
    if match:
        active.append((match.group(1), line_number, match.group(2)))

expected_counts = {"F0": 1, "F1": 1, "F2": 8, "F3": 11, "F4": 4,
                   "F5": 6, "F6": 5, "F7": 8, "F8": 5, "F9": 5, "F10": 16}
counts = Counter(re.match(r"(F\d+)", item[0]).group(1) for item in active)
expected_ids = {f"F{family}.{number}"
                for family, count in ((0, 1), (1, 1), (2, 8), (3, 11), (4, 4),
                                      (5, 6), (6, 5), (7, 8), (8, 5), (9, 5), (10, 16))
                for number in range(1, count + 1)}
expected_ids.remove("F3.6")
expected_ids.add("F3.6-R")
ids = [item[0] for item in active]
assert len(ids) == 70 and len(set(ids)) == 70
assert set(ids) == expected_ids, (sorted(set(ids) - expected_ids), sorted(expected_ids - set(ids)))
assert dict(counts) == expected_counts, counts
assert "F3.6-R" in ids and "F3.6" not in ids
assert "supersedes F3.6" in text

gates = text.split("## 4. Open gates", 1)[1].split("## 5. Foundational audit", 1)[0]
gate_numbers = re.findall(r"(?m)^(\d+)\. ", gates)
assert gate_numbers == [str(number) for number in range(1, 13)], gate_numbers
audit = text.split("## 5. Foundational audit", 1)[1].split("## 6. Instructions", 1)[0]
audit_letters = re.findall(r"(?m)^\*\*([A-J])\. ", audit)
assert audit_letters == list("ABCDEFGHIJ"), audit_letters
audit_report = Path("docs/reconstruction/A1_DECISION_COMPLETENESS_AUDIT.md").read_text(encoding="utf-8")
index_section = audit_report.split(chr(10) + "## Active approval IDs and source locations" + chr(10), 1)[1].split(chr(10) + "## Verification record" + chr(10), 1)[0]
indexed = [(decision_id, int(line_number), title) for decision_id, line_number, title in
           re.findall(r"(?m)^\| (F(?:10|[0-9])\.\d+(?:-R)?) \| (\d+) \| (.*?) \|$", index_section)]
assert indexed == active, (indexed, active)

print(f"source_sha256={sha256(raw).hexdigest()} bytes={len(raw)}")
print(f"active_unique_ids={len(ids)} family_counts={dict(sorted(counts.items(), key=lambda item: int(item[0][1:])))}")
print("supersession=F3.6-R active; F3.6 superseded and not counted")
print(f"open_gates={len(gate_numbers)} foundational_audit={len(audit_letters)}")
for decision_id, line_number, title in active:
    print(f"{decision_id}\t{line_number}\t{title}")
```

The extraction pattern accepts only the bold decision-heading bullets in §1; mentions of IDs elsewhere in the source do not inflate the count. To use the index, compare each active ID and verbatim short heading with its linked source line. The source hash ensures the line map is tied to the immutable supplied bytes.

## Active approval IDs and source locations

The short headings below are copied from the decision headings and are locators only. The numbered source line is the canonical location.

| Active ID | Source line | Heading (locator only) |
|---|---:|---|
| F0.1 | 22 | Unary class predicates. |
| F1.1 | 23 | Scoped closed-world generator targets. |
| F2.1 | 27 | One observed entity maps to one latent point. |
| F2.2 | 28 | Membership-first semantics. |
| F2.3 | 29 | One shared semantic domain. |
| F2.4 | 30 | Per-occurrence points. |
| F2.5 | 31 | Visual inference, no target leakage. |
| F2.6 | 32 | Learned latent axes. |
| F2.7 | 33 | Bounded/unbounded controlled comparison. |
| F2.8 | 34 | Configurable dimension. |
| F3.1 | 38 | Global class regions. |
| F3.2 | 39 | Exact set membership. |
| F3.3 | 40 | Membership-first learning. |
| F3.4 | 41 | Closed axis-aligned boxes. |
| F3.5 | 42 | Town construction. |
| F3.6-R | 43 | Open window interior, retained boundaries. |
| F3.7 | 44 | Contained vs independent windows. |
| F3.8 | 45 | Empty-set distinctions. |
| F3.9 | 46 | Fixed per-run capacity. |
| F3.10 | 47 | Positive trainable primitive widths. |
| F3.11 | 48 | No-window equivalence. |
| F4.1 | 54 | Flat point classifier interface. |
| F4.2 | 55 | Linear multilabel baseline. |
| F4.3 | 56 | DomainThing is fixed. |
| F4.4 | 57 | Separately typed outputs. |
| F5.1 | 61 | Box signed coordinate clearance. |
| F5.2 | 62 | Module separation. |
| F5.3 | 63 | Exact `min`/`max` composition. |
| F5.4 | 64 | Representative gradient coverage. |
| F5.5 | 65 | Framework AD tie subgradients. |
| F5.6 | 66 | Raw physical units. |
| F6.1 | 70 | Signed logistic and hinge controlled comparison. |
| F6.2 | 71 | Fixed temperature and margin. |
| F6.3 | 72 | Baseline BCE-with-logits. |
| F6.4 | 73 | Two additional reference modes. |
| F6.5 | 74 | Source-gated town reference correction. |
| F7.1 | 82 | Valid geometry by construction. |
| F7.2 | 83 | Center and positive half-width. |
| F7.3 | 84 | Validated float32 invariant domain. |
| F7.4 | 85 | Boundary approach versus evaluation. |
| F7.5 | 86 | Separate min width and clearance. |
| F7.6 | 87 | Matched physical meaning across domains. |
| F7.7 | 88 | Label-agnostic dimension-aware initialization. |
| F7.8 | 89 | Reject invalid optimization updates. |
| F8.1 | 95 | Weighted known-class means; entity/image hierarchy. |
| F8.2 | 96 | Unit weights primary; named variants. |
| F8.3 | 97 | Fixed task coefficients. |
| F8.4 | 98 | Undefined enabled tasks. |
| F8.5 | 99 | Accumulation optional only on measured need. |
| F9.1 | 105 | Three scientific stages. |
| F9.2 | 106 | Functional optimizer groups. |
| F9.3 | 107 | No raw geometric weight decay initially. |
| F9.4 | 108 | AdamW with constant initial schedule. |
| F9.5 | 109 | Global gradient norm clipping. |
| F10.1 | 113 | Global geometry primary for RQ2. |
| F10.2 | 114 | Isolate union versus window capacity. |
| F10.3 | 115 | Supplementary fixed-point geometry benchmark. |
| F10.4 | 116 | Raw-score scale diagnostics. |
| F10.5 | 117 | Topology and finite-box-union theorem. |
| F10.6 | 118 | Sound relation certification. |
| F10.7 | 119 | Optional compactness bias only. |
| F10.8 | 120 | Optional auxiliary point organization only. |
| F10.9 | 121 | Three-state relation reference with provenance. |
| F10.10 | 122 | Separate vacuity from formal truth. |
| F10.11 | 123 | Predeclared RQ2 pair population and multidimensional metrics. |
| F10.12 | 124 | Two distinct empirical violation distributions. |
| F10.13 | 125 | Minimum-enclosure theorem as diagnostic. |
| F10.14 | 126 | Box sum-of-widths compactness first. |
| F10.15 | 127 | Fixed-point compactness before co-adaptation. |
| F10.16 | 128 | Weak point organization study sequence. |

## Verification record

The first R1 source check reproduced the expected SHA-256 and 34,910-byte size before copying and confirmed byte equality afterward. The active-heading extraction found 70 unique IDs with family counts F0=1, F1=1, F2=8, F3=11, F4=4, F5=6, F6=5, F7=8, F8=5, F9=5, F10=16. F3.6-R is active; F3.6 is superseded and excluded. The 12 open-gate items and A–J foundational questions are present in their separate source sections. Independent review of this report and repository copy remains pending; see [R1 completion report](R1_COMPLETION_REPORT.md).
