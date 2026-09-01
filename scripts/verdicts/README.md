# Verification of the Phase 4 findings

One file per analysis run, recording the verdict given to every finding after checking it
against the text of the requirements it cites.

| File | Dataset | Partition | Findings |
|------|---------|-----------|----------|
| `uw_graph.json` | Uniwhere | graph-based | 107 |
| `uw_base.json` | Uniwhere | direct-clustering baseline | 109 |
| `v2i_graph.json` | V2I case study | graph-based | 87 |
| `v2i_base.json` | V2I case study | direct-clustering baseline | 86 |

## Verdicts

| Verdict | Meaning |
|---------|---------|
| `TP` | The finding holds against the text of the requirements it cites. |
| `BORDERLINE` | Defensible but resting on one reading of an underspecified requirement, or extrapolating beyond what the requirements entail. |
| `FP` | The finding does not hold. Includes gaps whose supposedly missing requirement is present elsewhere in the collection. |

## Format

Each key identifies a finding as `C<cluster>|<category>|<index>`, where the index is the
position of the finding within its category in the analysis report, counting from 1.
Values are either the verdict alone or the verdict followed by `|` and the reason.

```json
"C7|gaps|2": "FP|deletion is specified by R19 (C11), 'Students must be able to delete their room (if saved)'."
```

## Reproducing

```bash
# rebuild the worksheet: every finding next to the requirements it cites
python scripts/export_findings.py

# summarise the verdicts: precision, false positives per category
python scripts/verification_stats.py
```
