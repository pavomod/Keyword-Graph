# Keyword-Graph Pipeline: Overview

## Project Status

This project implements a complete pipeline for analyzing software requirements through semantic graph clustering.

- **Phase 1**: ✅ Keyword Extraction from Requirements
- **Phase 2**: ✅ Semantic Graph Construction
- **Phase 3**: ✅ Keyword Clustering and Visualization
- **Phase 4**: 🔄 Planned — Inconsistency Detection and Resolution

## What This Project Does

The pipeline transforms raw requirement documents into structured semantic clusters, enabling:

1. **Automated keyword extraction** — Understanding what concepts are mentioned in each requirement
2. **Semantic relationship mapping** — Discovering how requirements interconnect through shared concepts
3. **Concept clustering** — Grouping related ideas into coherent themes
4. **Requirement traceability** — Maintaining links from clusters back to original requirements
5. **Consistency analysis** — Comparing different clustering strategies to validate findings

## Data Flow

```
Requirement Dataset (CSV)
    ↓
[Phase 1] Extract Keywords from Each Requirement
    ↓
[Phase 2] Build Semantic Graph of Keywords
    ↓
[Phase 3] Cluster Keywords + Compare Approaches
    ↓
Clustered Requirements + Traceability Metadata
```

## Key Innovation: Requirement-Aware Clustering

Instead of clustering only based on graph structure, this pipeline combines:
- **Graph topology** — How keywords connect to each other
- **Requirement linkage** — Which original requirements mention each keyword

This dual approach reduces noise and produces more coherent clusters aligned with actual requirement semantics.

## Future Work

Phase 4 will use language models to detect inconsistencies within clusters and suggest improvements.

**Example**: If a cluster contains "turn on before 8 AM" and "turn off before 8 AM" for different systems, the pipeline will flag this as a potential inconsistency requiring clarification.
