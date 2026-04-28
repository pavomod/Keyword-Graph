# Workflow: Keyword Graph from Requirements Engineering

Current code status:
- Phase 1: implemented and documented
- Phase 2: implemented and documented
- Phase 3: implemented and documented
- Phase 4: not implemented yet in `src/`

## Current Pipeline

1. Use a model, with optional fine-tuning, to extract keywords from `data/crowd.csv`.
2. Build a global semantic keyword graph from the predicted keywords.
3. Remove isolated nodes and optionally reduce low-degree nodes into hubs.
4. Cluster the graph with Node2Vec + K-Means + UMAP.
5. Compare keyword-graph clustering against direct requirement embedding clustering.

## Planned Follow-Up

The future phase described below is still a roadmap item:

- Use an LLM or a pretrained model to detect inconsistencies and suggest improvements for each cluster.
- Return the cluster, the requirement-to-node mapping, the inconsistent requirements with explanations, and a suggested improved requirement.

Example of a cluster inconsistency:
- The television cannot be turned on before 8 a.m.
- The irrigation system must be turned on before 8 a.m.

## Pipeline Overview

```text
Crowd CSV Dataset
      |
      v
[PHASE 1] flan-t5-base fine-tuning      -> Keyword Extraction
      |
      v
[PHASE 2] semantic graph construction   -> NetworkX (global similarity + traceability)
      |
      v
[PHASE 3] Node2Vec + K-Means + UMAP     -> Keyword Clusters
      |
      v
[PHASE 3b] direct comparison            -> Requirement embeddings vs keyword graph
```
