# Workflow: Keyword Graph from Requirements Engineering

1) Use a model (Optional Fine tune) to extract keywords from a requirements dataset (crowd).

2) Build an algorithm that creates an N:N relationship graph among keywords extracted from the model.

3) Use an unsupervised approach such as K-means to generate clusters in the generated graph.

4) Use an LLM to detect inconsistencies and suggest improvements for each cluster.

## Pipeline Overview

```text
Crowd CSV Dataset
      |
      v
[PHASE 1] flan-t5-base fine-tuning      -> Keyword Extraction
      |
      v
[PHASE 2] N:N graph construction         -> NetworkX (co-occurrence + semantic similarity)
      |
      v
[PHASE 3] Node2Vec + K-Means            -> Keyword Clusters
      |
      v
[PHASE 4] LLM analysis                  -> Inconsistencies + Suggestions
```
