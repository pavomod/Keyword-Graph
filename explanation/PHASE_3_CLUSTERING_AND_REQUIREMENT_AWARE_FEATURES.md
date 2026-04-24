# Phase 3 Technical README: Clustering (Node2Vec + K-Means + UMAP)

This document explains how Phase 3 is implemented in the current pipeline.

## Scope of Phase 3

Phase 3 covers:
- Graph node embedding with Node2Vec
- Requirement-aware feature augmentation
- Cluster selection and assignment with K-Means
- 2D projection with UMAP for visualization
- Clustering diagnostics export (JSON + elbow/silhouette chart)

Main modules:
- `src/run_pipeline.py`
- `src/clustering/clustering.py`

## Phase 3 Input Contract

Input graph comes from Phase 2 and already contains:
- Keyword nodes
- Semantic edges
- Node traceability attributes:
  - `requirement_ids`
  - `requirement_count`

This means clustering can use both:
- Graph structure (Node2Vec)
- Explicit requirement provenance (requirement feature matrix)

## End-to-End Pipeline in Phase 3

1. Build Node2Vec embeddings for each keyword node
2. Build requirement-incidence matrix from `requirement_ids`
3. Reduce requirement matrix dimensionality with Truncated SVD
4. Normalize and combine feature blocks (Node2Vec + requirement block)
5. Select `k` (manual or auto by silhouette)
6. Run K-Means
7. Compute UMAP 2D coordinates
8. Annotate graph nodes and export clustering report

## Node2Vec Embeddings

Node2Vec is computed with edge `weight` as walk bias.

Main controls:
- `embedding_dim`
- `node2vec_walk_length`
- `node2vec_num_walks`
- `node2vec_workers`
- `random_state`

Output:
- One dense vector per node keyword

## Requirement-Aware Feature Block

For each node keyword:
- Read `requirement_ids` from node attributes
- Create a binary row over all unique requirement IDs in the graph
- Value `1.0` if keyword appears in that requirement, else `0.0`

The binary matrix is reduced using Truncated SVD to avoid excessive dimensionality.

Main controls:
- `requirement_svd_dim`
- `requirement_feature_weight`

Combination rule:
- Normalize Node2Vec block
- Normalize requirement block
- Concatenate `[node2vec, requirement_block * weight]`

If no requirement IDs are available:
- Requirement block is disabled automatically
- Clustering falls back to pure Node2Vec features

## K Selection Strategy

Two modes are supported:

1. Manual `k`
- If `requested_k` is provided, that value is used (bounded by node count)

2. Automatic `k`
- Evaluate `k` in `[k_min, k_max]`
- Compute inertia and silhouette for each candidate
- Select the candidate with maximum silhouette

## K-Means and Cluster Labels

K-Means runs on the final combined feature vectors.

For each node, the following attributes are written into the graph:
- `cluster_id`
- `umap_x`
- `umap_y`

These values are later available in the GEXF output.

## UMAP Projection

UMAP projects clustering vectors to 2D for visualization.

Details:
- `n_components=2`
- `n_neighbors` is automatically adapted to node count
- Robust handling for tiny graphs (`0` or `1` node)

## Clustering Report Output

JSON report default path:
- `results/clustering_report.json`

Main report sections:
- `clustering`
  - `selected_k`
  - `num_nodes`
  - `embedding_dim`
  - `effective_feature_dim`
  - `inertia`
  - `silhouette`
  - `k_selection` (strategy + candidate scores)
  - `elbow_plot`
  - `requirement_features` (enabled, dims, weight, requirement count)
- `nodes`
  - `keyword`
  - `cluster_id`
  - `umap_x`
  - `umap_y`

Elbow/silhouette plot default path:
- `results/elbow.png`

## Outputs Produced by Phase 3

- Enriched graph node attributes (`cluster_id`, `umap_x`, `umap_y`) in final GEXF
- Clustering report JSON for diagnostics and downstream analysis
- Elbow/silhouette plot image for `k` interpretation
