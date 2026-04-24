# Phase 3: Clustering and Requirement-Aware Features

This step groups graph concepts into themes and explains cluster behavior with requirement-aware features.

## Quick Purpose

Phase 3 answers two questions:
- Which keyword nodes belong to the same semantic theme?
- How do graph structure and requirement provenance jointly shape clusters?

## What This Step Works On

Input graph from Phase 2, including:
- Semantic nodes and edges
- Node attributes `requirement_ids` and `requirement_count`

This allows clustering to combine:
- Structural signals (Node2Vec)
- Requirement traceability signals (incidence features)

## What This Step Produces

- Cluster IDs assigned to graph nodes
- 2D coordinates (`umap_x`, `umap_y`) for visualization
- Cluster labels generated with FLAN-T5
- Clustering diagnostics report in JSON
- Elbow/silhouette image for k-selection interpretation

Default outputs:
- `results/clustering_report.json`
- `results/elbow.png`

Optional comparison output:
- `results/direct_clustering_report.json`

## How This Step Works

### 1. Node2Vec embedding

Build dense vectors for each keyword node using weighted random walks.

Main controls include:
- `embedding_dim`
- `node2vec_walk_length`
- `node2vec_num_walks`
- `node2vec_workers`
- `random_state`

### 2. Requirement-aware feature matrix

For each node:
- Read `requirement_ids`
- Build a binary incidence vector over all unique requirement IDs

Then reduce dimensionality with Truncated SVD to keep features compact.

Main controls:
- `requirement_svd_dim`
- `requirement_feature_weight`

If requirement IDs are unavailable, this block is disabled automatically and clustering falls back to pure Node2Vec.

### 3. Feature fusion

Fusion strategy:
- Normalize Node2Vec block
- Normalize requirement block
- Concatenate `[node2vec, requirement_block * weight]`

This balances topology with explicit requirement context.

### 4. Choose number of clusters (k)

Two modes:

1. Manual
- Use `requested_k` (bounded by node count)

2. Automatic
- Evaluate candidates in `[k_min, k_max]`
- Compute inertia and silhouette for each
- Select the k with highest silhouette

### 5. Run K-Means and annotate nodes

K-Means assigns a `cluster_id` to each node.

Then UMAP projects vectors to 2D and writes:
- `umap_x`
- `umap_y`

These attributes are added directly to graph nodes.

### 6. Generate cluster labels with FLAN-T5

For each cluster:
- Collect top keywords (up to 20)
- Prompt FLAN-T5 for a short descriptive label (2-5 words)
- Clean and store the generated label

Result example:
- `"0": "user authentication"`

### 7. Export diagnostics and mappings

The report contains:
- `clustering` summary (k, scores, feature dimensions)
- `cluster_requirements_mapping`
- `requirement_to_cluster_mapping`
- `cluster_labels`
- `nodes` with cluster and UMAP fields

## Processing Order Summary

1. Build Node2Vec vectors
2. Build and reduce requirement feature matrix
3. Fuse feature blocks
4. Select k (manual or automatic)
5. Run K-Means
6. Project to 2D with UMAP
7. Label clusters with FLAN-T5
8. Export reports and plots

## Main Code Entry Points

- `src/run_pipeline.py`
- `src/clustering/clustering.py`

## Final Output Context

After Phase 3, the graph is ready for interpretation and downstream analysis:
- Each concept has a cluster assignment
- Clusters have human-readable labels
- Requirement-level traceability is preserved across the full pipeline
