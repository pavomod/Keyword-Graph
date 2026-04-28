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
- Clustering diagnostics report in JSON
- Elbow and cluster-quality image for k-selection interpretation

Default outputs:
- `results/clustering_report.json`
- `results/K.png`

Optional comparison output:
- `results/direct_clustering_report.json`
- `results/clustering_comparison.json`

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
- Compute inertia, Davies-Bouldin, and Calinski-Harabasz for each
- Select the k with the best combined score from Davies-Bouldin and Calinski-Harabasz

When `selected_k` is provided, the code still bounds it to the available sample count. If the graph has fewer than 2 nodes, the phase returns a single-cluster payload.

### 5. Run K-Means and annotate nodes

K-Means assigns a `cluster_id` to each node.

Then UMAP projects vectors to 2D and writes:
- `umap_x`
- `umap_y`

These attributes are added directly to graph nodes.

### 6. Export diagnostics and mappings

The keyword-graph report contains:
- `clustering` summary (k, scores, feature dimensions, requirement feature metadata, isolated node count)
- `cluster_requirements_mapping`
- `requirement_to_cluster_mapping`
- `nodes` with keyword, cluster, and UMAP fields

The direct requirement clustering report uses the same top-level keys, but its `clustering` section includes `mode: "direct_requirement_clustering"` and its `nodes` entries contain `requirement_id` and the requirement text.

## Processing Order Summary

1. Build Node2Vec vectors
2. Build and reduce requirement feature matrix
3. Fuse feature blocks
4. Select k (manual or automatic)
5. Run K-Means
6. Project to 2D with UMAP
7. Export reports and plots

## Main Code Entry Points

- `src/run_pipeline.py`
- `src/clustering/clustering.py`

## Final Output Context

After Phase 3, the graph is ready for interpretation and downstream analysis:
- Each concept has a cluster assignment
- Requirement-level traceability is preserved across the full pipeline
- The optional direct comparison output can be used to measure how different the keyword-graph clusters are from direct requirement embedding clusters
