# Phase 3: Semantic Clustering and Comparison

## Executive Summary

Phase 3 groups keywords into coherent semantic clusters and compares two clustering approaches. The key innovation is "requirement-aware" clustering, which combines graph structure with explicit requirement linkage to produce clusters that reflect both keyword relationships and original requirement context. The comparison validates whether graph-based clustering produces similar results to direct requirement clustering, increasing confidence in the findings.

## What Phase 3 Achieves

**Input**: Semantic graph from Phase 2 with all traceability metadata  
**Output**: 
- Cluster assignments for all keywords
- 2D visualization coordinates
- Diagnostic reports comparing two clustering approaches

The phase answers:
- Which keywords belong to the same semantic theme?
- Are clusters coherent when validated against original requirements?
- How do graph-based and direct requirement clustering compare?

## Clustering Architecture: Two Parallel Approaches

### Approach A: Graph-Based Clustering (Primary)

This approach leverages the graph structure built in Phase 2.

#### Step 1: Graph Embeddings (Node2Vec)

**Purpose**: Convert graph structure into dense numerical vectors

**How Node2Vec works**:
- Performs random walks starting from each node
- Walks explore the graph topology, favoring neighbors with strong connections
- The walk sequences are treated like language: frequent patterns are captured
- A Skip-gram model learns embeddings where "close" nodes in walk patterns get similar vectors

**Configuration parameters**:
- `embedding_dim` — Vector dimensionality (default: 64)
- `walk_length` — Steps per random walk (default: 30)
- `num_walks` — Walks per node (default: 200)
- `workers` — Parallel processing (default: 4; set to 1 for reproducibility)

**Result**: Each keyword gets a 64-dimensional vector representing its structural role in the graph

#### Step 2: Requirement-Aware Features (Innovation)

**Purpose**: Inject explicit requirement context into the clustering

**How it works**:
- For each keyword node, extract the list of requirements where it appears
- Build a binary indicator vector: `[req_1, req_2, ..., req_N]`
  - 1 if the keyword appears in that requirement, 0 otherwise
- This vector can be very high-dimensional (one dimension per requirement)
- Reduce dimensionality using Truncated SVD to keep features compact (default: 16 dimensions)

**Why this matters**: Keywords that appear in many of the same requirements are probably related, even if they're not structurally close in the graph. This captures semantic relationships that graph structure alone might miss.

**Example**:
```
Keyword: "lighting control"
  appears in requirements: [req_42, req_103, req_157]
  
Requirement incidence vector (before SVD): [1, 0, 1, 0, 1, 0, ...]
  (after SVD reduction to 16 dims): [-0.3, 0.8, -0.1, ..., 0.2]
```

#### Step 3: Feature Fusion

**Strategy**: Combine the two feature types

The final feature vector for each keyword is:
```
Final Features = [Node2Vec_embedding] + [w × SVD_requirement_features]
```

Where:
- Node2Vec block is normalized to zero mean and unit variance
- Requirement block is normalized the same way
- `w` = `--cluster-requirement-feature-weight` (default: 0.5)
  - Controls balance between graph structure and requirement linkage
  - Higher weight = more emphasis on requirement co-occurrence
  - Lower weight = more emphasis on semantic graph structure

**Effect**: Keywords co-mentioned in requirements are pulled together, while still respecting their semantic distance in the graph.

#### Step 4: Select Number of Clusters (k)

**Two modes**:

**Manual Mode**: Specify `--cluster-k` directly
- Use this when you have domain knowledge about how many themes exist
- Value is bounded to [2, number of keywords]

**Automatic Mode** (default): Evaluate different k values
- Test k from `--cluster-k-min` to `--cluster-k-max` (default: 2 to 15)
- For each k, compute:
  - **Inertia** — Within-cluster distance (lower is better)
  - **Davies-Bouldin Index** — Cluster separation (lower is better)
  - **Calinski-Harabasz Index** — Compactness vs separation (higher is better)
- Select k with best combined score (lowest DB + highest CH)
- Returns candidate scores so you can inspect trade-offs

#### Step 5: Apply K-Means Clustering

- Standard K-Means algorithm with 10 random initializations
- Assigns each keyword to one of k clusters
- Produces cluster centers in the feature space

#### Step 6: Visualize with UMAP

- Project high-dimensional features to 2D for visualization
- UMAP preserves local neighborhood structure better than PCA
- Each keyword gets (`umap_x`, `umap_y`) coordinates
- Used for creating interactive visualizations and understanding cluster relationships

### Approach B: Direct Requirement Clustering (Comparison)

**Purpose**: Validate graph-based clustering against a simpler baseline

**How it works**:
1. Embed entire requirement texts using SentenceTransformer
   - Each requirement becomes a single embedding vector
   - Captures the overall semantic content
2. Apply the same K-Means clustering
3. For each keyword, determine its cluster based on the requirements it appears in
   - If keyword appears in requirements [42, 103, 157]
   - And those requirements are in clusters [A, A, B]
   - Majority rule assigns keyword to cluster A

**Why this comparison?**: 
- Direct approach is simpler, avoiding graph construction
- Provides a sanity check: "Are we overcomplicating with the graph?"
- Often produces similar results if both approaches are valid
- Differences reveal structural insights: where do graph relationships add value?

## Comparison Metrics

Phase 3 computes agreement metrics between the two approaches:

| Metric | Meaning | Range |
|--------|---------|-------|
| **Adjusted Rand Index** | How much cluster structure agrees | 0–1 (1 = identical) |
| **Normalized Mutual Info** | Information overlap | 0–1 (1 = identical) |
| **Purity** | Cluster homogeneity | 0–1 (1 = pure) |

**Interpretation**:
- Score > 0.8 — Strong agreement, both methods work similarly
- Score 0.5–0.8 — Partial agreement, interesting differences exist
- Score < 0.5 — Methods diverge significantly, graph structure adds unique value

## Output Artifacts

### 1. Clustering Report (`clustering_report.json`)

Contains:
```json
{
  "clustering": {
    "selected_k": 8,
    "selection_strategy": "auto-combined",
    "candidates": [
      {"k": 2, "davies_bouldin": 1.2, "calinski_harabasz": 234.5},
      ...
    ],
    "node2vec_embedding_dim": 64,
    "requirement_feature_weight": 0.5,
    "requirement_svd_dim": 16,
    "has_requirement_features": true,
    "isolated_node_count": 23
  },
  "cluster_requirements_mapping": {
    "0": ["req_42", "req_103", ...],
    "1": ["req_65", "req_128", ...],
    ...
  }
}
```

### 2. Visualization Plot (`K.png`)

Shows:
- **Top panel**: Inertia vs k (elbow curve)
- **Bottom panel**: Davies-Bouldin vs Calinski-Harabasz trade-off
- Helps analysts understand why a particular k was selected

### 3. Comparison Reports

**Graph-based vs Direct**:
```json
{
  "agreement_metrics": {
    "adjusted_rand_index": 0.67,
    "normalized_mutual_info": 0.71,
    "purity": 0.84
  },
  "graph_clustering": { ... },
  "direct_clustering": { ... }
}
```

### 4. GEXF with Cluster Assignments

The keyword graph is updated with:
- `cluster_id` — Which cluster each keyword belongs to
- `umap_x`, `umap_y` — 2D visualization coordinates
- Ready for import into graph visualization tools

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Combine graph structure + requirement linkage | More robust than either signal alone |
| Use Node2Vec instead of spectral clustering | Captures long-range relationships, not just immediate neighbors |
| Apply SVD to requirement features | Reduces dimensionality while preserving variance |
| Auto-select k from multiple metrics | Single metrics (elbow) can be ambiguous; combined approach is more reliable |
| Include comparison approach | Validates findings; simple vs complex trade-off |
| Preserve 2D coordinates | Enables interactive visualization and qualitative exploration |

## Typical Results

For a moderate-sized requirement graph (500 nodes, 2000 requirements):

| Metric | Typical Range |
|--------|---|
| Selected k | 5–12 clusters |
| Cluster size (nodes) | 20–100 per cluster |
| Graph approach agreement | 0.65–0.85 with direct approach |
| Inertia reduction | ~60–80% when k increases from 2 to selected k |
| UMAP variance captured | ~80–90% of high-D structure in 2D |

## Troubleshooting

**Q: Why do the two approaches disagree?**
- A: Graph structure reveals relationships invisible in requirement embedding space
- The graph captures co-occurrence and semantic similarity explicitly
- Direct approach only sees overall requirement semantics
- Disagreement is informative, not wrong

**Q: Is a lower Davies-Bouldin always better?**
- A: Generally yes, but context matters
- Very low DB (< 0.5) might indicate overclustering (k too high)
- Combined score (DB + CH) is more robust than any single metric

**Q: How to choose between graph and direct clustering for final use?**
- A: Recommendation depends on your use case:
  - **Use graph clustering** if you care about concept relationships and want to understand *why* requirements are grouped
  - **Use direct clustering** if you want a simpler, faster, easier-to-explain approach
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
