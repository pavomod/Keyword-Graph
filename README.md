# Keyword-Graph

A step-by-step pipeline for keyword extraction from requirements and graph-based analysis.

## Technical Explanations

Detailed technical documentation for each implemented phase is available in:

- [Phase 1 - Model and Inference](explanation/PHASE_1_MODEL_AND_INFERENCE.md)
- [Phase 2 - Graph and Traceability](explanation/PHASE_2_GRAPH_AND_TRACEABILITY.md)
- [Phase 3 - Clustering and Requirement-Aware Features](explanation/PHASE_3_CLUSTERING_AND_REQUIREMENT_AWARE_FEATURES.md)

## Dependency installation

Install project dependencies first:

```bash
pip install -r requirements.txt
```

Then install PyTorch explicitly based on your hardware.

For NVIDIA CUDA on Windows:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```


## Pipeline execution (`src.run_pipeline`)

Start command:

```bash
python -m src.run_pipeline [arguments...]
```

The pipeline automatically reads a JSON config file from `config/run_pipeline.json`.
CLI arguments always override values from the config file.


### Quick start (JSON-based)

Use the default JSON configuration in `config/run_pipeline.json`:

```bash
python -m src.run_pipeline
```

### Full argument reference

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `--config` | `str` | `config/run_pipeline.json` | JSON config file path. If present, values are loaded before execution. |
| `--train` | flag | `False` | Enable fine-tuning stage. If omitted, training is skipped. |
| `--no-train` | flag | `True` (effective default) | Explicitly skip fine-tuning stage. |
| `--train-input-csv` | `str` | `None` | Input dataset CSV path used only for training split preparation. |
| `--hf-dataset-dir` | `str` | `data/hf_dataset` | Output folder for HuggingFace `DatasetDict` used by training. |
| `--model-dir` | `str` | `models/flan-t5-keywords` | Folder where the fine-tuned model is saved/loaded. |
| `--base-model-name` | `str` | `google/flan-t5-base` | Base model used when no fine-tuned model is available. |
| `--seed` | `int` | `42` | Random seed for deterministic shuffling and split generation. |
| `--epochs` | `int` | `5` | Number of fine-tuning epochs. |
| `--batch-size` | `int` | `8` | Train/eval batch size per device. |
| `--lr` | `float` | `3e-4` | Learning rate for optimizer. |
| `--max-input-len` | `int` | `256` | Max token length for input prompt tokenization. |
| `--max-target-len` | `int` | `64` | Max generation length for target/predicted keywords. |
| `--fp16` | flag | `False` | Enable mixed precision training. Use on CUDA-capable GPU. |
| `--no-fp16` | flag | `False` | Force full precision training (disables fp16). |
| `--require-cuda` | flag | `False` | Fail fast if CUDA is not available. Useful to avoid accidental CPU training. |
| `--train-size` | `int` | `2000` | Fixed number of rows used for training split. |
| `--validation-size` | `int` | `250` | Fixed number of rows used for validation split. |
| `--test-size` | `int` | `250` | Fixed number of rows used for test split. |
| `--comparison-json-out` | `str` | `results/real_case_comparison.json` | Output JSON file with prediction vs ground truth and aggregate metrics over rows that contain `tags`. |
| `--graph-out` | `str` | `results/keyword_graph.gexf` | Output GEXF graph file generated from predicted keywords. |
| `--inference-batch-size` | `int` | `16` | Batch size used during full-dataset inference/prediction phase. |
| `--semantic-threshold` | `float` | `0.4` | Minimum cosine similarity required to create an edge between two keywords in the semantic graph. |
| `--semantic-model` | `str` | `all-MiniLM-L6-v2` | Sentence-Transformers model used to compute keyword semantic similarity. |
| `--enable-node-reduction` | flag | `True` | Enable node reduction: absorb low-degree nodes into high-degree hub nodes. |
| `--disable-node-reduction` | flag | `False` | Disable node reduction. |
| `--node-reduction-degree-threshold-low` | `int` | `2` | Maximum degree for a node to be considered "low" and eligible for absorption. |
| `--node-reduction-degree-threshold-high` | `int` | `5` | Minimum degree for a node to be considered a "hub" node. |
| `--enable-clustering` | flag | `True` | Enable Phase 3 clustering (Node2Vec + K-Means + UMAP). |
| `--no-clustering` | flag | `False` | Disable Phase 3 clustering. |
| `--cluster-k` | `int` | `None` | Fixed number of clusters. If omitted, `k` is selected automatically by silhouette score. |
| `--cluster-k-min` | `int` | `2` | Lower bound for automatic `k` search. |
| `--cluster-k-max` | `int` | `15` | Upper bound for automatic `k` search. |
| `--cluster-embedding-dim` | `int` | `64` | Node2Vec embedding dimension before feature fusion. |
| `--cluster-random-state` | `int` | `42` | Random seed for Node2Vec, K-Means, and UMAP reproducibility. |
| `--cluster-node2vec-walk-length` | `int` | `30` | Random walk length for Node2Vec. |
| `--cluster-node2vec-num-walks` | `int` | `200` | Number of random walks per node for Node2Vec. |
| `--cluster-node2vec-workers` | `int` | `4` | Number of worker threads for Node2Vec. |
| `--cluster-requirement-feature-weight` | `float` | `0.5` | Weight applied to requirement-aware feature block before concatenation with Node2Vec vectors. |
| `--cluster-requirement-svd-dim` | `int` | `16` | SVD-reduced dimensionality of requirement incidence features. |
| `--cluster-report-out` | `str` | `results/clustering_report.json` | Output JSON report path for clustering diagnostics and per-node cluster assignments. |
| `--cluster-elbow-plot-out` | `str` | `results/elbow.png` | Output plot path for inertia and silhouette by candidate `k`. |
| `--enable-clustering-comparison` | flag | `True` | Enable clustering comparison: evaluate keyword-graph clustering vs direct requirement embedding clustering. |
| `--disable-clustering-comparison` | flag | `False` | Disable clustering comparison. |
| `--clustering-comparison-out` | `str` | `results/clustering_comparison.json` | Output JSON path for clustering comparison diagnostics and cross-approach agreement metrics. |

### Notes on training behavior

- Fine-tuning is optional and runs only when `--train` is provided.
- Without `--train`, the pipeline tries to load the model from `--model-dir`; if unavailable, it uses the base model.
- Training data path and split sizes are fully controllable via CLI/config.

### Config precedence

Order used by the script for each parameter:

1. CLI argument (highest priority)
2. JSON config value from `--config`
3. Internal default (lowest priority)

This lets you keep stable experiment defaults in JSON and override only a few values from CLI when needed.

## Dataset usage and outputs

Training stage (optional):

- Applies a fixed split on the training CSV:
	- train: 2000 rows
	- validation: 250 rows
	- test: 250 rows
	- held-out: all remaining rows

Inference and graph stage (always executed):

- Always reads the full `data/crowd.csv`.
- Graph generation ignores `tags` as input features.
- `tags` are used only for verification report when available.
- Predicted keywords are normalized to singular, lowercase, deduplicated and lexicographically sorted.

Outputs:

- JSON report with metrics and per-row predicted vs real tags:
	- `results/real_case_comparison.json`
- Graph generated from predicted full-dataset keywords:
	- `results/keyword_graph.gexf`

Graph logic:

- The graph compares all extracted keywords globally (not only inside the same user story).
- An edge is created only when semantic similarity is greater than or equal to `--semantic-threshold`.
- Edge weight is the cosine semantic similarity score.

## Clustering Comparison (Phase 3+)

**Purpose**: Evaluate whether the keyword-graph clustering approach produces better results than direct requirement clustering.

**How it works**:
1. Runs the keyword-graph clustering pipeline (Phase 3)
2. In parallel, clusters requirements directly using their text embeddings (without keywords)
3. Compares cluster assignments between the two approaches
4. Generates a JSON report with agreement metrics

**Outputs**:
- JSON comparison report: `results/clustering_comparison.json`
- Direct clustering report with pipeline-like schema: `results/direct_clustering_report.json`

**Report structure**:
```json
{
  "clustering": {
    "enabled": true,
    "mode": "direct_requirement_clustering",
    "selected_k": 12,
    "num_nodes": 5000,
    "embedding_dim": 384,
    "effective_feature_dim": 384,
    "inertia": 45000.5,
    "silhouette": 0.42,
    "k_selection": {
      "selected_k": 12,
      "strategy": "auto-silhouette",
      "candidates": []
    },
    "elbow_plot": null,
    "requirement_features": {
      "enabled": false,
      "reason": "direct_requirement_clustering",
      "num_requirements": 5000,
      "raw_dim": 384,
      "reduced_dim": 384,
      "weight": 0.0
    }
  },
  "cluster_requirements_mapping": {
    "0": ["1001", "1008"],
    "1": ["1002", "1010"]
  },
  "requirement_to_cluster_mapping": {
    "1001": [0],
    "1002": [1]
  },
  "cluster_labels": {
    "0": "user access control",
    "1": "notification scheduling"
  },
  "nodes": [
    {
      "requirement_id": "1001",
      "input": "As a ...",
      "cluster_id": 0,
      "umap_x": 0.12,
      "umap_y": -0.34
    }
  ]
}
```

The same structure is used for the keyword-graph clustering report in `results/clustering_report.json`, so you can compare the two outputs directly.

## Direct Clustering Report

The direct clustering report mirrors the keyword-graph report structure, but it is built from requirement embeddings instead of keyword graph nodes.

**Purpose**:
- Keep the same JSON shape for both approaches
- Make quantitative comparison and inspection easier
- Preserve cluster labels and per-node assignments for direct clustering

**Configuration**:
- Output path: `--direct-clustering-report-out`

**Schema notes**:
- `cluster_requirements_mapping` maps cluster IDs to unique requirement IDs
- `requirement_to_cluster_mapping` maps each requirement to its direct cluster as a one-item list
- `cluster_labels` are generated with FLAN-T5 in the same way as the pipeline report

The comparison report in `results/clustering_comparison.json` is still generated alongside it.
```

**Interpretation**:
- **High agreement ratio** (>0.75): Both approaches produce similar clusters, suggesting robust clustering structure
- **Low agreement ratio** (<0.6): Keyword graph captures different semantic structures; may indicate:
  - Keywords provide valuable semantic compression
  - Direct embedding misses intermediate abstractions that keywords capture
- **Cluster count difference**: Different k values can indicate different granularity levels
  - Keyword graph typically produces fewer clusters (nodes as abstractions)
  - Direct embedding may create more fine-grained clusters

**Configuration**:
- Enable with `--enable-clustering-comparison` (default: enabled when clustering is enabled)
- Disable with `--disable-clustering-comparison`
- Output path: `--clustering-comparison-out`

## Cluster Labeling (Phase 3)

**Purpose**: Automatically generate human-readable, descriptive labels for each cluster using FLAN-T5.

**How it works**:
1. For each cluster, the top 20 keywords are collected
2. A prompt is created: *"Generate a brief, descriptive label (2-5 words) for a cluster of requirements represented by these keywords: [keywords]. The cluster contains [N] requirements. Label:"*
3. FLAN-T5 generates a concise label capturing the cluster's semantic essence
4. Labels are cleaned and stored in the clustering report

**Example labels**:
```
Cluster 0: "user authentication"
Cluster 1: "notification scheduling" 
Cluster 2: "data export functionality"
Cluster 3: "permission management"
```

**Output**:
- Labels are included in the clustering report under `cluster_labels`
- Format: `{ "0": "user authentication", "1": "notification scheduling", ... }`
- Each label is 2-5 words, lowercase, free of punctuation

**Configuration**:
- Automatically enabled when clustering is enabled
- Uses the same model (fine-tuned or base FLAN-T5) as Phase 1 keyword extraction
- Ensures semantic consistency across the entire pipeline

**Benefits**:
- Provides human-interpretable summaries of discovered clusters
- Enables quick understanding of requirement groupings
- Facilitates communication with non-technical stakeholders
- Uses the same semantic understanding as keyword extraction for consistency




