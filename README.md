# Keyword-Graph

A step-by-step pipeline for keyword extraction from requirements and graph-based analysis.

## Technical Explanations

Detailed technical documentation for the implemented pipeline phases is available in:

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
| `--cluster-k` | `int` | `None` | Fixed number of clusters. If omitted, `k` is selected automatically using Davies-Bouldin and Calinski-Harabasz scores. |
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
| `--cluster-elbow-plot-out` | `str` | `results/k.png` | Output plot path for inertia and cluster quality scores by candidate `k`. |
| `--enable-clustering-comparison` | flag | `True` | Enable clustering comparison: evaluate keyword-graph clustering vs direct requirement embedding clustering. |
| `--disable-clustering-comparison` | flag | `False` | Disable clustering comparison. |
| `--clustering-comparison-out` | `str` | `results/clustering_comparison.json` | Output JSON path for clustering comparison diagnostics and cross-approach agreement metrics. |
| `--direct-clustering-report-out` | `str` | `results/direct_clustering_report.json` | Output JSON path for the direct requirement clustering report. |
| `--use-finetuned` | flag | `False` | Force the pipeline to use the fine-tuned model for inference even when KeyBERT is available. |

### Notes on training behavior

- Fine-tuning is optional and runs only when `--train` is provided.
- Without `--train`, the pipeline tries to load the model from `--model-dir`; if unavailable, it uses the base model.
- Training data path and split sizes are fully controllable via CLI/config.

### KeyBERT default extractor

- The pipeline now prefers KeyBERT for keyword extraction by default when KeyBERT is installed in the environment.
- To force using the fine-tuned model instead, pass `--use-finetuned` on the command line.
- If `--use-finetuned` is provided but no checkpoint exists in `--model-dir`, the pipeline will warn and fall back to KeyBERT or the base model.

Ensure KeyBERT is installed if you want the default extractor behavior: `pip install keybert` (already suggested in `requirements.txt`).

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
- Nodes with degree `0` are removed before clustering.
- Node reduction merges low-degree nodes into hubs when `--enable-node-reduction` is active.

## Clustering Comparison (Phase 3+)

**Purpose**: Evaluate whether the keyword-graph clustering approach produces better results than direct requirement clustering.

**How it works**:
1. Runs the keyword-graph clustering pipeline (Phase 3)
2. In parallel, clusters requirements directly using their text embeddings (without keywords)
3. Compares cluster assignments between the two approaches
4. Generates a JSON report with agreement metrics

**Outputs**:
- Keyword-graph clustering report: `results/clustering_report.json`
- Direct clustering report with pipeline-like schema: `results/direct_clustering_report.json`
- Comparison report: `results/clustering_comparison.json`

**Keyword-graph report shape**:
```json
{
  "clustering": {
    "enabled": true,
    "selected_k": 12,
    "num_nodes": 1477,
    "embedding_dim": 64,
    "effective_feature_dim": 80,
    "inertia": 81390.92,
    "k_selection": {
      "selected_k": 12,
      "strategy": "auto-combined",
      "candidates": []
    },
    "elbow_plot": null,
    "requirement_features": {
      "enabled": true,
      "num_requirements": 2965,
      "raw_dim": 2965,
      "reduced_dim": 16,
      "weight": 0.5
    }
  },
  "cluster_requirements_mapping": {
    "0": ["1001", "1008"],
    "1": ["1002", "1010"]
  },
  "requirement_to_cluster_mapping": {
    "1001": [0, 2],
    "1002": [1]
  },
  "nodes": [
    {
      "keyword": "temperature",
      "cluster_id": 0,
      "umap_x": 0.12,
      "umap_y": -0.34
    }
  ]
}
```

The direct requirement report in `results/direct_clustering_report.json` uses the same top-level shape, but its `clustering` section includes `mode: "direct_requirement_clustering"` and its `nodes` list stores `requirement_id` and `input` instead of graph keywords.

## Direct Clustering Report

The direct clustering report mirrors the keyword-graph report structure, but it is built from requirement embeddings instead of keyword graph nodes.

**Purpose**:
- Keep the same top-level JSON shape for both approaches
- Make quantitative comparison and inspection easier
- Preserve per-requirement assignments for direct clustering

**Configuration**:
- Output path: `--direct-clustering-report-out`

**Schema notes**:
- `cluster_requirements_mapping` maps cluster IDs to unique requirement IDs
- `requirement_to_cluster_mapping` maps each requirement to its direct cluster as a one-item list
- `nodes` contains `requirement_id`, `input`, `cluster_id`, `umap_x`, and `umap_y`

The comparison report in `results/clustering_comparison.json` is still generated alongside it.






