# Keyword-Graph: Requirements Engineering Analysis Pipeline

A comprehensive pipeline for extracting keywords from software requirements and performing semantic clustering to discover requirement themes and relationships.

## What This Project Does

This pipeline transforms raw requirement datasets into structured semantic analysis outputs:

- **Phase 1**: Extract keywords from unstructured requirement text (automatic or fine-tuned extraction)
- **Phase 2**: Build a semantic graph showing how concepts relate across requirements
- **Phase 3**: Cluster related keywords into themes and validate results with multiple approaches
- **Phase 4** (planned): Detect inconsistencies and suggest improvements

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Then install PyTorch for your hardware:

```bash
# NVIDIA CUDA on Windows (example)
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

### 2. Run the Pipeline

```bash
python -m src.run_pipeline
```

This runs all phases using the default configuration in `config/run_pipeline.json`.

### 3. Inspect Results

Results are saved to `results/`:
- `keyword_graph.gexf` — Semantic graph for visualization
- `clustering_report.json` — Cluster assignments and metrics
- `clustering_comparison.json` — Comparison with direct approach
- `K.png` — Cluster optimization visualization

## Technical Documentation

Detailed explanations of each phase:

- [Phase 1 - Keyword Extraction](explanation/PHASE_1_MODEL_AND_INFERENCE.md)
- [Phase 2 - Semantic Graph Construction](explanation/PHASE_2_GRAPH_AND_TRACEABILITY.md)
- [Phase 3 - Semantic Clustering and Comparison](explanation/PHASE_3_CLUSTERING_AND_REQUIREMENT_AWARE_FEATURES.md)
- [Overview & Project Status](explanation/note.md)

## Configuration

Configuration is stored in `config/run_pipeline.json` and can be overridden via CLI arguments.

### Command-Line Arguments

Run with custom settings:

```bash
python -m src.run_pipeline --cluster-k 10 --semantic-threshold 0.5
```

The following table lists all configurable options. **CLI arguments take precedence over config file values.**

### Phase 1: Keyword Extraction Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--train` | flag | `False` | Fine-tune the model on provided training data. Recommended when ground-truth keyword tags are available. |
| `--train-input-csv` | `str` | `None` | Path to training CSV (default: `data/crowd.csv`). |
| `--model-dir` | `str` | `models/flan-t5-keywords` | Directory for fine-tuned model artifacts. |
| `--base-model-name` | `str` | `google/flan-t5-base` | Base model for fine-tuning or fallback inference. |
| `--use-finetuned` | flag | `False` | Force use of fine-tuned model instead of KeyBERT default. |
| `--epochs` | `int` | `5` | Fine-tuning epochs. |
| `--batch-size` | `int` | `8` | Training batch size per device. |
| `--lr` | `float` | `3e-4` | Learning rate. |
| `--max-input-len` | `int` | `256` | Max input prompt tokens. |
| `--max-target-len` | `int` | `64` | Max keyword generation tokens. |
| `--fp16` | flag | `False` | Enable mixed-precision training (requires CUDA). |
| `--require-cuda` | flag | `False` | Fail if CUDA unavailable. |
| `--seed` | `int` | `42` | Random seed for reproducibility. |
| `--train-size` | `int` | `2000` | Training set size. |
| `--validation-size` | `int` | `250` | Validation set size. |
| `--test-size` | `int` | `250` | Test set size. |
| `--hf-dataset-dir` | `str` | `data/hf_dataset` | HuggingFace dataset cache directory. |
| `--comparison-json-out` | `str` | `results/real_case_comparison.json` | Output path for extraction quality report. |
| `--inference-batch-size` | `int` | `16` | Batch size during full-dataset keyword extraction. |

### Phase 2: Graph Construction Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--semantic-threshold` | `float` | `0.4` | Minimum cosine similarity for creating graph edges (0.0–1.0). |
| `--semantic-model` | `str` | `all-MiniLM-L6-v2` | Sentence-Transformers model for keyword embedding. |
| `--enable-node-reduction` | flag | `True` | Merge low-degree nodes into hubs to reduce graph complexity. |
| `--disable-node-reduction` | flag | `False` | Keep all nodes unchanged. |
| `--node-reduction-degree-threshold-low` | `int` | `2` | Max degree for a node to be absorbed. |
| `--node-reduction-degree-threshold-high` | `int` | `5` | Min degree to qualify as a hub node. |
| `--graph-out` | `str` | `results/keyword_graph.gexf` | Output graph file path (GEXF format). |

### Phase 3: Clustering Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--enable-clustering` | flag | `True` | Enable clustering phase. |
| `--no-clustering` | flag | `False` | Skip clustering. |
| `--cluster-k` | `int` | `None` | Fixed number of clusters. If `None`, auto-select using Davies-Bouldin + Calinski-Harabasz. |
| `--cluster-k-min` | `int` | `2` | Minimum k for automatic selection. |
| `--cluster-k-max` | `int` | `15` | Maximum k for automatic selection. |
| `--cluster-embedding-dim` | `int` | `64` | Node2Vec embedding dimension. |
| `--cluster-node2vec-walk-length` | `int` | `30` | Length of random walks for Node2Vec. |
| `--cluster-node2vec-num-walks` | `int` | `200` | Number of walks per node. |
| `--cluster-node2vec-workers` | `int` | `4` | Parallel workers for Node2Vec (set to 1 for reproducibility). |
| `--cluster-requirement-feature-weight` | `float` | `0.5` | Weight for requirement-aware features (0.0–1.0). |
| `--cluster-requirement-svd-dim` | `int` | `16` | Dimensionality of reduced requirement features. |
| `--cluster-random-state` | `int` | `42` | Random seed for reproducibility. |
| `--cluster-report-out` | `str` | `results/clustering_report.json` | Clustering output report path. |
| `--cluster-elbow-plot-out` | `str` | `results/k.png` | Elbow plot path. |

### Phase 3: Clustering Comparison Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--enable-clustering-comparison` | flag | `True` | Compare graph-based clustering with direct requirement clustering. |
| `--disable-clustering-comparison` | flag | `False` | Skip comparison. |
| `--clustering-comparison-out` | `str` | `results/clustering_comparison.json` | Comparison report output path. |
| `--direct-clustering-report-out` | `str` | `results/direct_clustering_report.json` | Direct clustering report output path. |

## Understanding the Results

### Output Files

After running the pipeline, the `results/` directory contains:

| File | Purpose | Format |
|------|---------|--------|
| `keyword_graph.gexf` | Semantic graph with all keywords and relationships | GEXF (importable into Gephi, Cytoscape) |
| `clustering_report.json` | Graph-based clustering results with cluster assignments | JSON |
| `clustering_comparison.json` | Comparison metrics between graph and direct approaches | JSON |
| `direct_clustering_report.json` | Direct requirement clustering results (parallel approach) | JSON |
| `K.png` | Visualization of cluster quality metrics by k | PNG |
| `real_case_comparison.json` | Phase 1 extraction quality vs ground-truth (if tags provided) | JSON |

### Interpreting Key Metrics

**Clustering Quality**:
- **Davies-Bouldin Index** (lower is better): Cluster separation; target < 1.5
- **Calinski-Harabasz Index** (higher is better): Compactness vs separation; no fixed target
- **Inertia**: Within-cluster variance; only useful for trend, not absolute value

**Approach Comparison**:
- **Adjusted Rand Index** (0–1): Agreement between two clustering methods; >0.8 = strong agreement
- **Normalized Mutual Information** (0–1): Information overlap; >0.7 = good agreement
- **Purity** (0–1): Cluster homogeneity; >0.85 = high quality

## Configuration File

The `config/run_pipeline.json` file stores default parameters:

```json
{
  "train": false,
  "semantic_threshold": 0.4,
  "cluster_k": null,
  "cluster_k_min": 2,
  "cluster_k_max": 15,
  "cluster_requirement_feature_weight": 0.5,
  "enable_clustering": true,
  "enable_clustering_comparison": true
}
```

**Priority order for parameters**:
1. CLI arguments (highest priority) — `python -m src.run_pipeline --cluster-k 10`
2. Config file values — loaded from `config/run_pipeline.json`
3. Built-in defaults (lowest priority)

This design lets you keep stable defaults in JSON and override specific values via CLI.

## Common Workflows

### Workflow 1: Extract Keywords Without Fine-Tuning

```bash
# Uses KeyBERT by default
python -m src.run_pipeline --no-clustering
```

Outputs: `keyword_graph.gexf` only (Phases 1–2)

### Workflow 2: Fine-Tune on Labeled Data

```bash
# Trains on data/crowd.csv, extracts keywords, builds graph
python -m src.run_pipeline --train --epochs 5
```

Requires: `tags` column in input CSV

### Workflow 3: Fixed k Clustering

```bash
# Use exactly 8 clusters instead of auto-selecting
python -m src.run_pipeline --cluster-k 8
```

### Workflow 4: Experiment with Feature Weight

```bash
# Emphasize graph structure over requirement linkage
python -m src.run_pipeline --cluster-requirement-feature-weight 0.2

# Emphasize requirement linkage over graph structure
python -m src.run_pipeline --cluster-requirement-feature-weight 0.8
```

## Best Practices

1. **Start with defaults** — Run `python -m src.run_pipeline` first to understand your data
2. **Review K.png** — Inspect the elbow plot to validate automatic k selection
3. **Check comparison metrics** — High agreement (ARI >0.8) between approaches validates results
4. **Adjust semantic-threshold if needed**:
   - Lower (0.2–0.3) for more lenient keyword relationships
   - Higher (0.5–0.6) for stricter, more specific relationships
5. **Use node reduction** — Enabled by default; reduces noise without losing traceability
6. **Fine-tune only if** — You have ground-truth keyword tags and want to optimize extraction accuracy

## Visualization with Gephi

1. Open `results/keyword_graph.gexf` in Gephi (free software)
2. Use the built-in layout algorithms (ForceAtlas2 recommended)
3. Color nodes by `relation_color` attribute for quick visual analysis
4. Use `cluster_id` to highlight clusters







