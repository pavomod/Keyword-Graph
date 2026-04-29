# Keyword-Graph: requirements analysis pipeline

Keyword-Graph is an end-to-end pipeline that turns natural-language software requirements into normalized keywords, a semantic graph, thematic clusters, and comparison reports. The project is designed to support Requirements Engineering analysis, traceability, and semantic interpretation of requirements.

## Goal

The core idea is straightforward: start from a CSV of textual requirements, extract the important concepts, connect them in a graph, group them into coherent themes, and compare multiple clustering strategies to understand how stable the results are.

The full pipeline covers these stages:

1. Keyword extraction from requirements
2. Semantic graph construction with traceability back to the original requirements
3. Clustering of keyword nodes with hybrid features
4. Comparison against a direct approach based on the full requirement text
5. Export of readable and visualizable artifacts

Phase 4 in the documentation is only planned: the current implementation goes as far as cluster comparison and validation.

## How the pipeline works

### High-level data flow

```text
Requirements CSV
  -> preprocessing and normalization
  -> keyword extraction
  -> keyword-to-keyword semantic graph
  -> node reduction and graph cleanup
  -> hybrid clustering with Node2Vec + requirement features
  -> comparison with direct requirement clustering
  -> JSON + GEXF + plot reports
```

The entrypoint is [src/run_pipeline.py](src/run_pipeline.py). The main command is `python -m src.run_pipeline`, which reads configuration from [config/run_pipeline.json](config/run_pipeline.json) and can be overridden from the CLI.

## Phase 1 - Keyword extraction

The first phase converts each requirement into a clean, coherent list of keywords. This is the step that moves the text from narrative form to analyzable form.

### Methodology

The project supports two modes:

1. KeyBERT as the default, lightweight, ready-to-use approach.
2. FLAN-T5 fine-tuning when reference tags are available in the dataset.

### Implementation choices

- Pragmatic default: use KeyBERT when nothing is trained, so the pipeline remains immediately runnable.
- Optional training: with `--train`, the system prepares train/validation/test splits, trains the model, and saves artifacts in `models/flan-t5-keywords/`.
- Robust fallback: if the fine-tuned model is not available, the pipeline falls back to the default extractor.
- Normalization: keywords are standardized before moving to the next phase.

### Input and output

Expected input: `data/crowd.csv`, with columns such as `role`, `feature`, `benefit`, and optionally `tags` for evaluation.

Main outputs:

- extracted keywords for each requirement
- comparison report against ground-truth tags, when available
- HuggingFace dataset saved to `data/hf_dataset/` when training is enabled

### Why this choice

The dual-mode approach balances quality and operational simplicity. Users who want immediate results can use KeyBERT; users with labeled data can improve extraction with a model adapted to the domain.

## Phase 2 - Semantic graph and traceability

In this phase, keywords become nodes in a semantic graph. The goal is not only to connect similar concepts, but also to keep the link to the original requirements at all times.

### Methodology

The graph combines two complementary signals:

1. Co-occurrence: keywords that appear in the same requirement.
2. Semantic similarity: keywords close in embedding space, computed with a Sentence-Transformers model.

The default model is `all-MiniLM-L6-v2`. An edge is created only if similarity exceeds the threshold defined by `--semantic-threshold`.

### Implementation choices

- Combining co-occurrence and semantic similarity avoids relying on a single signal.
- Every node keeps the IDs of the requirements that support it, so traceability stays complete.
- Isolated nodes are removed because they do not add useful relational structure.
- Node reduction is enabled by default: low-information keywords can be absorbed into stronger hubs while preserving requirement evidence.

### Result

The graph is exported as GEXF in [results/keyword_graph.gexf](results/keyword_graph.gexf), so it can be opened in Gephi or Cytoscape for visual analysis.

## Phase 3 - Clustering and comparison

The third phase groups keywords into semantic themes and checks the stability of the result with an alternative approach.

### Main methodology

Clustering does not rely only on graph structure. It combines two feature families:

1. Structural graph embeddings with Node2Vec.
2. Requirement-aware features, meaning vectors that represent which requirements each keyword appears in.

The two components are fused with a configurable weight via `--cluster-requirement-feature-weight`.

### Why this choice

This is the most important point in the pipeline: clustering should not be purely topological, it should also reflect the requirement context. In practice, keywords that are close in the graph and often appear in the same requirements are more likely to end up in the same cluster.

### Selecting k

The number of clusters can be:

- fixed manually with `--cluster-k`
- selected automatically by testing a range of values and comparing multiple metrics

The metrics used are:

- Davies-Bouldin Index, which should be minimized
- Calinski-Harabasz Index, which should be maximized
- inertia, useful for trend reading but not as an absolute criterion

### Visualization

After clustering, nodes are projected into 2D with UMAP to make reading and visual inspection easier.

### Direct baseline comparison

The pipeline also computes a direct clustering of the requirements, without using the graph, to answer a very concrete methodological question: does the graph add value or not?

The comparison produces metrics such as:

- Adjusted Rand Index
- Normalized Mutual Information
- Purity

If the metrics are high, the two approaches are coherent; if they diverge, the graph is likely capturing additional informative structure.

## Phase 4 - Current status and future work

In the project documentation, phase 4 is marked as planned. The idea is to use the clustering results to identify inconsistencies, ambiguities, or potentially conflicting requirements and provide support for correction.

## Project structure

```text
README.md
config/run_pipeline.json
data/
  crowd.csv
  hf_dataset/
explanation/
  PHASE_1_MODEL_AND_INFERENCE.md
  PHASE_2_GRAPH_AND_TRACEABILITY.md
  PHASE_3_CLUSTERING_AND_REQUIREMENT_AWARE_FEATURES.md
results/
src/
  run_pipeline.py
  phase1/
  phase2/
  phase3/
  phase4/
```

## Installation

```bash
pip install -r requirements.txt
```

On Windows, if you want to use an NVIDIA GPU with PyTorch, you may need to manually reinstall the correct torch build for your CUDA version. A typical example is:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

## Running the pipeline

### Standard run

```bash
python -m src.run_pipeline
```

This run:

- uses the default configuration
- extracts keywords from `data/crowd.csv`
- builds the semantic graph
- runs clustering, if enabled
- saves outputs to `results/`

### Run with custom parameters

```bash
python -m src.run_pipeline --cluster-k 8 --semantic-threshold 0.5
```

CLI parameters take precedence over the JSON file.

### Training the extraction model

```bash
python -m src.run_pipeline --train --epochs 5
```

This mode is useful when the CSV contains reference tags and you want to adapt the model to the specific requirement domain.

### Disabling clustering

```bash
python -m src.run_pipeline --no-clustering
```

Useful if you want to stop after semantic graph construction.

## Main parameters

### Keyword extraction

- `--train`: enables fine-tuning
- `--use-finetuned`: forces the use of the already trained model
- `--train-input-csv`: training CSV
- `--model-dir`: fine-tuned model directory
- `--base-model-name`: base model for training or fallback

### Semantic graph

- `--semantic-threshold`: minimum cosine similarity required to create an edge
- `--semantic-model`: model used for keyword embeddings
- `--enable-node-reduction` / `--disable-node-reduction`: enable or disable node reduction

### Clustering

- `--cluster-k`: fixed number of clusters
- `--cluster-k-min`, `--cluster-k-max`: range for automatic selection
- `--cluster-embedding-dim`: Node2Vec embedding dimension
- `--cluster-requirement-feature-weight`: weight of requirement-aware features
- `--cluster-requirement-svd-dim`: SVD reduction dimension

### Comparison

- `--enable-clustering-comparison` / `--disable-clustering-comparison`: enable comparison with direct clustering
- `--clustering-comparison-out`: comparison report output
- `--direct-clustering-report-out`: direct baseline report output

## Configuration

Defaults are defined in [config/run_pipeline.json](config/run_pipeline.json). Value precedence is:

1. CLI parameters
2. configuration file values
3. hardcoded defaults in the code

This keeps the pipeline reproducible while still making experimentation easy.

## Generated outputs

In the `results/` folder you will typically find:

- [results/keyword_graph.gexf](results/keyword_graph.gexf): exportable semantic graph
- [results/clustering_report.json](results/clustering_report.json): graph-based clustering report
- [results/clustering_comparison.json](results/clustering_comparison.json): comparison between the two approaches
- [results/direct_clustering_report.json](results/direct_clustering_report.json): direct clustering report
- [results/K.png](results/K.png): visual support for selecting k
- [results/real_case_comparison.json](results/real_case_comparison.json): keyword extraction comparison against ground-truth tags, when present

## How to read the results

### Clustering quality

- Lower Davies-Bouldin means better-separated clusters
- Higher Calinski-Harabasz means more compact and better-separated clusters
- inertia is useful only for understanding the trend as k changes

### Approach comparison

- Higher ARI means strong agreement between graph-based and direct clustering
- Higher NMI means strong information overlap
- Higher Purity means more homogeneous clusters

## When to use each mode

Use the standard pipeline if you want a complete and fast reading of the requirements. Enable training only if you have a labeled dataset and want to improve extraction quality. Fix `k` manually when running controlled experiments; leave automatic selection on if you want a data-driven estimate.

## Supporting documentation

- [Phase 1 - Keyword extraction](explanation/PHASE_1_MODEL_AND_INFERENCE.md)
- [Phase 2 - Semantic graph and traceability](explanation/PHASE_2_GRAPH_AND_TRACEABILITY.md)
- [Phase 3 - Clustering and requirement-aware features](explanation/PHASE_3_CLUSTERING_AND_REQUIREMENT_AWARE_FEATURES.md)
- [Project overview note](explanation/note.md)

## Final summary

Keyword-Graph combines NLP, graph analysis, and clustering to turn raw requirements into analyzable structure. The key design choice is to never lose traceability: every keyword, edge, and cluster remains connected to the original requirements, so the results are not only interesting, but also verifiable.







