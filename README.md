# Keyword-Graph

A step-by-step pipeline for keyword extraction from requirements and graph-based analysis.

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


### Recommended startup commands by scenario

1. Default run (no training). Uses the fine-tuned model if available, otherwise falls back to base `google/flan-t5-base`.

```bash
python -m src.run_pipeline
```

2. Run with explicit training (fine-tuning is optional and runs only when requested)

```bash
python -m src.run_pipeline --train
```

3. Explicit skip training

```bash
python -m src.run_pipeline --no-train
```

4. GPU training with mixed precision

```bash
python -m src.run_pipeline --train --fp16 --require-cuda
```

5. Select a custom CSV for training (only for fine-tune stage)

```bash
python -m src.run_pipeline --train --train-input-csv data/my_train.csv
```

6. Stronger semantic filtering in graph edges (fewer, stronger relations)

```bash
python -m src.run_pipeline --semantic-threshold 0.8
```

7. Custom output paths for report and graph

```bash
python -m src.run_pipeline --comparison-json-out results/my_report.json --graph-out results/my_graph.gexf
```

8. Use an alternative config file

```bash
python -m src.run_pipeline --config config/my_experiment.json
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
| `--graph-out` | `str` | `results/keyword_graph_real_case.gexf` | Output GEXF graph file generated from predicted keywords. |
| `--inference-batch-size` | `int` | `16` | Batch size used during full-dataset inference/prediction phase. |
| `--semantic-threshold` | `float` | `0.4` | Minimum cosine similarity required to create an edge between two keywords in the semantic graph. |
| `--semantic-model` | `str` | `all-MiniLM-L6-v2` | Sentence-Transformers model used to compute keyword semantic similarity. |

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
	- `results/keyword_graph_real_case.gexf`

Graph logic:

- The graph compares all extracted keywords globally (not only inside the same user story).
- An edge is created only when semantic similarity is greater than or equal to `--semantic-threshold`.
- Edge weight is the cosine semantic similarity score.


