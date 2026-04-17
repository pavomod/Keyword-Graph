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

Quick verification:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

## Pipeline execution (`src.run_pipeline`)

Start command:

```bash
python -m src.run_pipeline [arguments...]
```

### Recommended startup commands by scenario

1. First run on a new machine/project (train if needed, then evaluate real-case and build graph)

```bash
python -m src.run_pipeline --mode auto
```

2. Force a brand-new training run (ignore existing model)

```bash
python -m src.run_pipeline --mode finetune
```

3. Reuse only an existing model (no training), useful for quick graph regeneration

```bash
python -m src.run_pipeline --mode reuse --model-dir models/flan-t5-keywords
```

4. GPU training with mixed precision

```bash
python -m src.run_pipeline --mode finetune --fp16 --require-cuda
```

5. CPU-safe run (explicit full precision)

```bash
python -m src.run_pipeline --mode auto --no-fp16
```

6. Stronger semantic filtering in graph edges (fewer, stronger relations)

```bash
python -m src.run_pipeline --mode reuse --semantic-threshold 0.8
```

7. Custom output paths for report and graph

```bash
python -m src.run_pipeline --mode auto --comparison-json-out results/my_report.json --graph-out results/my_graph.gexf
```

### Full argument reference

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `--mode` | `auto\|finetune\|reuse` | `auto` | Pipeline mode: `auto` trains only if model is missing, `finetune` always trains, `reuse` requires an existing model and skips training. |
| `--input-csv` | `str` | `None` | Input dataset CSV path. If omitted, the pipeline auto-detects `data/crowd.csv` (or `requirements.csv`). |
| `--hf-dataset-dir` | `str` | `data/hf_dataset` | Output folder for HuggingFace `DatasetDict` used by training. |
| `--model-dir` | `str` | `models/flan-t5-keywords` | Folder where the fine-tuned model is saved/loaded. |
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
| `--comparison-json-out` | `str` | `results/real_case_comparison.json` | Output JSON file with real-case prediction vs ground truth and aggregate metrics. |
| `--graph-out` | `str` | `results/keyword_graph_real_case.gexf` | Output GEXF graph file generated from predicted real-case keywords. |
| `--inference-batch-size` | `int` | `16` | Batch size used during real-case inference/prediction phase. |
| `--semantic-threshold` | `float` | `0.7` | Minimum cosine similarity required to create an edge between two keywords in the semantic graph. |
| `--semantic-model` | `str` | `all-MiniLM-L6-v2` | Sentence-Transformers model used to compute keyword semantic similarity. |

### Notes on mode behavior

- `auto` is the best default for daily usage.
- `finetune` is the right choice when you changed data prep, split sizes, or training hyperparameters.
- `reuse` is the fastest option when you only want fresh predictions/report/graph with an already trained model.

## Real-case split and outputs

The pipeline now applies a fixed split:

- train: 2000 rows
- validation: 250 rows
- test: 250 rows
- real case: all remaining rows

For the real-case split, the model predicts keywords from the user story input only.
Ground-truth tags are used only to compute evaluation metrics and produce a comparison report.

Outputs:

- JSON report with metrics and per-row predicted vs real tags:
	- `results/real_case_comparison.json`
- Graph generated from predicted real-case tags:
	- `results/keyword_graph_real_case.gexf`

Graph logic:

- The graph compares all extracted keywords globally (not only inside the same user story).
- An edge is created only when semantic similarity is greater than or equal to `--semantic-threshold`.
- Edge weight is the cosine semantic similarity score.


