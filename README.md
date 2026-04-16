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

## One-command execution

Run the model pipeline with a single command:

```bash
python -m src.run_pipeline --mode auto
```

Available modes:

- `auto`: reuse a saved model if it exists; otherwise prepare data and fine-tune.
- `finetune`: always run a new fine-tuning job.
- `reuse`: only use an existing saved model (fails if the model does not exist).

Examples:

```bash
python -m src.run_pipeline --mode finetune --epochs 3 --batch-size 8
python -m src.run_pipeline --mode reuse --model-dir models/flan-t5-keywords
```

Useful parameters:

- `--input-csv requirements.csv`
- `--hf-dataset-dir data/hf_dataset`
- `--model-dir models/flan-t5-keywords`
- `--fp16` or `--no-fp16`


