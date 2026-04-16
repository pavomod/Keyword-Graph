from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.finetune.prepare_data import (
    _pick_input_csv,
    prepare_dataframe,
    split_dataset,
    to_hf_dataset_dict,
)
from src.finetune.train import train_model


def _model_exists(model_dir: Path) -> bool:
    required_any = [
        model_dir / "adapter_config.json",
        model_dir / "pytorch_model.bin",
        model_dir / "model.safetensors",
    ]
    tokenizer_required = model_dir / "tokenizer_config.json"
    return model_dir.exists() and tokenizer_required.exists() and any(p.exists() for p in required_any)


def _prepare_dataset(input_csv: str | None, hf_out_dir: str, seed: int) -> dict:
    input_path = _pick_input_csv(input_csv)
    raw_df = pd.read_csv(input_path)

    formatted_df = prepare_dataframe(raw_df)
    train_df, val_df, test_df = split_dataset(formatted_df, seed=seed)

    hf_ds = to_hf_dataset_dict(train_df, val_df, test_df)
    hf_ds.save_to_disk(hf_out_dir)

    return {
        "input": str(input_path),
        "samples": len(formatted_df),
        "train": len(train_df),
        "validation": len(val_df),
        "test": len(test_df),
        "dataset_dir": hf_out_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One-command pipeline for keyword model. "
            "Use --mode finetune to always train, --mode reuse to only use saved model, "
            "or --mode auto to train only if model is missing."
        )
    )
    parser.add_argument("--mode", choices=["auto", "finetune", "reuse"], default="auto")
    parser.add_argument("--input-csv", type=str, default=None)
    parser.add_argument("--hf-dataset-dir", type=str, default="data/hf_dataset")
    parser.add_argument("--model-dir", type=str, default="models/flan-t5-keywords")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-input-len", type=int, default=256)
    parser.add_argument("--max-target-len", type=int, default=64)
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--no-fp16", action="store_true", help="Force full precision training")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    if args.fp16 and args.no_fp16:
        raise ValueError("Use only one of --fp16 or --no-fp16")

    requested_fp16 = True if args.fp16 else False if args.no_fp16 else None

    model_dir = Path(args.model_dir)
    has_model = _model_exists(model_dir)

    if args.mode == "reuse" and not has_model:
        raise FileNotFoundError(
            f"Model not found in {model_dir}. Use --mode finetune or --mode auto first."
        )

    should_train = args.mode == "finetune" or (args.mode == "auto" and not has_model)

    if should_train:
        print("[1/2] Preparing HuggingFace dataset...")
        prep_info = _prepare_dataset(args.input_csv, args.hf_dataset_dir, args.seed)
        print(
            "Prepared dataset from {input} | samples={samples} | split={train}/{validation}/{test}".format(
                **prep_info
            )
        )

        print("[2/2] Fine-tuning model...")
        metrics = train_model(
            dataset_dir=args.hf_dataset_dir,
            output_dir=args.model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_input_len=args.max_input_len,
            max_target_len=args.max_target_len,
            use_fp16=requested_fp16,
            require_cuda=args.require_cuda,
        )
        print(f"Training complete. Model saved in: {args.model_dir}")
        print(f"Eval metrics: {metrics}")
    else:
        print(f"Using existing model from: {args.model_dir}")


if __name__ == "__main__":
    main()
