from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.finetune.prepare_data import (
    _pick_input_csv,
    prepare_dataframe,
    split_dataset_fixed,
    to_hf_dataset_dict,
)
from src.finetune.train import (
    compute_keyword_comparison_metrics,
    predict_keywords,
    train_model,
)
from src.graph.build_graph import (
    build_semantic_similarity_graph,
    parse_keywords,
    save_graph_gexf,
)


def _model_exists(model_dir: Path) -> bool:
    required_any = [
        model_dir / "adapter_config.json",
        model_dir / "pytorch_model.bin",
        model_dir / "model.safetensors",
    ]
    tokenizer_required = model_dir / "tokenizer_config.json"
    return model_dir.exists() and tokenizer_required.exists() and any(p.exists() for p in required_any)


def _prepare_dataset(
    input_csv: str | None,
    hf_out_dir: str,
    seed: int,
    train_size: int,
    validation_size: int,
    test_size: int,
) -> tuple[dict, pd.DataFrame]:
    input_path = _pick_input_csv(input_csv)
    raw_df = pd.read_csv(input_path)

    formatted_df = prepare_dataframe(raw_df)
    train_df, val_df, test_df, real_df = split_dataset_fixed(
        formatted_df,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        seed=seed,
    )

    hf_ds = to_hf_dataset_dict(train_df, val_df, test_df)
    hf_ds.save_to_disk(hf_out_dir)

    return (
        {
            "input": str(input_path),
            "samples": len(formatted_df),
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
            "real_case": len(real_df),
            "dataset_dir": hf_out_dir,
        },
        real_df,
    )


def _save_real_case_report(
    real_df: pd.DataFrame,
    predictions: list[str],
    comparison_json_out: str,
) -> tuple[str, dict]:
    references = real_df["target"].tolist()
    metrics = compute_keyword_comparison_metrics(predictions, references)

    comparison_records = []
    for row, prediction, reference in zip(real_df.to_dict(orient="records"), predictions, references):
        comparison_records.append(
            {
                "source_id": row.get("source_id"),
                "input": row.get("input"),
                "predicted_tags": prediction,
                "real_tags": reference,
            }
        )

    payload = {
        "metrics": metrics,
        "records": comparison_records,
    }

    output_path = Path(comparison_json_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return str(output_path), metrics


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
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--validation-size", type=int, default=250)
    parser.add_argument("--test-size", type=int, default=250)
    parser.add_argument(
        "--comparison-json-out",
        type=str,
        default="results/real_case_comparison.json",
    )
    parser.add_argument(
        "--graph-out",
        type=str,
        default="results/keyword_graph_real_case.gexf",
    )
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.4,
        help="Minimum cosine similarity to create an edge between keywords",
    )
    parser.add_argument(
        "--semantic-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-Transformers model used to compute keyword similarity",
    )
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

    print("[1/4] Preparing fixed dataset split...")
    prep_info, real_df = _prepare_dataset(
        args.input_csv,
        args.hf_dataset_dir,
        args.seed,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
    )
    print(
        "Prepared dataset from {input} | samples={samples} | split={train}/{validation}/{test} | real_case={real_case}".format(
            **prep_info
        )
    )

    if should_train:
        print("[2/4] Fine-tuning model...")
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

    print("[3/4] Predicting tags for real-case split...")
    real_inputs = real_df["input"].tolist()
    predictions = predict_keywords(
        model_dir=args.model_dir,
        inputs=real_inputs,
        batch_size=args.inference_batch_size,
        max_input_len=args.max_input_len,
        max_target_len=args.max_target_len,
    )

    comparison_path, comparison_metrics = _save_real_case_report(
        real_df=real_df,
        predictions=predictions,
        comparison_json_out=args.comparison_json_out,
    )
    print(f"Real-case comparison JSON saved in: {comparison_path}")
    print(f"Real-case metrics: {comparison_metrics}")

    print("[4/4] Building global semantic graph from predicted keywords and exporting GEXF...")
    predicted_keyword_lists = [parse_keywords(prediction) for prediction in predictions]
    graph = build_semantic_similarity_graph(
        keyword_lists=predicted_keyword_lists,
        threshold=args.semantic_threshold,
        model_name=args.semantic_model,
    )
    graph_path = save_graph_gexf(graph, args.graph_out)
    print(
        f"Graph saved in: {graph_path} | nodes={graph.number_of_nodes()} | edges={graph.number_of_edges()}"
    )


if __name__ == "__main__":
    main()
