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


DEFAULT_PIPELINE_CONFIG = {
    "mode": "auto",
    "input_csv": None,
    "hf_dataset_dir": "data/hf_dataset",
    "model_dir": "models/flan-t5-keywords",
    "seed": 42,
    "epochs": 5,
    "batch_size": 8,
    "lr": 3e-4,
    "max_input_len": 256,
    "max_target_len": 64,
    "fp16": None,
    "require_cuda": False,
    "train_size": 2000,
    "validation_size": 250,
    "test_size": 250,
    "comparison_json_out": "results/real_case_comparison.json",
    "graph_out": "results/keyword_graph_real_case.gexf",
    "inference_batch_size": 16,
    "semantic_threshold": 0.4,
    "semantic_model": "all-MiniLM-L6-v2",
}


def _load_pipeline_config(config_path: str | None) -> dict:
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.exists():
        return {}

    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return content


def _resolve_setting(cli_value, config: dict, key: str):
    if cli_value is not None:
        return cli_value
    if key in config:
        return config[key]
    return DEFAULT_PIPELINE_CONFIG[key]


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
    parser.add_argument(
        "--config",
        type=str,
        default="config/run_pipeline.json",
        help="JSON config path. CLI args override config values.",
    )
    parser.add_argument("--mode", choices=["auto", "finetune", "reuse"], default=None)
    parser.add_argument("--input-csv", type=str, default=None)
    parser.add_argument("--hf-dataset-dir", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-input-len", type=int, default=None)
    parser.add_argument("--max-target-len", type=int, default=None)
    fp16_group = parser.add_mutually_exclusive_group()
    fp16_group.add_argument("--fp16", dest="fp16", action="store_const", const=True)
    fp16_group.add_argument("--no-fp16", dest="fp16", action="store_const", const=False)
    parser.set_defaults(fp16=None)
    parser.add_argument("--require-cuda", action="store_true", default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--validation-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument(
        "--comparison-json-out",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--graph-out",
        type=str,
        default=None,
    )
    parser.add_argument("--inference-batch-size", type=int, default=None)
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=None,
        help="Minimum cosine similarity to create an edge between keywords",
    )
    parser.add_argument(
        "--semantic-model",
        type=str,
        default=None,
        help="Sentence-Transformers model used to compute keyword similarity",
    )
    args = parser.parse_args()

    config_values = _load_pipeline_config(args.config)
    unknown_config_keys = sorted(set(config_values) - set(DEFAULT_PIPELINE_CONFIG))
    if unknown_config_keys:
        print(f"[config] Ignoring unknown keys: {unknown_config_keys}")

    mode = _resolve_setting(args.mode, config_values, "mode")
    input_csv = _resolve_setting(args.input_csv, config_values, "input_csv")
    hf_dataset_dir = _resolve_setting(args.hf_dataset_dir, config_values, "hf_dataset_dir")
    model_dir_value = _resolve_setting(args.model_dir, config_values, "model_dir")
    seed = int(_resolve_setting(args.seed, config_values, "seed"))
    epochs = int(_resolve_setting(args.epochs, config_values, "epochs"))
    batch_size = int(_resolve_setting(args.batch_size, config_values, "batch_size"))
    lr = float(_resolve_setting(args.lr, config_values, "lr"))
    max_input_len = int(_resolve_setting(args.max_input_len, config_values, "max_input_len"))
    max_target_len = int(_resolve_setting(args.max_target_len, config_values, "max_target_len"))
    requested_fp16 = _resolve_setting(args.fp16, config_values, "fp16")
    require_cuda = bool(_resolve_setting(args.require_cuda, config_values, "require_cuda"))
    train_size = int(_resolve_setting(args.train_size, config_values, "train_size"))
    validation_size = int(_resolve_setting(args.validation_size, config_values, "validation_size"))
    test_size = int(_resolve_setting(args.test_size, config_values, "test_size"))
    comparison_json_out = _resolve_setting(
        args.comparison_json_out,
        config_values,
        "comparison_json_out",
    )
    graph_out = _resolve_setting(args.graph_out, config_values, "graph_out")
    inference_batch_size = int(
        _resolve_setting(args.inference_batch_size, config_values, "inference_batch_size")
    )
    semantic_threshold = float(
        _resolve_setting(args.semantic_threshold, config_values, "semantic_threshold")
    )
    semantic_model = _resolve_setting(args.semantic_model, config_values, "semantic_model")

    model_dir = Path(model_dir_value)
    has_model = _model_exists(model_dir)

    if mode == "reuse" and not has_model:
        raise FileNotFoundError(
            f"Model not found in {model_dir}. Use --mode finetune or --mode auto first."
        )

    should_train = mode == "finetune" or (mode == "auto" and not has_model)

    print("[1/4] Preparing fixed dataset split...")
    prep_info, real_df = _prepare_dataset(
        input_csv,
        hf_dataset_dir,
        seed,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
    )
    print(
        "Prepared dataset from {input} | samples={samples} | split={train}/{validation}/{test} | real_case={real_case}".format(
            **prep_info
        )
    )

    if should_train:
        print("[2/4] Fine-tuning model...")
        metrics = train_model(
            dataset_dir=hf_dataset_dir,
            output_dir=model_dir_value,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            max_input_len=max_input_len,
            max_target_len=max_target_len,
            use_fp16=requested_fp16,
            require_cuda=require_cuda,
        )
        print(f"Training complete. Model saved in: {model_dir_value}")
        print(f"Eval metrics: {metrics}")
    else:
        print(f"Using existing model from: {model_dir_value}")

    print("[3/4] Predicting tags for real-case split...")
    real_inputs = real_df["input"].tolist()
    predictions = predict_keywords(
        model_dir=model_dir_value,
        inputs=real_inputs,
        batch_size=inference_batch_size,
        max_input_len=max_input_len,
        max_target_len=max_target_len,
    )

    comparison_path, comparison_metrics = _save_real_case_report(
        real_df=real_df,
        predictions=predictions,
        comparison_json_out=comparison_json_out,
    )
    print(f"Real-case comparison JSON saved in: {comparison_path}")
    print(f"Real-case metrics: {comparison_metrics}")

    print("[4/4] Building global semantic graph from predicted keywords and exporting GEXF...")
    predicted_keyword_lists = [parse_keywords(prediction) for prediction in predictions]
    graph = build_semantic_similarity_graph(
        keyword_lists=predicted_keyword_lists,
        threshold=semantic_threshold,
        model_name=semantic_model,
    )
    graph_path = save_graph_gexf(graph, graph_out)
    print(
        f"Graph saved in: {graph_path} | nodes={graph.number_of_nodes()} | edges={graph.number_of_edges()}"
    )


if __name__ == "__main__":
    main()
