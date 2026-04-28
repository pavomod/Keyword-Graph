from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import logging as _py_logging
from transformers import logging as hf_logging

# Reduce verbosity from transformers / sentence-transformers to hide LOAD REPORT logs
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
hf_logging.set_verbosity_error()
_py_logging.getLogger("transformers").setLevel(_py_logging.ERROR)
_py_logging.getLogger("sentence_transformers").setLevel(_py_logging.ERROR)

import pandas as pd
from src.phase3.clustering import run_clustering_phase

from src.phase3.comparison.compare_pipelines import (
    build_direct_clustering_report,
    cluster_requirements_directly,
    compare_clustering_approaches,
)

from src.phase1.finetune.prepare_data import (
    _pick_input_csv,
    prepare_inference_dataframe,
    prepare_dataframe,
    split_dataset_fixed,
    to_hf_dataset_dict,
)
from src.phase1.finetune.train import (
    compute_keyword_comparison_metrics,
    predict_keywords,
    train_model,
)
from src.phase1.keyBERT.keybert_adapter import extract_requirement_text
from src.phase2.graph.build_graph import (
    attach_requirement_references,
    build_semantic_similarity_graph,
    parse_keywords,
    reduce_graph_nodes,
    remove_isolated_nodes,
    save_graph_gexf,
    style_nodes_by_degree,
)


DEFAULT_PIPELINE_CONFIG = {
    "train": False,
    "train_input_csv": None,
    "hf_dataset_dir": "data/hf_dataset",
    "model_dir": "models/flan-t5-keywords",
    "base_model_name": "google/flan-t5-base",
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
    "graph_out": "results/keyword_graph.gexf",
    "inference_batch_size": 16,
    "semantic_threshold": 0.4,
    "semantic_model": "all-MiniLM-L6-v2",
    "enable_node_reduction": True,
    "node_reduction_degree_threshold_low": 2,
    "node_reduction_degree_threshold_high": 5,
    "enable_clustering": True,
    "cluster_k": None,
    "cluster_k_min": 2,
    "cluster_k_max": 15,
    "cluster_embedding_dim": 64,
    "cluster_random_state": 42,
    "cluster_node2vec_walk_length": 30,
    "cluster_node2vec_num_walks": 200,
    "cluster_node2vec_workers": 4,
    "cluster_requirement_feature_weight": 0.5,
    "cluster_requirement_svd_dim": 16,
    "cluster_report_out": "results/clustering_report.json",
    "cluster_elbow_plot_out": "results/K.png",
    "enable_clustering_comparison": True,
    "clustering_comparison_out": "results/clustering_comparison.json",
    "direct_clustering_report_out": "results/direct_clustering_report.json",
    "use_finetuned": False,
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


def _save_verification_report(
    reference_df: pd.DataFrame,
    predictions_by_source: dict[str, str],
    comparison_json_out: str,
    inference_diagnostics: dict | None = None,
) -> tuple[str, dict]:
    records = []
    predictions: list[str] = []
    references: list[str] = []

    for row in reference_df.to_dict(orient="records"):
        source_id = str(row.get("source_id"))
        predicted = predictions_by_source.get(source_id, "")
        reference = str(row.get("target", ""))
        predictions.append(predicted)
        references.append(reference)
        records.append(
            {
                "source_id": source_id,
                "input": extract_requirement_text(str(row.get("input", ""))),
                "predicted_tags": predicted,
                "real_tags": reference,
            }
        )

    metrics = compute_keyword_comparison_metrics(predictions, references)

    payload = {
        "metrics": metrics,
        "diagnostics": {
            "keyword_inference": inference_diagnostics or {},
        },
        "records": records,
    }

    output_path = Path(comparison_json_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return str(output_path), metrics


def _print_verification_summary(comparison_path: str, metrics: dict) -> None:
    def fmt_pct(value: float) -> str:
        return f"{value * 100.0:.2f}%"

    print("[report] Comparison JSON saved")
    print(f"  path            : {comparison_path}")
    print("[report] Verification metrics (rows with tags)")
    print(f"  samples         : {int(metrics.get('samples', 0))}")
    print(
        "  exact_match     : "
        f"{fmt_pct(float(metrics.get('exact_match', 0.0)))} "
        f"(raw={float(metrics.get('exact_match', 0.0)):.6f})"
    )
    print(
        "  precision_micro : "
        f"{fmt_pct(float(metrics.get('precision_micro', 0.0)))} "
        f"(raw={float(metrics.get('precision_micro', 0.0)):.6f})"
    )
    print(
        "  recall_micro    : "
        f"{fmt_pct(float(metrics.get('recall_micro', 0.0)))} "
        f"(raw={float(metrics.get('recall_micro', 0.0)):.6f})"
    )
    print(
        "  f1_micro        : "
        f"{fmt_pct(float(metrics.get('f1_micro', 0.0)))} "
        f"(raw={float(metrics.get('f1_micro', 0.0)):.6f})"
    )
    print(
        "  mean_sample_f1  : "
        f"{fmt_pct(float(metrics.get('mean_sample_f1', 0.0)))} "
        f"(raw={float(metrics.get('mean_sample_f1', 0.0)):.6f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One-command pipeline for keyword extraction and graph generation. "
            "Training is optional and runs only if --train is specified."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/run_pipeline.json",
        help="JSON config path. CLI args override config values.",
    )
    train_group = parser.add_mutually_exclusive_group()
    train_group.add_argument("--train", dest="train", action="store_const", const=True)
    train_group.add_argument("--no-train", dest="train", action="store_const", const=False)
    parser.set_defaults(train=None)
    parser.add_argument("--train-input-csv", type=str, default=None)
    parser.add_argument("--hf-dataset-dir", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--base-model-name", type=str, default=None)
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
    parser.add_argument(
        "--enable-node-reduction",
        dest="enable_node_reduction",
        action="store_true",
        default=None,
        help="Enable node reduction: absorb low-degree nodes into high-degree hub nodes",
    )
    parser.add_argument(
        "--disable-node-reduction",
        dest="enable_node_reduction",
        action="store_false",
        help="Disable node reduction",
    )
    parser.add_argument(
        "--node-reduction-degree-threshold-low",
        type=int,
        default=None,
        help="Maximum degree for a node to be considered low (default: 2)",
    )
    parser.add_argument(
        "--node-reduction-degree-threshold-high",
        type=int,
        default=None,
        help="Minimum degree for a node to be considered hub (default: 5)",
    )
    cluster_group = parser.add_mutually_exclusive_group()
    cluster_group.add_argument(
        "--enable-clustering",
        dest="enable_clustering",
        action="store_const",
        const=True,
    )
    cluster_group.add_argument(
        "--no-clustering",
        dest="enable_clustering",
        action="store_const",
        const=False,
    )
    parser.set_defaults(enable_clustering=None)
    parser.add_argument("--cluster-k", type=int, default=None)
    parser.add_argument("--cluster-k-min", type=int, default=None)
    parser.add_argument("--cluster-k-max", type=int, default=None)
    parser.add_argument("--cluster-embedding-dim", type=int, default=None)
    parser.add_argument("--cluster-random-state", type=int, default=None)
    parser.add_argument("--cluster-node2vec-walk-length", type=int, default=None)
    parser.add_argument("--cluster-node2vec-num-walks", type=int, default=None)
    parser.add_argument("--cluster-node2vec-workers", type=int, default=None)
    parser.add_argument("--cluster-requirement-feature-weight", type=float, default=None)
    parser.add_argument("--cluster-requirement-svd-dim", type=int, default=None)
    parser.add_argument("--cluster-report-out", type=str, default=None)
    parser.add_argument("--cluster-elbow-plot-out", type=str, default=None)
    comparison_group = parser.add_mutually_exclusive_group()
    comparison_group.add_argument(
        "--enable-clustering-comparison",
        dest="enable_clustering_comparison",
        action="store_true",
    )
    comparison_group.add_argument(
        "--disable-clustering-comparison",
        dest="enable_clustering_comparison",
        action="store_false",
    )
    parser.set_defaults(enable_clustering_comparison=None)
    parser.add_argument("--clustering-comparison-out", type=str, default=None)
    parser.add_argument(
        "--direct-clustering-report-out",
        type=str,
        default=None,
        help="Output JSON path for the direct requirement clustering report",
    )
    # Prefer KeyBERT by default. Use this flag to force using the fine-tuned model instead.
    parser.add_argument(
        "--use-finetuned",
        dest="use_finetuned",
        action="store_const",
        const=True,
    )
    parser.set_defaults(use_finetuned=None)
    args = parser.parse_args()

    config_values = _load_pipeline_config(args.config)
    unknown_config_keys = sorted(set(config_values) - set(DEFAULT_PIPELINE_CONFIG))
    if unknown_config_keys:
        print(f"[config] Ignoring unknown keys: {unknown_config_keys}")

    should_train = bool(_resolve_setting(args.train, config_values, "train"))
    train_input_csv = _resolve_setting(args.train_input_csv, config_values, "train_input_csv")
    hf_dataset_dir = _resolve_setting(args.hf_dataset_dir, config_values, "hf_dataset_dir")
    model_dir_value = _resolve_setting(args.model_dir, config_values, "model_dir")
    base_model_name = _resolve_setting(args.base_model_name, config_values, "base_model_name")
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
    enable_node_reduction = bool(
        _resolve_setting(args.enable_node_reduction, config_values, "enable_node_reduction")
    )
    node_reduction_degree_threshold_low = int(
        _resolve_setting(
            args.node_reduction_degree_threshold_low,
            config_values,
            "node_reduction_degree_threshold_low",
        )
    )
    node_reduction_degree_threshold_high = int(
        _resolve_setting(
            args.node_reduction_degree_threshold_high,
            config_values,
            "node_reduction_degree_threshold_high",
        )
    )
    enable_clustering = bool(
        _resolve_setting(args.enable_clustering, config_values, "enable_clustering")
    )
    cluster_k = _resolve_setting(args.cluster_k, config_values, "cluster_k")
    cluster_k = int(cluster_k) if cluster_k is not None else None
    cluster_k_min = int(_resolve_setting(args.cluster_k_min, config_values, "cluster_k_min"))
    cluster_k_max = int(_resolve_setting(args.cluster_k_max, config_values, "cluster_k_max"))
    cluster_embedding_dim = int(
        _resolve_setting(args.cluster_embedding_dim, config_values, "cluster_embedding_dim")
    )
    cluster_random_state = int(
        _resolve_setting(args.cluster_random_state, config_values, "cluster_random_state")
    )
    cluster_node2vec_walk_length = int(
        _resolve_setting(
            args.cluster_node2vec_walk_length,
            config_values,
            "cluster_node2vec_walk_length",
        )
    )
    cluster_node2vec_num_walks = int(
        _resolve_setting(
            args.cluster_node2vec_num_walks,
            config_values,
            "cluster_node2vec_num_walks",
        )
    )
    cluster_node2vec_workers = int(
        _resolve_setting(
            args.cluster_node2vec_workers,
            config_values,
            "cluster_node2vec_workers",
        )
    )
    cluster_requirement_feature_weight = float(
        _resolve_setting(
            args.cluster_requirement_feature_weight,
            config_values,
            "cluster_requirement_feature_weight",
        )
    )
    cluster_requirement_svd_dim = int(
        _resolve_setting(
            args.cluster_requirement_svd_dim,
            config_values,
            "cluster_requirement_svd_dim",
        )
    )
    cluster_report_out = _resolve_setting(args.cluster_report_out, config_values, "cluster_report_out")
    cluster_elbow_plot_out = _resolve_setting(
        args.cluster_elbow_plot_out,
        config_values,
        "cluster_elbow_plot_out",
    )
    enable_clustering_comparison = bool(
        _resolve_setting(args.enable_clustering_comparison, config_values, "enable_clustering_comparison")
    )
    clustering_comparison_out = _resolve_setting(
        args.clustering_comparison_out, config_values, "clustering_comparison_out"
    )
    direct_clustering_report_out = _resolve_setting(
        args.direct_clustering_report_out,
        config_values,
        "direct_clustering_report_out",
    )

    model_dir = Path(model_dir_value)
    has_model = _model_exists(model_dir)

    prefer_finetuned = bool(_resolve_setting(args.use_finetuned, config_values, "use_finetuned"))

    if should_train:
        print("[1/4] Preparing training dataset split...")
        prep_info, _ = _prepare_dataset(
            train_input_csv,
            hf_dataset_dir,
            seed,
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
        )
        print(
            "Prepared training dataset from {input} | samples={samples} | split={train}/{validation}/{test} | held_out={real_case}".format(
                **prep_info
            )
        )
    else:
        print("[1/4] Training skipped (--no-train).")

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
        inference_model_dir: str | None = model_dir_value
    else:
        if prefer_finetuned:
            if has_model:
                print(f"[2/4] Using existing fine-tuned model from: {model_dir_value}")
                inference_model_dir = model_dir_value
            else:
                print(
                    f"[2/4] --use-finetuned requested but no fine-tuned model found in {model_dir_value}. "
                    f"Falling back to KeyBERT or base model."
                )
                inference_model_dir = None
        else:
            print(
                "[2/4] Using KeyBERT extractor by default. "
                "Pass --use-finetuned to use the fine-tuned model instead."
            )
            inference_model_dir = None

    print("[3/4] Predicting keywords for full dataset...")
    inference_csv_path = _pick_input_csv("data/crowd.csv")
    raw_df = pd.read_csv(inference_csv_path)
    inference_df = prepare_inference_dataframe(raw_df)
    inference_inputs = inference_df["input"].tolist()
    predictions, inference_diagnostics = predict_keywords(
        model_dir=inference_model_dir,
        base_model_name=base_model_name,
        inputs=inference_inputs,
        batch_size=inference_batch_size,
        max_input_len=max_input_len,
        max_target_len=max_target_len,
    )

    predictions_by_source = {
        str(source_id): prediction
        for source_id, prediction in zip(inference_df["source_id"].tolist(), predictions)
    }

    if "tags" in raw_df.columns:
        reference_df = prepare_dataframe(raw_df)
        comparison_path, comparison_metrics = _save_verification_report(
            reference_df=reference_df,
            predictions_by_source=predictions_by_source,
            comparison_json_out=comparison_json_out,
            inference_diagnostics=inference_diagnostics,
        )
        _print_verification_summary(comparison_path, comparison_metrics)
    else:
        print("No 'tags' column found in inference CSV. Skipping comparison report.")

    print("[4/5] Building global semantic graph from full-dataset predictions and exporting GEXF...")
    predicted_keyword_lists = [parse_keywords(prediction) for prediction in predictions]
    graph = build_semantic_similarity_graph(
        keyword_lists=predicted_keyword_lists,
        threshold=semantic_threshold,
        model_name=semantic_model,
    )
    graph = attach_requirement_references(
        graph=graph,
        keyword_lists=predicted_keyword_lists,
        requirement_ids=[str(source_id) for source_id in inference_df["source_id"].tolist()],
    )

    graph, removed_isolated = remove_isolated_nodes(graph)
    if removed_isolated:
        print(f"Removed {removed_isolated} isolated nodes with zero relations")

    if enable_node_reduction:
        print(
            "[4/5] Reducing graph nodes "
            f"(low_degree<={node_reduction_degree_threshold_low}, "
            f"high_degree>{node_reduction_degree_threshold_high})..."
        )
        initial_nodes = graph.number_of_nodes()
        graph = reduce_graph_nodes(
            graph=graph,
            degree_threshold_low=node_reduction_degree_threshold_low,
            degree_threshold_high=node_reduction_degree_threshold_high,
        )
        reduced_nodes = graph.number_of_nodes()
        print(
            f"Node reduction completed: {initial_nodes} nodes -> {reduced_nodes} nodes "
            f"({initial_nodes - reduced_nodes} absorbed)"
        )

    if enable_clustering:
        print("[5/5] Running clustering phase (Node2Vec -> K-Means -> UMAP)...")
        clustering_payload, graph = run_clustering_phase(
            graph=graph,
            output_json_path=cluster_report_out,
            output_elbow_plot_path=cluster_elbow_plot_out,
            embedding_dim=cluster_embedding_dim,
            requested_k=cluster_k,
            k_min=cluster_k_min,
            k_max=cluster_k_max,
            random_state=cluster_random_state,
            node2vec_walk_length=cluster_node2vec_walk_length,
            node2vec_num_walks=cluster_node2vec_num_walks,
            node2vec_workers=cluster_node2vec_workers,
            requirement_feature_weight=cluster_requirement_feature_weight,
            requirement_svd_dim=cluster_requirement_svd_dim,
            isolated_nodes_removed=removed_isolated,
        )
        print(
            "Clustering report saved in: "
            f"{cluster_report_out} | selected_k={clustering_payload['clustering'].get('selected_k')}"
        )
        elbow_plot_path = clustering_payload["clustering"].get("elbow_plot")
        if elbow_plot_path:
            print(f"Clustering elbow plot saved in: {elbow_plot_path}")
        
        if enable_clustering_comparison:
            print("[6/6] Running clustering comparison (keyword-graph vs direct embedding)...")
            raw_requirement_data = raw_df.to_dict(orient="records")
            direct_requirement_texts = [
                f"As a {str(row.get('role', 'user')).strip()}, "
                f"I want {str(row.get('feature', '')).strip()} "
                f"so that {str(row.get('benefit', '')).strip()}"
                for row in raw_requirement_data
            ]
            
            # Cluster requirements directly using their text embeddings
            direct_clustering_result = cluster_requirements_directly(
                requirement_texts=direct_requirement_texts,
                selected_k=cluster_k,
                k_min=cluster_k_min,
                k_max=cluster_k_max,
                embedding_model=semantic_model,
                random_state=cluster_random_state,
            )

            direct_report_path = build_direct_clustering_report(
                requirement_texts=direct_requirement_texts,
                requirement_ids=[str(source_id) for source_id in inference_df["source_id"].tolist()],
                direct_clustering_result=direct_clustering_result,
                output_path=direct_clustering_report_out,
            )
            print(f"Direct clustering report saved in: {direct_report_path}")
            
            # Compare the two approaches
            comparison_path = compare_clustering_approaches(
                keyword_graph_clustering=clustering_payload,
                direct_clustering_result=direct_clustering_result,
                requirement_ids=[str(source_id) for source_id in inference_df["source_id"].tolist()],
                output_path=clustering_comparison_out,
            )
            print(f"Clustering comparison report saved in: {comparison_path}")
    else:
        print("[5/5] Clustering phase skipped (--no-clustering).")

    graph = style_nodes_by_degree(graph)
    graph_path = save_graph_gexf(graph, graph_out)
    print(
        f"Graph saved in: {graph_path} | nodes={graph.number_of_nodes()} | edges={graph.number_of_edges()}"
    )


if __name__ == "__main__":
    main()
