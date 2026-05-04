from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import logging as _py_logging
from transformers import logging as hf_logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

from src.phase1.keyBERT.prepare_data import (
    _pick_input_csv,
    prepare_inference_dataframe,
)
from src.phase1.keyBERT.prediction import predict_keywords
from src.phase2.graph.build_graph import (
    attach_requirement_references,
    build_semantic_similarity_graph,
    parse_keywords,
    reduce_graph_nodes,
    remove_isolated_nodes,
    save_graph_gexf,
    style_nodes_by_degree,
)
from src.phase4.llm.analyzer import run_analysis as run_llm_analysis


DEFAULT_PIPELINE_CONFIG = {
    "graph_out": "results/keyword_graph.gexf",
    "extraction_out": "results/extraction.json",
    # KeyBERT defaults
    "keybert_model": "BAAI/bge-m3",
    "keybert_top_n": 6,
    "keybert_keyphrase_ngram_range": [1, 2],
    "keybert_stop_words": "english",
    "keybert_threshold": 0.5,
    "keybert_use_mmr": True,
    "keybert_diversity": 0.7,
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
    "input_csv": None,
    "csv_sep": ",",
    # Phase 4 - LLM Analysis defaults
    "enable_phase4_analysis": False,
    "phase4_clustering_report": "results/clustering_report.json",
    "phase4_system_prompt": "src/phase4/llm/prompt/system_prompt.txt",
    "phase4_output": "results/phase4_analysis.json",
    "phase4_use_direct_clustering": False,
    "phase4_preview": False,
    "phase4_preview_output": "results/phase4_preview.txt",
    "phase4_gemini_model": "gemini-2.5-pro",
    "phase4_gemini_temperature": 0.2,
    "phase4_gemini_top_p": 0.9,
    "phase4_gemini_max_tokens": 16000,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Keyword extraction pipeline using KeyBERT and semantic graph generation."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/run_pipeline.json",
        help="JSON config path. CLI args override config values.",
    )
    parser.add_argument("--input-csv", type=str, default=None, help="Path to input CSV")
    parser.add_argument("--csv-sep", type=str, default=None, help="CSV separator (default: ',')")
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
    )
    parser.add_argument(
        "--disable-node-reduction",
        dest="enable_node_reduction",
        action="store_false",
    )
    parser.add_argument("--node-reduction-degree-threshold-low", type=int, default=None)
    parser.add_argument("--node-reduction-degree-threshold-high", type=int, default=None)
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
    parser.add_argument("--direct-clustering-report-out", type=str, default=None)
    
    # Phase 4 - LLM Analysis arguments
    phase4_group = parser.add_mutually_exclusive_group()
    phase4_group.add_argument(
        "--phase4-only",
        dest="phase4_only",
        action="store_true",
        help="Run Phase 4 analysis only (skip phases 1-3)",
    )
    phase4_group.add_argument(
        "--with-phase4",
        dest="run_phase4",
        action="store_true",
        help="Run full pipeline including Phase 4",
    )
    parser.set_defaults(phase4_only=False, run_phase4=False)
    parser.add_argument("--phase4-clustering-report", type=str, default=None, 
                       help="Path to clustering report JSON for Phase 4")
    parser.add_argument("--phase4-system-prompt", type=str, default=None,
                       help="Path to system prompt file for Phase 4")
    parser.add_argument("--phase4-output", type=str, default=None,
                       help="Output path for Phase 4 analysis results")
    parser.add_argument("--phase4-use-direct-clustering", action="store_true",
                       help="Use direct clustering report instead of keyword-graph clustering")
    parser.add_argument("--phase4-preview", action="store_true",
                       help="Preview the input that will be sent to Gemini without sending it")
    parser.add_argument("--phase4-preview-output", type=str, default=None,
                       help="Path to save the preview output (default: results/phase4_preview.txt)")
    
    args = parser.parse_args()

    config_values = _load_pipeline_config(args.config)
    unknown_config_keys = sorted(set(config_values) - set(DEFAULT_PIPELINE_CONFIG))
    if unknown_config_keys:
        print(f"[config] Ignoring unknown keys: {unknown_config_keys}")

    input_csv = _resolve_setting(args.input_csv, config_values, "input_csv")
    csv_sep = _resolve_setting(args.csv_sep, config_values, "csv_sep")
    graph_out = _resolve_setting(args.graph_out, config_values, "graph_out")
    extraction_out = _resolve_setting(None, config_values, "extraction_out")
    semantic_threshold = float(
        _resolve_setting(args.semantic_threshold, config_values, "semantic_threshold")
    )
    semantic_model = _resolve_setting(args.semantic_model, config_values, "semantic_model")
    # KeyBERT params (from config)
    keybert_model = _resolve_setting(None, config_values, "keybert_model")
    keybert_top_n = _resolve_setting(None, config_values, "keybert_top_n")
    keybert_keyphrase_ngram_range = _resolve_setting(
        None, config_values, "keybert_keyphrase_ngram_range"
    )
    keybert_stop_words = _resolve_setting(None, config_values, "keybert_stop_words")
    keybert_threshold = _resolve_setting(None, config_values, "keybert_threshold")
    keybert_use_mmr = _resolve_setting(None, config_values, "keybert_use_mmr")
    keybert_diversity = _resolve_setting(None, config_values, "keybert_diversity")
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
    
    # Phase 4 - LLM Analysis parameters
    phase4_clustering_report = _resolve_setting(
        args.phase4_clustering_report, config_values, "phase4_clustering_report"
    )
    phase4_system_prompt = _resolve_setting(
        args.phase4_system_prompt, config_values, "phase4_system_prompt"
    )
    phase4_output = _resolve_setting(
        args.phase4_output, config_values, "phase4_output"
    )
    phase4_use_direct_clustering = args.phase4_use_direct_clustering or bool(
        _resolve_setting(None, config_values, "phase4_use_direct_clustering")
    )
    phase4_preview = args.phase4_preview or bool(
        _resolve_setting(None, config_values, "phase4_preview")
    )
    phase4_preview_output = args.phase4_preview_output or _resolve_setting(
        None, config_values, "phase4_preview_output"
    )
    phase4_gemini_model = _resolve_setting(
        None, config_values, "phase4_gemini_model"
    )
    phase4_gemini_temperature = float(
        _resolve_setting(None, config_values, "phase4_gemini_temperature")
    )
    phase4_gemini_top_p = float(
        _resolve_setting(None, config_values, "phase4_gemini_top_p")
    )
    phase4_gemini_max_tokens = int(
        _resolve_setting(None, config_values, "phase4_gemini_max_tokens")
    )
    
    # Phase 4 only mode
    if args.phase4_only:
        print("=" * 60)
        print("Phase 4 Only Mode: Analyzing clusters with Gemini 2.5 Pro")
        print("=" * 60)
        
        # Determine which clustering report to use
        if phase4_use_direct_clustering:
            clustering_report_path = direct_clustering_report_out
            print(f"Using direct clustering report: {clustering_report_path}")
        else:
            clustering_report_path = phase4_clustering_report
            print(f"Using keyword-graph clustering report: {clustering_report_path}")
        
        # Preview mode
        if args.phase4_preview:
            print("\n📋 PREVIEW MODE - Showing input without sending to API\n")
            try:
                run_llm_analysis(
                    clustering_report_path=clustering_report_path,
                    extraction_path=extraction_out,
                    system_prompt_path=phase4_system_prompt,
                    output_path=phase4_preview_output,
                    gemini_model=phase4_gemini_model,
                    gemini_temperature=phase4_gemini_temperature,
                    gemini_top_p=phase4_gemini_top_p,
                    gemini_max_tokens=phase4_gemini_max_tokens,
                    dry_run=True,
                )
                print("=" * 60)
                print("✓ Preview generated successfully!")
                print(f"✓ View the preview in: {phase4_preview_output}")
                print("=" * 60)
            except Exception as e:
                print(f"Error generating Phase 4 preview: {e}")
                raise
        else:
            # Normal mode - send to API
            try:
                run_llm_analysis(
                    clustering_report_path=clustering_report_path,
                    extraction_path=extraction_out,
                    system_prompt_path=phase4_system_prompt,
                    output_path=phase4_output,
                    gemini_model=phase4_gemini_model,
                    gemini_temperature=phase4_gemini_temperature,
                    gemini_top_p=phase4_gemini_top_p,
                    gemini_max_tokens=phase4_gemini_max_tokens,
                )
                print("=" * 60)
                print("Phase 4 analysis completed successfully!")
                print("=" * 60)
            except Exception as e:
                print(f"Error during Phase 4 analysis: {e}")
                raise
        
        return

    print("[1/3] Extracting keywords with KeyBERT...")
    inference_csv_path = _pick_input_csv(input_csv)
    raw_df = pd.read_csv(inference_csv_path, sep=csv_sep)
    inference_df = prepare_inference_dataframe(raw_df)
    model_inputs = raw_df["feature"].fillna("").astype(str).tolist()
    keybert_kwargs = {
        "top_n": int(keybert_top_n) if keybert_top_n is not None else None,
        "keyphrase_ngram_range": tuple(keybert_keyphrase_ngram_range)
        if keybert_keyphrase_ngram_range
        else None,
        "stop_words": keybert_stop_words,
        "threshold": float(keybert_threshold) if keybert_threshold is not None else None,
        "use_mmr": bool(keybert_use_mmr) if keybert_use_mmr is not None else None,
        "diversity": float(keybert_diversity) if keybert_diversity is not None else None,
        "model_name": keybert_model,
    }

    predictions, _ = predict_keywords(
        inputs=model_inputs,
        keybert_kwargs=keybert_kwargs,
    )
    print(f"Keywords extracted for {len(predictions)} requirements.")

    predictions_by_source = {
        str(source_id): prediction
        for source_id, prediction in zip(inference_df["source_id"].tolist(), predictions)
    }

    extraction_records = [
        {
            "source_id": str(source_id),
            "input": str(feature).strip(),
            "predicted_tags": predictions_by_source.get(str(source_id), ""),
        }
        for source_id, feature in zip(
            raw_df["id"].astype(str) if "id" in raw_df.columns else raw_df.index.astype(str),
            raw_df["feature"].tolist(),
        )
    ]

    extraction_path = Path(extraction_out)
    extraction_path.parent.mkdir(parents=True, exist_ok=True)
    extraction_path.write_text(
        json.dumps({"records": extraction_records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Extraction saved in: {extraction_path}")

    print("[2/3] Building global semantic graph and exporting GEXF...")
    predicted_keyword_lists = [parse_keywords(prediction) for prediction in predictions]
    graph = build_semantic_similarity_graph(
        keyword_lists=predicted_keyword_lists,
        threshold=semantic_threshold,
        model_name=semantic_model,
    )
    print(
        "[diagnostic] semantic graph before pruning | "
        f"nodes={graph.number_of_nodes()} | edges={graph.number_of_edges()} | "
        f"threshold={semantic_threshold}"
    )
    graph = attach_requirement_references(
        graph=graph,
        keyword_lists=predicted_keyword_lists,
        requirement_ids=[str(source_id) for source_id in inference_df["source_id"].tolist()],
    )

    graph, removed_isolated = remove_isolated_nodes(graph)
    if removed_isolated:
        print(f"Removed {removed_isolated} isolated nodes with zero relations")
    if graph.number_of_nodes() == 0:
        print(
            "[diagnostic] graph is empty after removing isolated nodes; "
            "lower semantic_threshold or add co-occurrence edges to keep keywords connected"
        )

    if enable_node_reduction:
        print(
            "[2/3] Reducing graph nodes "
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
        print("[3/3] Running clustering phase (Node2Vec -> K-Means -> UMAP)...")
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
            print("[+] Running clustering comparison (keyword-graph vs direct embedding)...")
            raw_requirement_data = raw_df.to_dict(orient="records")
            direct_requirement_texts = [
                f"As a {str(row.get('role', 'user')).strip()}, "
                f"I want {str(row.get('feature', '')).strip()}"
                for row in raw_requirement_data
            ]

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

            comparison_path = compare_clustering_approaches(
                keyword_graph_clustering=clustering_payload,
                direct_clustering_result=direct_clustering_result,
                requirement_ids=[str(source_id) for source_id in inference_df["source_id"].tolist()],
                output_path=clustering_comparison_out,
            )
            print(f"Clustering comparison report saved in: {comparison_path}")
    else:
        print("[3/3] Clustering phase skipped (--no-clustering).")

    graph = style_nodes_by_degree(graph)
    graph_path = save_graph_gexf(graph, graph_out)
    print(
        f"Graph saved in: {graph_path} | nodes={graph.number_of_nodes()} | edges={graph.number_of_edges()}"
    )
    
    # Phase 4 - LLM Analysis (optional)
    if args.run_phase4 or _resolve_setting(None, config_values, "enable_phase4_analysis"):
        print("\n" + "=" * 60)
        print("[Phase 4] Running LLM-based analysis with Gemini 2.5 Pro...")
        print("=" * 60)
        
        # Determine which clustering report to use
        if phase4_use_direct_clustering:
            clustering_report_path = direct_clustering_report_out
            print(f"Using direct clustering report: {clustering_report_path}")
        else:
            clustering_report_path = phase4_clustering_report
            print(f"Using keyword-graph clustering report: {clustering_report_path}")
        
        # Preview mode
        if args.phase4_preview:
            print("\n📋 PREVIEW MODE - Showing input without sending to API\n")
            try:
                run_llm_analysis(
                    clustering_report_path=clustering_report_path,
                    extraction_path=extraction_out,
                    system_prompt_path=phase4_system_prompt,
                    output_path=phase4_preview_output,
                    gemini_model=phase4_gemini_model,
                    gemini_temperature=phase4_gemini_temperature,
                    gemini_top_p=phase4_gemini_top_p,
                    gemini_max_tokens=phase4_gemini_max_tokens,
                    dry_run=True,
                )
                print("=" * 60)
                print("[Phase 4] ✓ Preview generated successfully!")
                print(f"[Phase 4] ✓ View the preview in: {phase4_preview_output}")
                print("=" * 60)
            except Exception as e:
                print(f"[Phase 4] Error generating preview: {e}")
                raise
        else:
            # Normal mode - send to API
            try:
                run_llm_analysis(
                    clustering_report_path=clustering_report_path,
                    extraction_path=extraction_out,
                    system_prompt_path=phase4_system_prompt,
                    output_path=phase4_output,
                    gemini_model=phase4_gemini_model,
                    gemini_temperature=phase4_gemini_temperature,
                    gemini_top_p=phase4_gemini_top_p,
                    gemini_max_tokens=phase4_gemini_max_tokens,
                )
                print("=" * 60)
                print("[Phase 4] Analysis completed successfully!")
                print("=" * 60)
            except Exception as e:
                print(f"[Phase 4] Error during analysis: {e}")
                raise


if __name__ == "__main__":
    main()