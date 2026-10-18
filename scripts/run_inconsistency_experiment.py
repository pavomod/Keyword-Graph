"""Inconsistency-only analysis of a requirement list, with and without the baseline clustering.

Two strategies, same model, same inconsistency-only prompt:

1. ``baseline``  requirements clustered directly on their sentence embeddings
                 ("As a {role}, I want {feature}", BAAI/bge-m3, K-Means, K chosen with the
                 elbow method), then analysed cluster by cluster in a single call.
2. ``flat``      all the requirements sent as a single cluster (no clustering).

Outputs in ``results/<dataset>/inconsistency/``:
    baseline_clustering.json     cluster -> requirement ids and the K selection
    baseline.json, flat.json     raw model output + requirement_to_cluster_mapping + token usage
    pairs.csv                    one row per inconsistent pair, both strategies
    summary.md                   counts, pairs and token usage

Usage (from the repository root):
    python scripts/run_inconsistency_experiment.py --input-csv data/ceps_mutated.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import google.generativeai as genai  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.run_token_experiments import parse_json, phase4_message, usage_of  # noqa: E402
from src.phase3.comparison.compare_pipelines import cluster_requirements_directly  # noqa: E402
from src.phase4.llm.analyzer import build_cluster_input  # noqa: E402

PROMPT = ROOT / "src" / "phase4" / "llm" / "prompt" / "system_prompt_inconsistency.txt"


def baseline_partition(df: pd.DataFrame, out_dir: Path, k_min: int, k_max: int) -> dict[str, list[str]]:
    texts = [f"As a {str(r['role']).strip()}, I want {str(r['feature']).strip()}" for _, r in df.iterrows()]
    _, labels, diagnostics = cluster_requirements_directly(
        requirement_texts=texts, selected_k=None, k_min=k_min, k_max=k_max,
        embedding_model="BAAI/bge-m3", random_state=42,
    )
    mapping: dict[str, list[str]] = {}
    for rid, label in zip(df["id"].astype(str), labels):
        mapping.setdefault(str(label), []).append(rid)
    mapping = dict(sorted(mapping.items(), key=lambda kv: int(kv[0])))
    report = {"selected_k": diagnostics["selected_k"], "k_selection": diagnostics["k_selection"],
              "clusters": mapping, "sizes": {c: len(r) for c, r in mapping.items()}}
    (out_dir / "baseline_clustering.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[baseline] K = {diagnostics['selected_k']}, sizes = {sorted(report['sizes'].values(), reverse=True)}")
    return mapping


def analyse(mapping: dict[str, list[str]], texts: dict[str, str], model_name: str, out_path: Path) -> dict[str, Any]:
    system_prompt = PROMPT.read_text(encoding="utf-8")
    clusters = build_cluster_input(mapping, texts)
    model = genai.GenerativeModel(model_name)
    start = time.time()
    response = model.generate_content(
        phase4_message(system_prompt, clusters),
        generation_config=genai.types.GenerationConfig(temperature=0.2, top_p=0.9, max_output_tokens=50000),
        request_options={"timeout": 1800, "retry": None},
        stream=True,
    )
    for n, _ in enumerate(response, 1):  # streaming keeps the connection alive and shows progress
        print(f"  chunk {n} at {round(time.time() - start)} s", flush=True)
    response.resolve()
    result = parse_json(response.text)
    usage = usage_of(response) | {"seconds": round(time.time() - start, 1), "model": model_name}
    result["requirement_to_cluster_mapping"] = {rid: cid for cid, rids in mapping.items() for rid in rids}
    result["token_usage"] = usage
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{out_path.stem}] {usage}")
    return result


def findings(result: dict[str, Any]) -> list[dict[str, Any]]:
    mapping = result["requirement_to_cluster_mapping"]
    out = []
    for cluster in result.get("clusters", []):
        cid = str(cluster.get("cluster_id"))
        for item in cluster.get("inconsistencies", []):
            ids = sorted({str(i) for i in item.get("requirements", [])}, key=int)
            out.append({"cluster": cid, "ids": ids, "type": item.get("type", ""), "reason": item.get("reason", ""),
                        "outside_cluster": [i for i in ids if str(mapping.get(i)) != cid]})
    return out


def write_report(dataset: str, out_dir: Path, results: dict[str, dict[str, Any]], texts: dict[str, str]) -> None:
    rows = []
    for strategy, result in results.items():
        for n, f in enumerate(findings(result), 1):
            for a, b in itertools.combinations(f["ids"], 2):
                rows.append({"strategy": strategy, "finding": n, "cluster": f["cluster"], "req_a": a, "req_b": b,
                             "type": f["type"], "group_size": len(f["ids"]), "reason": f["reason"]})
    with (out_dir / "pairs.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["strategy"])
        writer.writeheader()
        writer.writerows(rows)

    pair_sets = {s: {(r["req_a"], r["req_b"]) for r in rows if r["strategy"] == s} for s in results}
    lines = [f"# Inconsistency-only analysis: {dataset}", ""]
    lines += ["| Strategy | Clusters | Findings | Distinct pairs | Input tok | Output tok | Thinking tok | Total tok | s |",
              "|---|---|---|---|---|---|---|---|---|"]
    for strategy, result in results.items():
        u = result["token_usage"]
        lines.append(f"| {strategy} | {len(result.get('clusters', []))} | {len(findings(result))} | {len(pair_sets[strategy])} "
                     f"| {u['prompt']} | {u['output']} | {u['thinking']} | {u['total']} | {u['seconds']} |")
    if set(results) == {"baseline", "flat"}:
        both = pair_sets["baseline"] & pair_sets["flat"]
        lines += ["", f"Pairs found by both: {len(both)}; only baseline: {len(pair_sets['baseline'] - both)}; "
                      f"only flat: {len(pair_sets['flat'] - both)}."]
    for strategy, result in results.items():
        lines += ["", f"## {strategy}", ""]
        for n, f in enumerate(findings(result), 1):
            flag = f" (outside cluster: {', '.join(f['outside_cluster'])})" if f["outside_cluster"] else ""
            lines.append(f"{n}. **{' – '.join(f['ids'])}** [{f['type']}], cluster {f['cluster']}{flag}: {f['reason']}")
    lines += ["", "## Requirements cited", ""]
    cited = sorted({i for r in rows for i in (r["req_a"], r["req_b"])}, key=int)
    lines += [f"- **{i}**: {texts.get(i, '')}" for i in cited]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--k-min", type=int, default=5)
    parser.add_argument("--k-max", type=int, default=25)
    parser.add_argument("--only", choices=["baseline", "flat"], default=None)
    parser.add_argument("--report-only", action="store_true", help="Rebuild pairs.csv and summary.md from saved outputs")
    args = parser.parse_args()

    csv_path = ROOT / args.input_csv
    dataset = csv_path.stem
    out_dir = ROOT / "results" / dataset / "inconsistency"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    texts = {str(r["id"]): str(r["feature"]).strip() for _, r in df.iterrows()}
    strategies = [args.only] if args.only else ["baseline", "flat"]

    if not args.report_only:
        api_key = os.getenv("MODEL_API_KEY")
        if not api_key:
            raise SystemExit("MODEL_API_KEY not set in .env")
        genai.configure(api_key=api_key)
        for strategy in strategies:
            if strategy == "baseline":
                mapping = baseline_partition(df, out_dir, args.k_min, args.k_max)
            else:
                mapping = {"0": list(texts)}
            analyse(mapping, texts, args.model, out_dir / f"{strategy}.json")

    results = {s: json.loads((out_dir / f"{s}.json").read_text(encoding="utf-8"))
               for s in ["baseline", "flat"] if (out_dir / f"{s}.json").exists()}
    write_report(dataset, out_dir, results, texts)
    print(f"-> {(out_dir / 'summary.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
