"""Token accounting and the no-clustering experiment.

Three measurements, all with the models, prompts and parameters of the thesis profiles:

1. ``count``    exact token counts (free ``count_tokens`` endpoint) of the inputs and outputs
                of the runs reported in the thesis: keyword extraction and the four Phase 4
                analyses (graph-based and baseline partition, both datasets).
2. ``flat``     Phase 4 analysis WITHOUT clustering: all the requirements of a dataset are
                sent as a single cluster. Results go to ``results/<dataset>/phase4_analysis_flat.json``.
3. ``rerun``    the graph-based and baseline Phase 4 analyses run again only to measure the
                full token usage, thinking tokens included, which the original runs did not
                record. Results go to ``results/token_runs/`` and never replace the thesis results.
4. ``extract``  keyword extraction on a sample of requirements per dataset, again only to
                measure the full token usage of Phase 1.

5. ``compare``  graph-based, baseline and no-clustering analyses run with the same model
                (``--model``, default ``gemini-3.1-pro-preview``) on both datasets, so that the
                three strategies are compared under identical conditions. Results go to
                ``results/model_comparison/<model>/``.

Every call that generates text records the usage metadata returned by the API
(prompt, output, thinking and total tokens) in ``results/token_runs/token_usage.json``.

Usage (from the repository root):
    python scripts/run_token_experiments.py count
    python scripts/run_token_experiments.py flat
    python scripts/run_token_experiments.py rerun
    python scripts/run_token_experiments.py extract --sample 30
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import google.generativeai as genai  # noqa: E402

from src.phase4.llm.analyzer import build_cluster_input  # noqa: E402

DATASETS = ["crowd", "test_reale"]
RUNS_DIR = ROOT / "results" / "token_runs"
USAGE_PATH = RUNS_DIR / "token_usage.json"


def load_config(dataset: str) -> dict[str, Any]:
    return json.loads((ROOT / "config" / f"{dataset}.json").read_text(encoding="utf-8"))


def load_records(dataset: str) -> list[dict[str, str]]:
    return json.loads((ROOT / "results" / dataset / "extraction.json").read_text(encoding="utf-8"))["records"]


def partition(dataset: str, report: str) -> dict[str, list[str]]:
    """Cluster -> requirement ids, read from the mapping stored with a Phase 4 result."""
    data = json.loads((ROOT / "results" / dataset / report).read_text(encoding="utf-8"))
    mapping: dict[str, list[str]] = {}
    for rid, cid in data["requirement_to_cluster_mapping"].items():
        mapping.setdefault(str(cid[0] if isinstance(cid, list) else cid), []).append(str(rid))
    return mapping


def phase4_message(system_prompt: str, clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The exact message built by src/phase4/llm/analyzer.py."""
    user_message = json.dumps(clusters, indent=2, ensure_ascii=False)
    return [{"role": "user", "parts": [{"text": system_prompt}, {"text": f"\n\n## INPUT CLUSTERS\n\n{user_message}"}]}]


def usage_of(response: Any) -> dict[str, int]:
    meta = response.usage_metadata
    prompt = int(getattr(meta, "prompt_token_count", 0) or 0)
    output = int(getattr(meta, "candidates_token_count", 0) or 0)
    total = int(getattr(meta, "total_token_count", 0) or 0)
    thoughts = int(getattr(meta, "thoughts_token_count", 0) or 0) or max(0, total - prompt - output)
    return {"prompt": prompt, "output": output, "thinking": thoughts, "total": total}


def parse_json(text: str) -> dict[str, Any]:
    """Same fallback cascade as the analyzer: whole text, code fence, greedy braces."""
    for candidate in (text, *(m.group(1) for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL))):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"error": "Failed to parse JSON from response", "raw_response": text}


def save_usage(key: str, value: Any) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(USAGE_PATH.read_text(encoding="utf-8")) if USAGE_PATH.exists() else {}
    data[key] = value
    USAGE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_phase4(dataset: str, mapping: dict[str, list[str]], output_path: Path, usage_key: str,
               model_name: str | None = None) -> None:
    config = load_config(dataset)
    if model_name:
        config["phase4_gemini_model"] = model_name
    system_prompt = (ROOT / config["phase4_system_prompt"]).read_text(encoding="utf-8")
    texts = {r["source_id"]: r["input"] for r in load_records(dataset)}
    clusters = build_cluster_input(mapping, texts)
    model = genai.GenerativeModel(config["phase4_gemini_model"])
    start = time.time()
    response = model.generate_content(
        phase4_message(system_prompt, clusters),
        generation_config=genai.types.GenerationConfig(
            temperature=config["phase4_gemini_temperature"],
            top_p=config["phase4_gemini_top_p"],
            max_output_tokens=config["phase4_gemini_max_tokens"],
        ),
    )
    result = parse_json(response.text)
    result["requirement_to_cluster_mapping"] = {rid: cid for cid, rids in mapping.items() for rid in rids}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    usage = usage_of(response) | {"seconds": round(time.time() - start, 1), "model": config["phase4_gemini_model"]}
    save_usage(usage_key, usage)
    print(f"[{usage_key}] {usage} -> {output_path.relative_to(ROOT)}")


def cmd_count() -> None:
    """Exact input/output tokens of the runs reported in the thesis (count_tokens is free)."""
    out: dict[str, Any] = {}
    for dataset in DATASETS:
        config = load_config(dataset)
        records = load_records(dataset)
        flash = genai.GenerativeModel(config["phase1_gemini_model"])
        p1_prompt = (ROOT / config["phase1_gemini_system_prompt"]).read_text(encoding="utf-8")
        system_tokens = flash.count_tokens(p1_prompt).total_tokens
        user_tokens = sum(flash.count_tokens(f"REQUIREMENT:\n{r['input']}").total_tokens for r in records)
        output_tokens = sum(flash.count_tokens(json.dumps(r["predicted_tags"].split(", "))).total_tokens for r in records)
        out[f"{dataset}/phase1"] = {
            "calls": len(records),
            "system_prompt_tokens_per_call": system_tokens,
            "input_tokens": system_tokens * len(records) + user_tokens,
            "output_tokens_approx": output_tokens,
        }
        pro = genai.GenerativeModel(config["phase4_gemini_model"])
        system_prompt = (ROOT / config["phase4_system_prompt"]).read_text(encoding="utf-8")
        texts = {r["source_id"]: r["input"] for r in records}
        for label, report in [("graph", "phase4_analysis.json"), ("baseline", "phase4_analysis_direct.json")]:
            clusters = build_cluster_input(partition(dataset, report), texts)
            produced = json.loads((ROOT / "results" / dataset / report).read_text(encoding="utf-8"))["clusters"]
            out[f"{dataset}/phase4_{label}"] = {
                "calls": 1,
                "input_tokens": pro.count_tokens(phase4_message(system_prompt, clusters)).total_tokens,
                "output_tokens_approx": pro.count_tokens(json.dumps({"clusters": produced}, indent=2, ensure_ascii=False)).total_tokens,
            }
        print(dataset, json.dumps({k: v for k, v in out.items() if k.startswith(dataset)}, indent=1))
    save_usage("thesis_runs_count_tokens", out)


def cmd_flat() -> None:
    for dataset in DATASETS:
        ids = [r["source_id"] for r in load_records(dataset)]
        run_phase4(dataset, {"0": ids}, ROOT / "results" / dataset / "phase4_analysis_flat.json", f"{dataset}/phase4_flat")


def cmd_rerun() -> None:
    for dataset in DATASETS:
        for label, report in [("graph", "phase4_analysis.json"), ("baseline", "phase4_analysis_direct.json")]:
            run_phase4(dataset, partition(dataset, report), RUNS_DIR / f"{dataset}_phase4_{label}_rerun.json", f"{dataset}/phase4_{label}_rerun")


def cmd_compare(model_name: str) -> None:
    """Graph-based, baseline and no-clustering analyses with the same model, for both datasets."""
    out_dir = ROOT / "results" / "model_comparison" / model_name
    for dataset in DATASETS:
        ids = [r["source_id"] for r in load_records(dataset)]
        strategies = [
            ("graph", partition(dataset, "phase4_analysis.json")),
            ("baseline", partition(dataset, "phase4_analysis_direct.json")),
            ("flat", {"0": ids}),
        ]
        for label, mapping in strategies:
            run_phase4(dataset, mapping, out_dir / f"{dataset}_{label}.json", f"{model_name}/{dataset}/{label}", model_name)


def cmd_extract(sample: int) -> None:
    for dataset in DATASETS:
        config = load_config(dataset)
        p1_prompt = (ROOT / config["phase1_gemini_system_prompt"]).read_text(encoding="utf-8")
        model = genai.GenerativeModel(config["phase1_gemini_model"], system_instruction=p1_prompt)
        records = load_records(dataset)
        step = max(1, len(records) // sample)
        chosen = records[::step][:sample]
        calls = []
        for r in chosen:
            response = model.generate_content(
                f"REQUIREMENT:\n{r['input']}",
                generation_config=genai.types.GenerationConfig(
                    temperature=config["phase1_gemini_temperature"],
                    top_p=config["phase1_gemini_top_p"],
                    max_output_tokens=config["phase1_gemini_max_tokens"],
                ),
            )
            calls.append(usage_of(response) | {"source_id": r["source_id"]})
        mean = {k: round(sum(c[k] for c in calls) / len(calls), 1) for k in ("prompt", "output", "thinking", "total")}
        summary = {"model": config["phase1_gemini_model"], "sample": len(calls), "total_requirements": len(records),
                   "mean_per_call": mean, "calls": calls}
        save_usage(f"{dataset}/phase1_sample", summary)
        print(f"[{dataset}/phase1_sample] mean per call {mean} over {len(calls)} calls")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["count", "flat", "rerun", "extract", "compare"])
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument("--model", default="gemini-3.1-pro-preview", help="Phase 4 model for the compare command")
    args = parser.parse_args()
    api_key = os.getenv("MODEL_API_KEY")
    if not api_key:
        raise SystemExit("MODEL_API_KEY not set in .env")
    genai.configure(api_key=api_key)
    if args.command == "count":
        cmd_count()
    elif args.command == "flat":
        cmd_flat()
    elif args.command == "rerun":
        cmd_rerun()
    elif args.command == "compare":
        cmd_compare(args.model)
    else:
        cmd_extract(args.sample)


if __name__ == "__main__":
    main()
