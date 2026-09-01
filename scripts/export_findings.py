"""Export every Phase 4 finding together with the requirements it cites.

Produces the worksheet used to verify the findings one by one:

  results/verification/findings.json  every finding, its claim, the full text of the
                                      requirements it cites, and the verdict recorded
                                      in scripts/verdicts/ (if any)
  results/verification/findings.txt   the same content as a plain-text listing

Both cover the four analysis runs: the graph-based partition and the direct-clustering
baseline, on each of the two datasets.

Usage (from anywhere):
    python scripts/export_findings.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CATEGORIES = ["duplicates", "inconsistencies", "ambiguities", "gaps", "dependencies"]

RUNS = [
    # dataset folder, analysis file, verdict file, dataset label, partition label
    ("crowd", "phase4_analysis.json", "uw_graph", "Uniwhere", "Graph"),
    ("crowd", "phase4_analysis_direct.json", "uw_base", "Uniwhere", "Baseline"),
    ("test_reale", "phase4_analysis.json", "v2i_graph", "V2I", "Graph"),
    ("test_reale", "phase4_analysis_direct.json", "v2i_base", "V2I", "Baseline"),
]


def requirement_texts(dataset: str) -> dict[str, str]:
    data = json.loads((ROOT / "results" / dataset / "extraction.json").read_text(encoding="utf-8"))
    return {str(r["source_id"]): r["input"].strip() for r in data["records"]}


def load_verdicts(name: str) -> dict[str, str]:
    path = ROOT / "scripts" / "verdicts" / f"{name}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))["verdicts"]


def describe(category: str, item: dict) -> tuple[list[str], str]:
    """Return the requirement ids a finding refers to and its claim, per category."""
    if category in ("duplicates", "inconsistencies"):
        return [str(i) for i in item.get("requirements", [])], item.get("reason", "")
    if category == "ambiguities":
        return [str(item.get("requirement", ""))], f"{item.get('issue','')} -> {item.get('suggestion','')}"
    if category == "gaps":
        return [str(i) for i in (item.get("related_to") or [])], item.get("description", "")
    return [str(item.get("from", "")), str(item.get("to", ""))], item.get("reason", "")


def collect() -> list[dict]:
    rows: list[dict] = []
    for dataset, analysis_file, verdict_file, label, partition in RUNS:
        texts = requirement_texts(dataset)
        verdicts = load_verdicts(verdict_file)
        analysis = json.loads((ROOT / "results" / dataset / analysis_file).read_text(encoding="utf-8"))

        for cluster in analysis["clusters"]:
            cid = str(cluster["cluster_id"])
            for category in CATEGORIES:
                for index, item in enumerate(cluster.get(category) or [], 1):
                    key = f"C{cid}|{category}|{index}"
                    verdict, _, reason = verdicts.get(key, "UNJUDGED").partition("|")
                    ids, claim = describe(category, item)
                    rows.append({
                        "dataset": label,
                        "partition": partition,
                        "cluster": cid,
                        "clusterLabel": cluster.get("label", ""),
                        "score": cluster.get("quality_score"),
                        "category": category,
                        "n": index,
                        "type": item.get("type", ""),
                        "claim": claim,
                        "verdict": verdict,
                        "reason": reason,
                        "reqs": [{"id": i, "text": texts.get(i, "<missing>")} for i in ids if i],
                    })
    return rows


def as_text(rows: list[dict]) -> str:
    out: list[str] = []
    last = None
    for r in rows:
        key = (r["dataset"], r["partition"], r["cluster"])
        if key != last:
            last = key
            out.append("")
            out.append("#" * 78)
            out.append(f"# {r['dataset']} | {r['partition']} | CLUSTER {r['cluster']}"
                       f" | {r['clusterLabel']} | coherence={r['score']}")
            out.append("#" * 78)
        head = f"[{r['category']} #{r['n']}]"
        if r["type"]:
            head += f" type={r['type']}"
        out.append("")
        out.append(f"{head}  -> {r['verdict']}")
        out.append(f"   CLAIM: {r['claim']}")
        for req in r["reqs"]:
            out.append(f"   R{req['id']}: {req['text']}")
        if r["reason"]:
            out.append(f"   WHY:   {r['reason']}")
    return "\n".join(out).lstrip("\n")


def main() -> None:
    rows = collect()
    out_dir = ROOT / "results" / "verification"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "findings.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "findings.txt").write_text(as_text(rows), encoding="utf-8")

    judged = sum(1 for r in rows if r["verdict"] != "UNJUDGED")
    print(f"exported {len(rows)} findings ({judged} judged) to {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
