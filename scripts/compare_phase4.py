from __future__ import annotations

import json
from pathlib import Path

CATS = ["duplicates", "inconsistencies", "ambiguities", "gaps", "dependencies"]


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stats(analysis: dict, mapping: dict[str, list[str]]) -> dict:
    clusters = analysis.get("clusters", [])
    tot = {c: 0 for c in CATS}
    scores, violations, cited = [], 0, 0
    dup_pairs = 0
    for c in clusters:
        allowed = set(mapping.get(str(c["cluster_id"]), []))
        ids = set()
        for cat in CATS:
            items = c.get(cat, [])
            tot[cat] += len(items)
            for it in items:
                if cat == "duplicates":
                    dup_pairs += max(0, len(it.get("requirements", [])) - 1)
                for k in ("requirements", "related_to"):
                    ids.update(it.get(k, []) or [])
                for k in ("requirement", "from", "to"):
                    if it.get(k):
                        ids.add(it[k])
        cited += len(ids)
        violations += len([i for i in ids if str(i) not in allowed])
        try:
            scores.append(float(c["quality_score"]))
        except (KeyError, ValueError, TypeError):
            pass
    return {
        "clusters": len(clusters),
        **tot,
        "total_findings": sum(tot.values()),
        "dup_pairs": dup_pairs,
        "cited_ids": cited,
        "violations": violations,
        "avg_score": round(sum(scores) / len(scores), 1) if scores else None,
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
    }


for ds, label in [("crowd", "Uniwhere"), ("test_reale", "V2I")]:
    print("=" * 74)
    print(f"### {label}")
    print("=" * 74)
    rows = {}
    for kind, afile, cfile in [
        ("GRAPH ", "phase4_analysis.json", "clustering_report.json"),
        ("DIRECT", "phase4_analysis_direct.json", "direct_clustering_report.json"),
    ]:
        a = load(f"results/{ds}/{afile}")
        m = load(f"results/{ds}/{cfile}")["cluster_requirements_mapping"]
        rows[kind] = stats(a, m)
    hdr = ["clusters", "duplicates", "inconsistencies", "ambiguities", "gaps",
           "dependencies", "total_findings", "dup_pairs", "cited_ids",
           "violations", "avg_score", "min_score", "max_score"]
    print(f"{'metric':<18}{'GRAPH':>12}{'DIRECT':>12}")
    for h in hdr:
        print(f"{h:<18}{str(rows['GRAPH '][h]):>12}{str(rows['DIRECT'][h]):>12}")
    print()
