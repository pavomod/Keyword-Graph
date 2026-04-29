from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
)
from sentence_transformers import SentenceTransformer


def _coerce_cluster_ids(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return [int(item) for item in value if item is not None]
    return [int(value)]


def _compute_umap(vectors: np.ndarray, random_state: int = 42) -> np.ndarray:
    n_samples = int(vectors.shape[0])
    if n_samples == 0:
        return np.zeros((0, 2), dtype=float)
    if n_samples == 1:
        return np.array([[0.0, 0.0]], dtype=float)

    n_neighbors = min(15, max(2, n_samples - 1))
    reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=n_neighbors)
    return reducer.fit_transform(vectors)


def build_requirement_texts(df_data: list[dict]) -> list[str]:
    """Build user-story text from role, feature, benefit columns."""
    texts = []
    for row in df_data:
        role = str(row.get("role", "user")).strip()
        feature = str(row.get("feature", "")).strip()
        benefit = str(row.get("benefit", "")).strip()
        texts.append(f"As a {role}, I want {feature} so that {benefit}")
    return texts


def cluster_requirements_directly(
    requirement_texts: list[str],
    selected_k: int | None = None,
    k_min: int = 2,
    k_max: int = 15,
    embedding_model: str = "all-MiniLM-L6-v2",
    random_state: int = 42,
) -> tuple[np.ndarray, list[int], dict]:
    """
    Cluster requirements directly using their sentence embeddings.

    Returns:
        (embeddings, labels, diagnostics)
    """
    if not requirement_texts:
        return np.zeros((0, 384)), [], {"error": "no_requirements"}

    encoder = SentenceTransformer(embedding_model)
    embeddings = np.asarray(encoder.encode(requirement_texts, convert_to_tensor=False), dtype=float)
    n_samples = embeddings.shape[0]

    if selected_k is not None:
        k = max(1, min(selected_k, n_samples))
        k_selection: dict = {"selected_k": int(k), "strategy": "manual", "candidates": []}
    else:
        start = max(2, min(k_min, n_samples))
        stop = min(max(start, k_max), n_samples)

        candidates = []
        for k_candidate in range(start, stop + 1):
            km = KMeans(n_clusters=k_candidate, random_state=random_state, n_init=10)
            labels_tmp = km.fit_predict(embeddings)
            inertia = float(km.inertia_)
            davies_bouldin = float(davies_bouldin_score(embeddings, labels_tmp)) if k_candidate > 1 else float("inf")
            calinski_harabasz = float(calinski_harabasz_score(embeddings, labels_tmp)) if k_candidate > 1 else 0.0
            candidates.append({
                "k": k_candidate,
                "inertia": inertia,
                "davies_bouldin": davies_bouldin,
                "calinski_harabasz": calinski_harabasz,
            })

        best = sorted(
            candidates,
            key=lambda c: (c["davies_bouldin"], -c["calinski_harabasz"]),
        )[0]
        k = int(best["k"])
        k_selection = {"selected_k": int(k), "strategy": "auto-combined", "candidates": candidates}

    k = max(1, k)

    if k == 1:
        labels_arr = np.zeros(n_samples, dtype=int)
        inertia = davies_bouldin = calinski_harabasz = -1.0
    else:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels_arr = kmeans.fit_predict(embeddings)
        inertia = float(kmeans.inertia_)
        davies_bouldin = float(davies_bouldin_score(embeddings, labels_arr)) if k > 1 else -1.0
        calinski_harabasz = float(calinski_harabasz_score(embeddings, labels_arr)) if k > 1 else -1.0

    diagnostics = {
        "num_requirements": n_samples,
        "selected_k": int(k),
        "k_selection": k_selection,
        "inertia": float(inertia),
        "davies_bouldin": float(davies_bouldin),
        "calinski_harabasz": float(calinski_harabasz),
    }

    return embeddings, labels_arr.tolist(), diagnostics


def build_direct_clustering_report(
    requirement_texts: list[str],
    requirement_ids: list[str],
    direct_clustering_result: tuple[np.ndarray, list[int], dict],
    output_path: str,
) -> str:
    embeddings, direct_labels, direct_diagnostics = direct_clustering_result
    if len(requirement_texts) != len(requirement_ids):
        raise ValueError("requirement_texts and requirement_ids must have same length")

    cluster_to_requirements: dict[int, set[str]] = {}
    cluster_to_texts: dict[int, list[str]] = {}
    node_records = []

    umap_coords = _compute_umap(embeddings) if len(direct_labels) else np.zeros((0, 2), dtype=float)

    for idx, (requirement_id, text, cluster_id) in enumerate(zip(requirement_ids, requirement_texts, direct_labels)):
        cluster_id = int(cluster_id)
        requirement_id_text = str(requirement_id)
        cluster_to_requirements.setdefault(cluster_id, set()).add(requirement_id_text)
        cluster_to_texts.setdefault(cluster_id, []).append(text)

        x_coord = float(umap_coords[idx][0]) if idx < len(umap_coords) else 0.0
        y_coord = float(umap_coords[idx][1]) if idx < len(umap_coords) else 0.0
        node_records.append({
            "requirement_id": requirement_id_text,
            "input": text,
            "cluster_id": cluster_id,
            "umap_x": x_coord,
            "umap_y": y_coord,
        })

    cluster_requirements_mapping = {
        str(cid): sorted(rids) for cid, rids in cluster_to_requirements.items()
    }
    requirement_to_cluster_mapping = {
        rid: [cid]
        for cid, rids in cluster_to_requirements.items()
        for rid in rids
    }

    report = {
        "clustering": {
            "enabled": True,
            "mode": "direct_requirement_clustering",
            "selected_k": int(direct_diagnostics.get("selected_k", 0)),
            "num_nodes": len(requirement_ids),
            "embedding_dim": int(embeddings.shape[1]) if len(embeddings) else 0,
            "effective_feature_dim": int(embeddings.shape[1]) if len(embeddings) else 0,
            "inertia": float(direct_diagnostics.get("inertia", -1)),
            "davies_bouldin": float(direct_diagnostics.get("davies_bouldin", -1)),
            "calinski_harabasz": float(direct_diagnostics.get("calinski_harabasz", -1)),
            "k_selection": direct_diagnostics.get("k_selection", {"selected_k": 0, "strategy": "manual", "candidates": []}),
            "elbow_plot": None,
            "requirement_features": {
                "enabled": False,
                "reason": "direct_requirement_clustering",
                "num_requirements": len(requirement_ids),
                "raw_dim": int(embeddings.shape[1]) if len(embeddings) else 0,
                "reduced_dim": int(embeddings.shape[1]) if len(embeddings) else 0,
                "weight": 0.0,
            },
        },
        "cluster_requirements_mapping": cluster_requirements_mapping,
        "requirement_to_cluster_mapping": requirement_to_cluster_mapping,
        "nodes": node_records,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return str(out_path)


def compare_clustering_approaches(
    keyword_graph_clustering: dict,
    direct_clustering_result: tuple[np.ndarray, list[int], dict],
    requirement_ids: list[str],
    output_path: str,
) -> str:
    """
    Compare keyword-graph clustering against direct requirement embedding clustering.

    Agreement is measured with Adjusted Rand Index (ARI) and Normalized Mutual
    Information (NMI) — both are invariant to cluster label permutations, which
    makes them correct for comparing two independent partitions.

    The old agreement_ratio (direct cluster ID ∈ keyword-graph cluster list) was
    wrong because cluster IDs are arbitrary integers with no cross-method meaning.

    ARI interpretation:
        1.0  → perfect agreement
        ~0.0 → random agreement
        <0   → worse than random (very different structures)

    NMI interpretation:
        1.0  → identical clustering
        0.0  → completely independent clusterings
    """
    _, direct_labels, direct_diagnostics = direct_clustering_result

    requirement_to_cluster_kg = keyword_graph_clustering.get("requirement_to_cluster_mapping", {})
    cluster_id_set: set[int] = set()
    normalized_kg_mapping: dict[str, list[int]] = {}

    for req_id, cluster_values in requirement_to_cluster_kg.items():
        clusters = _coerce_cluster_ids(cluster_values)
        normalized_kg_mapping[str(req_id)] = sorted(set(clusters))
        cluster_id_set.update(clusters)

    kg_k = len(cluster_id_set)
    direct_k = direct_diagnostics.get("selected_k", 0)

    # Build aligned label vectors over the shared requirement_ids.
    # For requirements with multi-cluster membership in the keyword graph,
    # use the first (lowest) cluster id so we produce a flat partition
    # comparable to the direct approach.
    kg_label_vec: list[int] = []
    direct_label_vec: list[int] = []
    multi_membership_count = 0
    missing_in_kg = 0

    for idx, req_id in enumerate(requirement_ids):
        kg_clusters = normalized_kg_mapping.get(str(req_id), [])
        direct_cluster = direct_labels[idx] if idx < len(direct_labels) else -1

        if not kg_clusters:
            missing_in_kg += 1
            continue  # skip requirements not mapped in keyword graph

        if len(kg_clusters) > 1:
            multi_membership_count += 1

        kg_label_vec.append(kg_clusters[0])
        direct_label_vec.append(direct_cluster)

    total_compared = len(kg_label_vec)

    if total_compared >= 2:
        ari = float(adjusted_rand_score(kg_label_vec, direct_label_vec))
        nmi = float(normalized_mutual_info_score(kg_label_vec, direct_label_vec, average_method="arithmetic"))
    else:
        ari = -1.0
        nmi = -1.0

    report = {
        "comparison": {
            "method": "keyword_graph_vs_direct_embedding",
            "num_requirements": len(requirement_ids),
            "total_compared": total_compared,
            "missing_in_keyword_graph": missing_in_kg,
        },
        "keyword_graph_approach": {
            "num_clusters": kg_k,
            "requirements_mapped": len(requirement_to_cluster_kg),
            "multi_cluster_requirements": multi_membership_count,
        },
        "direct_embedding_approach": {
            "num_clusters": direct_k,
            "inertia": direct_diagnostics.get("inertia", -1),
            "davies_bouldin": direct_diagnostics.get("davies_bouldin", -1),
            "calinski_harabasz": direct_diagnostics.get("calinski_harabasz", -1),
        },
        "agreement": {
            # ARI and NMI are label-permutation invariant — correct metrics for
            # comparing two independent clustering partitions.
            "adjusted_rand_index": ari,
            "normalized_mutual_info": nmi,
            "interpretation": {
                "ari": "1.0=perfect, ~0.0=random, <0=anti-correlated",
                "nmi": "1.0=identical, 0.0=independent",
            },
        },
        "notes": (
            "Multi-cluster requirements in the keyword graph are resolved to their "
            "first (lowest) cluster ID for the purpose of ARI/NMI computation. "
            "Different cluster counts between approaches indicate different granularity."
        ),
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return str(out_path)