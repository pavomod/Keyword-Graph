from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sentence_transformers import SentenceTransformer
from src.clustering.clustering import generate_cluster_labels


_DIRECT_LABEL_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "i",
    "if",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "this",
    "to",
    "want",
    "with",
    "user",
    "system",
    "feature",
    "home",
    "page",
    "screen",
    "button",
    "menu",
    "view",
    "panel",
    "form",
    "field",
    "item",
    "data",
}


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


def _build_cluster_keywords_from_texts(texts: list[str], max_terms: int = 20) -> list[str]:
    counts: dict[str, int] = {}
    for text in texts:
        for token in re.findall(r"[a-zA-Z][a-zA-Z\-]+", str(text).lower()):
            if len(token) <= 2 or token in _DIRECT_LABEL_STOPWORDS:
                continue
            counts[token] = counts.get(token, 0) + 1

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ordered[:max_terms]]


def build_requirement_texts(df_data: list[dict]) -> list[str]:
    """Build text representation of requirements from role, feature, benefit columns."""
    texts = []
    for row in df_data:
        role = str(row.get("role", "user")).strip()
        feature = str(row.get("feature", "")).strip()
        benefit = str(row.get("benefit", "")).strip()
        
        text = f"As a {role}, I want {feature} so that {benefit}"
        texts.append(text)
    
    return texts


def cluster_requirements_directly(
    requirement_texts: list[str],
    selected_k: int = None,
    k_min: int = 2,
    k_max: int = 15,
    embedding_model: str = "all-MiniLM-L6-v2",
    random_state: int = 42,
) -> tuple[np.ndarray, list[int], dict]:
    """
    Cluster requirements directly using their text embeddings.
    
    Args:
        requirement_texts: List of requirement text descriptions
        selected_k: If provided, use this k. Otherwise, auto-select by silhouette
        k_min: Minimum k for automatic selection
        k_max: Maximum k for automatic selection
        embedding_model: Sentence-Transformers model name
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (embeddings, labels, diagnostics)
    """
    if not requirement_texts:
        return np.zeros((0, 384)), [], {"error": "no_requirements"}
    
    # Encode requirement texts
    encoder = SentenceTransformer(embedding_model)
    embeddings = encoder.encode(requirement_texts, convert_to_tensor=False)
    embeddings = np.asarray(embeddings, dtype=float)
    
    n_samples = embeddings.shape[0]
    
    # Determine k
    if selected_k is not None:
        k = max(1, min(selected_k, n_samples))
        k_selection = {
            "selected_k": int(k),
            "strategy": "manual",
            "candidates": [],
        }
    else:
        # Auto-select k
        start = max(2, min(k_min, n_samples))
        stop = min(max(start, k_max), n_samples)
        
        candidates = []
        for k_candidate in range(start, stop + 1):
            km = KMeans(n_clusters=k_candidate, random_state=random_state, n_init=10)
            labels = km.fit_predict(embeddings)
            inertia = float(km.inertia_)
            silhouette = float(silhouette_score(embeddings, labels)) if k_candidate < n_samples else -1.0
            candidates.append({
                "k": k_candidate,
                "inertia": inertia,
                "silhouette": silhouette,
            })
        
        best = max(candidates, key=lambda item: item["silhouette"])
        k = int(best["k"])
        k_selection = {
            "selected_k": int(k),
            "strategy": "auto-silhouette",
            "candidates": candidates,
        }
    
    # Run final clustering
    if k < 1:
        k = 1
    
    if k == 1:
        labels = np.zeros(n_samples, dtype=int)
        inertia = 0.0
        silhouette = -1.0
        davies_bouldin = -1.0
        calinski_harabasz = -1.0
    else:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        inertia = float(kmeans.inertia_)
        silhouette = float(silhouette_score(embeddings, labels)) if k < n_samples else -1.0
        davies_bouldin = float(davies_bouldin_score(embeddings, labels)) if k > 1 else -1.0
        calinski_harabasz = float(calinski_harabasz_score(embeddings, labels)) if k > 1 else -1.0
    
    diagnostics = {
        "num_requirements": n_samples,
        "selected_k": int(k),
        "k_selection": k_selection,
        "inertia": float(inertia),
        "silhouette": float(silhouette),
        "davies_bouldin": float(davies_bouldin),
        "calinski_harabasz": float(calinski_harabasz),
    }
    
    return embeddings, labels.tolist(), diagnostics


def build_direct_clustering_report(
    requirement_texts: list[str],
    requirement_ids: list[str],
    direct_clustering_result: tuple[np.ndarray, list[int], dict],
    output_path: str,
    model_dir: str | None = None,
    base_model_name: str = "google/flan-t5-base",
) -> str:
    embeddings, direct_labels, direct_diagnostics = direct_clustering_result
    if len(requirement_texts) != len(requirement_ids):
        raise ValueError("requirement_texts and requirement_ids must have same length")

    cluster_to_requirements: dict[int, set[str]] = {}
    cluster_to_texts: dict[int, list[str]] = {}
    node_records = []

    umap_coords = _compute_umap(embeddings, random_state=42) if len(direct_labels) else np.zeros((0, 2), dtype=float)

    for idx, (requirement_id, text, cluster_id) in enumerate(zip(requirement_ids, requirement_texts, direct_labels)):
        cluster_id = int(cluster_id)
        requirement_id_text = str(requirement_id)
        cluster_to_requirements.setdefault(cluster_id, set()).add(requirement_id_text)
        cluster_to_texts.setdefault(cluster_id, []).append(text)

        x_coord = float(umap_coords[idx][0]) if idx < len(umap_coords) else 0.0
        y_coord = float(umap_coords[idx][1]) if idx < len(umap_coords) else 0.0
        node_records.append(
            {
                "requirement_id": requirement_id_text,
                "input": text,
                "cluster_id": cluster_id,
                "umap_x": x_coord,
                "umap_y": y_coord,
            }
        )

    cluster_requirements_mapping = {
        str(cluster_id): sorted(requirement_ids_set)
        for cluster_id, requirement_ids_set in cluster_to_requirements.items()
    }
    requirement_to_cluster_mapping = {
        requirement_id: [cluster_id]
        for cluster_id, requirement_ids_set in cluster_to_requirements.items()
        for requirement_id in requirement_ids_set
    }

    cluster_keywords = {
        cluster_id: _build_cluster_keywords_from_texts(texts)
        for cluster_id, texts in cluster_to_texts.items()
    }
    cluster_requirements_for_labels = {
        cluster_id: sorted(requirement_ids_set)
        for cluster_id, requirement_ids_set in cluster_to_requirements.items()
    }
    cluster_labels, label_fallback_flags = generate_cluster_labels(
        clusters_keywords=cluster_keywords,
        cluster_requirements=cluster_requirements_for_labels,
        model_dir=model_dir,
        base_model_name=base_model_name,
    )

    report = {
        "clustering": {
            "enabled": True,
            "mode": "direct_requirement_clustering",
            "selected_k": int(direct_diagnostics.get("selected_k", 0)),
            "num_nodes": len(requirement_ids),
            "embedding_dim": int(embeddings.shape[1]) if len(embeddings) else 0,
            "effective_feature_dim": int(embeddings.shape[1]) if len(embeddings) else 0,
            "inertia": float(direct_diagnostics.get("inertia", -1)),
            "silhouette": float(direct_diagnostics.get("silhouette", -1)),
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
        "cluster_labels": {str(cluster_id): label for cluster_id, label in cluster_labels.items()},
        "cluster_labeling": {
            "enabled": True,
            "used_fallback_clusters": sorted(
                [int(cluster_id) for cluster_id, used_fallback in label_fallback_flags.items() if used_fallback]
            ),
            "per_cluster": {
                str(cluster_id): {
                    "label": label,
                    "used_fallback": bool(label_fallback_flags.get(cluster_id, False)),
                }
                for cluster_id, label in cluster_labels.items()
            },
        },
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
    Compare clustering results from keyword graph approach vs direct requirement clustering.
    
    Args:
        keyword_graph_clustering: Dictionary from clustering report with requirement_to_cluster_mapping
        direct_clustering_result: Tuple of (embeddings, labels, diagnostics) from direct clustering
        requirement_ids: List of requirement IDs in the same order as direct clustering
        output_path: Path to save comparison report
    
    Returns:
        Path to saved report
    """
    _, direct_labels, direct_diagnostics = direct_clustering_result
    
    # Build keyword graph clustering info
    requirement_to_cluster_kg = keyword_graph_clustering.get("requirement_to_cluster_mapping", {})
    cluster_id_set: set[int] = set()
    normalized_keyword_graph_mapping: dict[str, list[int]] = {}

    for requirement_id, cluster_values in requirement_to_cluster_kg.items():
        clusters = _coerce_cluster_ids(cluster_values)
        normalized_keyword_graph_mapping[str(requirement_id)] = sorted(set(clusters))
        cluster_id_set.update(clusters)
    
    # Count requirements in each approach
    num_requirements = len(requirement_ids)
    kg_k = len(cluster_id_set)
    direct_k = direct_diagnostics.get("selected_k", 0)
    
    # Compute agreement using many-to-many keyword-graph memberships.
    agreement_count = 0
    mismatch_count = 0
    multi_membership_count = 0
    
    for idx, req_id in enumerate(requirement_ids):
        kg_clusters = normalized_keyword_graph_mapping.get(str(req_id), [])
        direct_cluster = direct_labels[idx] if idx < len(direct_labels) else -1
        
        if len(kg_clusters) > 1:
            multi_membership_count += 1

        if kg_clusters and direct_cluster >= 0:
            if direct_cluster in kg_clusters:
                agreement_count += 1
            else:
                mismatch_count += 1
    
    total_compared = agreement_count + mismatch_count
    agreement_ratio = agreement_count / total_compared if total_compared > 0 else 0.0
    
    report = {
        "comparison": {
            "method": "keyword_graph_vs_direct_embedding",
            "num_requirements": num_requirements,
        },
        "keyword_graph_approach": {
            "num_clusters": kg_k,
            "requirements_mapped": len(requirement_to_cluster_kg),
            "multi_cluster_requirements": multi_membership_count,
        },
        "direct_embedding_approach": {
            "num_clusters": direct_k,
            "inertia": direct_diagnostics.get("inertia", -1),
            "silhouette": direct_diagnostics.get("silhouette", -1),
            "davies_bouldin": direct_diagnostics.get("davies_bouldin", -1),
            "calinski_harabasz": direct_diagnostics.get("calinski_harabasz", -1),
        },
        "cross_approach_agreement": {
            "total_compared": total_compared,
            "agreement_count": agreement_count,
            "mismatch_count": mismatch_count,
            "agreement_ratio": float(agreement_ratio),
        },
        "interpretation": (
            "High agreement ratio indicates both approaches produce similar clusters. "
            "Lower agreement suggests keyword graph captures different semantic structures. "
            "Different cluster counts (k) may indicate different granularity levels."
        ),
    }
    
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    
    return str(out_path)
