from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import umap
from node2vec import Node2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
)


def compute_node2vec_embeddings(
    graph: nx.Graph,
    dimensions: int = 64,
    walk_length: int = 30,
    num_walks: int = 200,
    workers: int = 1,
    window: int = 10,
    min_count: int = 1,
    batch_words: int = 4,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Compute Node2Vec embeddings for all nodes in the graph.

    workers defaults to 1 for full reproducibility: with workers > 1 the
    random walk order is non-deterministic even when seed is fixed, producing
    different embeddings across runs. Set workers > 1 only if runtime is a
    bottleneck and exact reproducibility is not required.
    """
    if graph.number_of_nodes() == 0:
        return {}

    node2vec = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
        weight_key="weight",
        seed=seed,
    )
    model = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)
    return {str(node): np.asarray(model.wv[str(node)], dtype=float) for node in graph.nodes()}


def _resolve_k(
    vectors: np.ndarray,
    requested_k: int | None,
    k_min: int,
    k_max: int,
    random_state: int,
) -> tuple[int, dict]:
    n_samples = int(vectors.shape[0])
    if n_samples < 2:
        return 1, {
            "selected_k": 1,
            "strategy": "single-node",
            "candidates": [],
        }

    if requested_k is not None:
        bounded_k = max(2, min(int(requested_k), n_samples)) if n_samples >= 2 else 1
        return bounded_k, {
            "selected_k": bounded_k,
            "strategy": "manual",
            "candidates": [],
        }

    start = max(2, min(k_min, n_samples))
    stop = min(max(start, k_max), n_samples)

    candidates = []
    for k in range(start, stop + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(vectors)
        inertia = float(km.inertia_)
        # davies_bouldin_score requires: 2 <= n_labels < n_samples
        davies_bouldin = float("inf")
        if k > 1 and k < n_samples:
            try:
                davies_bouldin = float(davies_bouldin_score(vectors, labels))
            except ValueError:
                davies_bouldin = float("inf")
        calinski_harabasz = float(calinski_harabasz_score(vectors, labels)) if k > 1 else 0.0
        candidates.append(
            {
                "k": int(k),
                "inertia": inertia,
                "davies_bouldin": davies_bouldin,
                "calinski_harabasz": calinski_harabasz,
            }
        )

    # Rank by combined signal: min Davies-Bouldin, max Calinski-Harabasz.
    candidates_sorted = sorted(
        candidates,
        key=lambda c: (
            c["davies_bouldin"],
            -c["calinski_harabasz"],
        ),
    )
    best = candidates_sorted[0]

    return int(best["k"]), {
        "selected_k": int(best["k"]),
        "strategy": "auto-combined",   # DB + CH
        "candidates": candidates,
    }


def _compute_umap(vectors: np.ndarray, random_state: int = 42) -> np.ndarray:
    n_samples = int(vectors.shape[0])
    if n_samples == 0:
        return np.zeros((0, 2), dtype=float)
    if n_samples == 1:
        return np.array([[0.0, 0.0]], dtype=float)

    n_neighbors = min(15, max(2, n_samples - 1))
    reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=n_neighbors)
    return reducer.fit_transform(vectors)


def _save_elbow_plot(candidates: list[dict], output_path: str) -> str | None:
    if not candidates:
        return None

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    k_values = [item["k"] for item in candidates]
    inertias = [item["inertia"] for item in candidates]
    davies_bouldins = [item["davies_bouldin"] for item in candidates]
    calinski_harabaszs = [item["calinski_harabasz"] for item in candidates]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(k_values, inertias, "bo-")
    axes[0].set_title("Elbow Method (Inertia)")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(k_values, davies_bouldins, "go-", label="Davies-Bouldin")
    axes[1].plot(k_values, calinski_harabaszs, "mo-", label="Calinski-Harabasz")
    axes[1].set_title("Cluster Quality Scores")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Score")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    return str(out_path)


def _build_requirement_feature_matrix(
    graph: nx.Graph,
    nodes: list[str],
    svd_dim: int,
) -> tuple[np.ndarray | None, dict]:
    requirement_sets: list[list[str]] = []
    unique_requirement_ids: set[str] = set()

    for node in nodes:
        raw = str(graph.nodes[node].get("requirement_ids", "")).strip()
        req_ids = [piece.strip() for piece in raw.split(",") if piece.strip()]
        requirement_sets.append(req_ids)
        unique_requirement_ids.update(req_ids)

    if not unique_requirement_ids:
        return None, {
            "enabled": False,
            "reason": "no_requirement_ids",
            "num_requirements": 0,
            "raw_dim": 0,
            "reduced_dim": 0,
        }

    sorted_requirements = sorted(unique_requirement_ids)
    req_index = {req_id: idx for idx, req_id in enumerate(sorted_requirements)}
    matrix = np.zeros((len(nodes), len(sorted_requirements)), dtype=float)

    for row_idx, req_ids in enumerate(requirement_sets):
        for req_id in req_ids:
            matrix[row_idx, req_index[req_id]] = 1.0

    max_rank = max(1, min(matrix.shape[0] - 1, matrix.shape[1] - 1))
    target_dim = max(1, min(int(svd_dim), max_rank))

    if matrix.shape[1] == target_dim:
        reduced = matrix
    else:
        svd = TruncatedSVD(n_components=target_dim, random_state=42)
        reduced = svd.fit_transform(matrix)

    return reduced, {
        "enabled": True,
        "num_requirements": int(len(sorted_requirements)),
        "raw_dim": int(matrix.shape[1]),
        "reduced_dim": int(reduced.shape[1]),
    }


def _normalize_block(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values

    means = values.mean(axis=0, keepdims=True)
    stds = values.std(axis=0, keepdims=True)
    stds = np.where(stds == 0, 1.0, stds)
    return (values - means) / stds


def _combine_clustering_features(
    node2vec_vectors: np.ndarray,
    requirement_vectors: np.ndarray | None,
    requirement_weight: float,
) -> np.ndarray:
    """
    Concatenate normalized Node2Vec embeddings with (optionally weighted)
    normalized requirement membership vectors.

    Returns node2vec_vectors alone when requirement_vectors is None or
    weight is 0. Otherwise returns a horizontally stacked matrix where the
    requirement block is scaled by requirement_weight before concatenation.
    """
    left = _normalize_block(node2vec_vectors)
    if requirement_vectors is None:
        return left

    weight = max(0.0, float(requirement_weight))
    if weight == 0.0:
        return left

    right = _normalize_block(requirement_vectors)
    # Scale the requirement block so its contribution to KMeans distance is
    # proportional to requirement_weight relative to the node2vec block.
    return np.hstack([left, right * weight])


def run_clustering_phase(
    graph: nx.Graph,
    output_json_path: str,
    output_elbow_plot_path: str | None = None,
    embedding_dim: int = 64,
    requested_k: int | None = None,
    k_min: int = 2,
    k_max: int = 15,
    random_state: int = 42,
    node2vec_walk_length: int = 30,
    node2vec_num_walks: int = 200,
    node2vec_workers: int = 1,
    requirement_feature_weight: float = 0.5,
    requirement_svd_dim: int = 16,
    isolated_nodes_removed: int = 0,
) -> tuple[dict, nx.Graph]:
    embeddings = compute_node2vec_embeddings(
        graph,
        dimensions=embedding_dim,
        walk_length=node2vec_walk_length,
        num_walks=node2vec_num_walks,
        workers=node2vec_workers,
        seed=random_state,
    )

    nodes = list(embeddings.keys())
    if not nodes:
        payload = {
            "clustering": {
                "enabled": True,
                "reason": "no_nodes",
                "selected_k": 0,
                "num_nodes": 0,
                "embedding_dim": embedding_dim,
                "k_selection": {"selected_k": 0, "strategy": "no-nodes", "candidates": []},
            },
            "nodes": [],
        }
        out_json = Path(output_json_path)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return payload, graph

    node2vec_vectors = np.array([embeddings[node] for node in nodes], dtype=float)
    requirement_vectors, requirement_feature_info = _build_requirement_feature_matrix(
        graph=graph,
        nodes=nodes,
        svd_dim=requirement_svd_dim,
    )
    vectors = _combine_clustering_features(
        node2vec_vectors=node2vec_vectors,
        requirement_vectors=requirement_vectors,
        requirement_weight=requirement_feature_weight,
    )
    selected_k, k_selection = _resolve_k(
        vectors=vectors,
        requested_k=requested_k,
        k_min=k_min,
        k_max=k_max,
        random_state=random_state,
    )

    if selected_k <= 1:
        labels = np.zeros(len(nodes), dtype=int)
        inertia = 0.0
    else:
        kmeans = KMeans(n_clusters=selected_k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(vectors)
        inertia = float(kmeans.inertia_)

    umap_coords = _compute_umap(vectors, random_state=random_state)

    node_records = []
    cluster_to_requirements: dict[int, set[str]] = {i: set() for i in range(selected_k)} if selected_k > 0 else {}

    for idx, node in enumerate(nodes):
        cluster_id = int(labels[idx])
        umap_x = float(umap_coords[idx][0])
        umap_y = float(umap_coords[idx][1])

        graph.nodes[node]["cluster_id"] = cluster_id
        graph.nodes[node]["umap_x"] = umap_x
        graph.nodes[node]["umap_y"] = umap_y

        raw = str(graph.nodes[node].get("requirement_ids", "")).strip()
        req_ids = [piece.strip() for piece in raw.split(",") if piece.strip()]
        if cluster_id in cluster_to_requirements:
            cluster_to_requirements[cluster_id].update(req_ids)

        node_records.append(
            {
                "keyword": node,
                "cluster_id": cluster_id,
                "umap_x": umap_x,
                "umap_y": umap_y,
            }
        )

    cluster_requirements_mapping = {
        str(cluster_id): sorted(list(req_ids))
        for cluster_id, req_ids in cluster_to_requirements.items()
    }

    requirement_to_cluster_mapping: dict[str, list[int]] = {}
    for cluster_id, req_ids in cluster_to_requirements.items():
        for req_id in req_ids:
            requirement_to_cluster_mapping.setdefault(str(req_id), [])
            requirement_to_cluster_mapping[str(req_id)].append(int(cluster_id))

    requirement_to_cluster_mapping = {
        req_id: sorted(clusters)
        for req_id, clusters in requirement_to_cluster_mapping.items()
    }

    elbow_plot = (
        _save_elbow_plot(k_selection.get("candidates", []), output_elbow_plot_path)
        if output_elbow_plot_path
        else None
    )

    payload = {
        "clustering": {
            "enabled": True,
            "selected_k": int(selected_k),
            "num_nodes": int(len(nodes)),
            "embedding_dim": int(embedding_dim),
            "effective_feature_dim": int(vectors.shape[1]),
            "inertia": float(inertia),
            "k_selection": k_selection,
            "elbow_plot": elbow_plot,
            "requirement_features": {
                **requirement_feature_info,
                "weight": float(requirement_feature_weight),
            },
            "isolated_nodes_removed": int(isolated_nodes_removed),
        },
        "cluster_requirements_mapping": cluster_requirements_mapping,
        "requirement_to_cluster_mapping": requirement_to_cluster_mapping,
        "nodes": node_records,
    }

    out_json = Path(output_json_path)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    return payload, graph