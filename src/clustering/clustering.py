from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import umap
from node2vec import Node2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def compute_node2vec_embeddings(
    graph: nx.Graph,
    dimensions: int = 64,
    walk_length: int = 30,
    num_walks: int = 200,
    workers: int = 4,
    window: int = 10,
    min_count: int = 1,
    batch_words: int = 4,
    seed: int = 42,
) -> dict[str, np.ndarray]:
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
        silhouette = float(silhouette_score(vectors, labels)) if k < n_samples else -1.0
        candidates.append(
            {
                "k": int(k),
                "inertia": inertia,
                "silhouette": silhouette,
            }
        )

    best = max(candidates, key=lambda item: item["silhouette"])
    return int(best["k"]), {
        "selected_k": int(best["k"]),
        "strategy": "auto-silhouette",
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
    silhouettes = [item["silhouette"] for item in candidates]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(k_values, inertias, "bo-")
    plt.title("Elbow Method")
    plt.xlabel("k")
    plt.ylabel("Inertia")

    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouettes, "ro-")
    plt.title("Silhouette Score")
    plt.xlabel("k")
    plt.ylabel("Score")

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    return str(out_path)


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
    node2vec_workers: int = 4,
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

    vectors = np.array([embeddings[node] for node in nodes], dtype=float)
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
        silhouette = -1.0
    else:
        kmeans = KMeans(n_clusters=selected_k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(vectors)
        inertia = float(kmeans.inertia_)
        silhouette = float(silhouette_score(vectors, labels)) if selected_k < len(nodes) else -1.0

    umap_coords = _compute_umap(vectors, random_state=random_state)

    node_records = []
    for idx, node in enumerate(nodes):
        cluster_id = int(labels[idx])
        umap_x = float(umap_coords[idx][0])
        umap_y = float(umap_coords[idx][1])

        graph.nodes[node]["cluster_id"] = cluster_id
        graph.nodes[node]["umap_x"] = umap_x
        graph.nodes[node]["umap_y"] = umap_y

        node_records.append(
            {
                "keyword": node,
                "cluster_id": cluster_id,
                "umap_x": umap_x,
                "umap_y": umap_y,
            }
        )

    elbow_plot = _save_elbow_plot(k_selection.get("candidates", []), output_elbow_plot_path) if output_elbow_plot_path else None

    payload = {
        "clustering": {
            "enabled": True,
            "selected_k": int(selected_k),
            "num_nodes": int(len(nodes)),
            "embedding_dim": int(embedding_dim),
            "inertia": float(inertia),
            "silhouette": float(silhouette),
            "k_selection": k_selection,
            "elbow_plot": elbow_plot,
        },
        "nodes": node_records,
    }

    out_json = Path(output_json_path)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    return payload, graph
