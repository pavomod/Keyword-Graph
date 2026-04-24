from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import umap
from node2vec import Node2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
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
    left = _normalize_block(node2vec_vectors)
    if requirement_vectors is None:
        return left

    right = _normalize_block(requirement_vectors)
    weight = max(0.0, float(requirement_weight))
    if weight == 0.0:
        return left

    return np.concatenate([left, right * weight], axis=1)


_LABEL_STOPWORDS = {
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
    "need",
    "use",
    "get",
    "check",
    "stay",
}


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _split_label_candidate(text: str) -> list[str]:
    parts = re.split(r"[,;/|]+", str(text))
    return [part.strip() for part in parts if part.strip()]


def _tokenize_label_text(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]+", str(text).lower())
    filtered: list[str] = []
    for token in tokens:
        if len(token) <= 2 or token in _LABEL_STOPWORDS:
            continue
        if token not in filtered:
            filtered.append(token)
    return filtered


def _fallback_cluster_label(keywords: list[str], max_words: int = 4) -> str:
    tokens: list[str] = []
    for keyword in keywords:
        keyword_text = str(keyword).strip().lower()
        if not keyword_text:
            continue
        for token in re.findall(r"[a-zA-Z][a-zA-Z\-]+", keyword_text):
            if len(token) <= 2 or token in _LABEL_STOPWORDS:
                continue
            if token not in tokens:
                tokens.append(token)
            if len(tokens) >= max_words:
                return " ".join(tokens[:max_words])

    if not tokens:
        return "general requirements"

    if len(tokens) == 1:
        return f"{tokens[0]} topic"

    return " ".join(tokens[:max_words])


def _clean_cluster_label(raw_label: str, fallback_keywords: list[str]) -> tuple[str, bool]:
    candidates = _split_label_candidate(raw_label)
    if candidates:
        raw_label = candidates[0]

    tokens = _tokenize_label_text(raw_label)
    if not tokens or len(tokens) < 2:
        return _fallback_cluster_label(fallback_keywords), True

    # Reject outputs that mostly look like repeated keyword lists.
    token_counts = Counter(tokens)
    repeated_tokens = sum(1 for count in token_counts.values() if count > 1)
    if len(tokens) > 5 or repeated_tokens > 0:
        return _fallback_cluster_label(fallback_keywords), True

    return " ".join(tokens[:4]), False


def generate_cluster_labels(
    clusters_keywords: dict[int, list[str]],
    cluster_requirements: dict[int, list[str]],
    model_dir: str | None = None,
    base_model_name: str = "google/flan-t5-base",
) -> tuple[dict[int, str], dict[int, bool]]:
    """
    Generate descriptive labels for each cluster using FLAN-T5.
    
    Args:
        clusters_keywords: Maps cluster_id -> list of keywords in that cluster
        cluster_requirements: Maps cluster_id -> list of requirement IDs in that cluster
        model_dir: Path to fine-tuned model (optional)
        base_model_name: Base model name if no fine-tuned model available
    
    Returns:
        Tuple of:
        - Dictionary mapping cluster_id -> label (string)
        - Dictionary mapping cluster_id -> whether fallback was used
    """
    try:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        import torch
    except ImportError:
        return (
            {cid: f"cluster {cid}" for cid in clusters_keywords},
            {cid: True for cid in clusters_keywords},
        )
    
    # Load a label-only model so cluster labeling stays isolated from keyword extraction.
    try:
        tokenizer = T5Tokenizer.from_pretrained(base_model_name)
        model = T5ForConditionalGeneration.from_pretrained(base_model_name)
    except Exception:
        return (
            {cid: f"cluster {cid}" for cid in clusters_keywords},
            {cid: True for cid in clusters_keywords},
        )
    
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    labels = {}
    fallback_used = {}
    
    for cluster_id, keywords in clusters_keywords.items():
        req_count = len(cluster_requirements.get(cluster_id, []))
        unique_keywords = _dedupe_preserve_order(keywords)
        keywords_text = ", ".join(unique_keywords[:20])
        
        # Create prompt for FLAN-T5
        prompt = (
            "You are labeling a cluster of software requirements.\n"
            "Return exactly ONE descriptive category label.\n\n"

            "RULES:\n"
            "- 2 to 5 words\n"
            "- lowercase only\n"
            "- no punctuation\n"
            "- no commas or lists\n"
            "- no repeated words\n"
            "- DO NOT copy or concatenate the given keywords\n"
            "- DO NOT list actions or verbs in sequence\n"
            "- the label must describe the general concept or category, not the keywords themselves\n"
            "- prefer abstract, human-readable categories (e.g., 'air quality monitoring', 'user notification system')\n"
            "- avoid generic labels like 'system', 'feature', 'functionality'\n\n"

            "GOOD EXAMPLES:\n"
            "keywords: air, purification, filter, quality -> air quality monitoring\n"
            "keywords: notify, reminder, alert, alarm -> notification management\n"
            "keywords: order, delivery, food, restaurant -> food delivery service\n\n"

            "BAD EXAMPLES:\n"
            "air purification filter\n"
            "notify reminder alert\n"
            "order send arrive open\n\n"

            f"KEYWORDS: {keywords_text}\n"
            f"REQUIREMENTS COUNT: {req_count}\n\n"

            "OUTPUT FORMAT:\n"
            "label only\n\n"

            "LABEL:"
        )
        
        try:
            # Tokenize and generate
            encoded = tokenizer(
                prompt,
                max_length=256,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=8,
                    min_new_tokens=2,
                    num_beams=5,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    do_sample=False,
                    early_stopping=True,
                )
            
            label_text = tokenizer.decode(generated[0], skip_special_tokens=True).strip()

            cleaned_label, used_fallback = _clean_cluster_label(label_text, unique_keywords)
            labels[cluster_id] = cleaned_label if cleaned_label else f"cluster {cluster_id}"
            fallback_used[cluster_id] = used_fallback
        except Exception:
            labels[cluster_id] = _fallback_cluster_label(unique_keywords)
            fallback_used[cluster_id] = True
    
    return labels, fallback_used


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
    requirement_feature_weight: float = 0.5,
    requirement_svd_dim: int = 16,
    model_dir: str | None = None,
    base_model_name: str = "google/flan-t5-base",
    enable_cluster_labeling: bool = True,
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
        silhouette = -1.0
    else:
        kmeans = KMeans(n_clusters=selected_k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(vectors)
        inertia = float(kmeans.inertia_)
        silhouette = float(silhouette_score(vectors, labels)) if selected_k < len(nodes) else -1.0

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

        # Collect requirements for each cluster
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

    # Build cluster-to-requirements mapping
    cluster_requirements_mapping = {
        str(cluster_id): sorted(list(req_ids))
        for cluster_id, req_ids in cluster_to_requirements.items()
    }
    
    # Build requirement-to-cluster mapping (one requirement can appear in multiple clusters)
    requirement_to_cluster_mapping = {}
    for cluster_id, req_ids in cluster_to_requirements.items():
        for req_id in req_ids:
            if str(req_id) not in requirement_to_cluster_mapping:
                requirement_to_cluster_mapping[str(req_id)] = []
            requirement_to_cluster_mapping[str(req_id)].append(int(cluster_id))
    
    # Sort cluster lists for consistency
    requirement_to_cluster_mapping = {
        req_id: sorted(clusters)
        for req_id, clusters in requirement_to_cluster_mapping.items()
    }

    # Generate cluster labels using FLAN-T5
    cluster_labels = {}
    label_fallback_flags: dict[int, bool] = {}
    if enable_cluster_labeling and selected_k > 0:
        # Build cluster-to-keywords mapping
        clusters_keywords = {}
        for node_idx, node in enumerate(nodes):
            cluster_id = int(labels[node_idx])
            if cluster_id not in clusters_keywords:
                clusters_keywords[cluster_id] = []
            clusters_keywords[cluster_id].append(node)
        
        cluster_labels, label_fallback_flags = generate_cluster_labels(
            clusters_keywords=clusters_keywords,
            cluster_requirements=cluster_to_requirements,
            model_dir=model_dir,
            base_model_name=base_model_name,
        )

    elbow_plot = _save_elbow_plot(k_selection.get("candidates", []), output_elbow_plot_path) if output_elbow_plot_path else None

    payload = {
        "clustering": {
            "enabled": True,
            "selected_k": int(selected_k),
            "num_nodes": int(len(nodes)),
            "embedding_dim": int(embedding_dim),
            "effective_feature_dim": int(vectors.shape[1]),
            "inertia": float(inertia),
            "silhouette": float(silhouette),
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
        "cluster_labels": {str(cid): label for cid, label in cluster_labels.items()},
        "cluster_labeling": {
            "enabled": bool(enable_cluster_labeling),
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

    out_json = Path(output_json_path)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    return payload, graph
