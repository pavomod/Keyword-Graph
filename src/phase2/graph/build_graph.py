from __future__ import annotations

from itertools import combinations
from pathlib import Path

import networkx as nx
from sentence_transformers import SentenceTransformer, util
from src.phase1.keyBERT.normalize import parse_keyword_text
from transformers import logging as hf_logging


def parse_keywords(keywords_text: str) -> list[str]:
    return parse_keyword_text(keywords_text)


def resolve_entities(
    keyword_lists: list[list[str]],
    threshold: float = 0.85,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[list[str]]:
    """
    Resolves semantic entities by grouping highly similar keywords
    and replacing them with a canonical term inside each list.
    This step should run before graph construction.
    """
    # Extract all unique keywords
    all_keywords = {kw for sublist in keyword_lists for kw in sublist}
    unique_keywords = list(all_keywords)

    if len(unique_keywords) < 2:
        return keyword_lists

    # Compute similarities while disabling verbose HuggingFace logs
    previous_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        encoder = SentenceTransformer(model_name)
    finally:
        hf_logging.set_verbosity(previous_verbosity)

    embeddings = encoder.encode(unique_keywords, convert_to_tensor=True)
    similarity_matrix = util.cos_sim(embeddings, embeddings)

    # Build a temporary graph to identify synonym clusters
    alias_graph = nx.Graph()
    alias_graph.add_nodes_from(unique_keywords)

    for i in range(len(unique_keywords)):
        for j in range(i + 1, len(unique_keywords)):
            sim = float(similarity_matrix[i][j])
            if sim >= threshold:
                alias_graph.add_edge(unique_keywords[i], unique_keywords[j])

    # Determine the canonical term for each connected component (cluster)
    canonical_map: dict[str, str] = {}
    for cluster in nx.connected_components(alias_graph):
        if len(cluster) == 1:
            continue
        
        # Pick the shortest canonical term (often the cleanest/most generic)
        cluster_list = list(cluster)
        canonical_term = min(cluster_list, key=len)
        
        for term in cluster_list:
            canonical_map[term] = canonical_term

    # Rewrite original keyword lists using canonical terms
    resolved_lists = []
    for keywords in keyword_lists:
        resolved_req = []
        for kw in keywords:
            canonical = canonical_map.get(kw, kw)
            # Avoid duplicates in the same post-resolution list
            if canonical not in resolved_req:
                resolved_req.append(canonical)
        resolved_lists.append(resolved_req)

    return resolved_lists


def build_cooccurrence_graph(keyword_lists: list[list[str]]) -> nx.Graph:
    """
    Builds a co-occurrence graph from keyword lists.

    Edge weight is the Jaccard similarity between the requirement sets
    containing each keyword pair, instead of a raw count.
    This prevents very long requirements from artificially inflating weights.
    """
    graph = nx.Graph()

    # Track which requirement index contains each keyword.
    keyword_to_req_indices: dict[str, set[int]] = {}
    for req_idx, keywords in enumerate(keyword_lists):
        for keyword in keywords:
            keyword_to_req_indices.setdefault(keyword, set()).add(req_idx)

    for req_idx, keywords in enumerate(keyword_lists):
        if not keywords:
            continue

        if len(keywords) == 1:
            graph.add_node(keywords[0])
            continue

        for left, right in combinations(keywords, 2):
            left_reqs = keyword_to_req_indices.get(left, set())
            right_reqs = keyword_to_req_indices.get(right, set())
            intersection = len(left_reqs & right_reqs)
            union = len(left_reqs | right_reqs)
            jaccard = intersection / union if union > 0 else 0.0

            if graph.has_edge(left, right):
                # Keep the maximum Jaccard weight observed across requirements.
                if jaccard > graph[left][right]["weight"]:
                    graph[left][right]["weight"] = jaccard
                    graph[left][right]["jaccard"] = jaccard
            else:
                graph.add_edge(left, right, weight=jaccard, jaccard=jaccard)

    return graph


def build_semantic_similarity_graph(
    keyword_lists: list[list[str]],
    threshold: float = 0.4,
    model_name: str = "all-MiniLM-L6-v2",
) -> nx.Graph:
    all_keywords: list[str] = []
    for keywords in keyword_lists:
        all_keywords.extend(keywords)

    unique_keywords = list(dict.fromkeys(all_keywords))
    graph = nx.Graph()
    graph.add_nodes_from(unique_keywords)

    if len(unique_keywords) < 2:
        return graph

    previous_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        encoder = SentenceTransformer(model_name)
    finally:
        hf_logging.set_verbosity(previous_verbosity)

    embeddings = encoder.encode(unique_keywords, convert_to_tensor=True)
    similarity_matrix = util.cos_sim(embeddings, embeddings)

    # Compare all extracted keywords globally and keep only strong semantic relations.
    for idx_left, idx_right in combinations(range(len(unique_keywords)), 2):
        similarity = float(similarity_matrix[idx_left][idx_right])
        if similarity >= threshold:
            left = unique_keywords[idx_left]
            right = unique_keywords[idx_right]
            graph.add_edge(
                left,
                right,
                weight=similarity,
                semantic_similarity=similarity,
            )

    return graph


def build_combined_graph(
    keyword_lists: list[list[str]],
    semantic_threshold: float = 0.4,
    model_name: str = "all-MiniLM-L6-v2",
) -> nx.Graph:
    """
    Builds one graph that merges co-occurrence and semantic-similarity edges.

    Each edge carries:
    - weight: max(jaccard, semantic_similarity) - used by Node2Vec
    - edge_type: 'cooccurrence', 'semantic', or 'both'
    - jaccard: Jaccard weight (if present)
    - semantic_similarity: cosine similarity (if present)
    """
    cooccurrence = build_cooccurrence_graph(keyword_lists)
    semantic = build_semantic_similarity_graph(
        keyword_lists, threshold=semantic_threshold, model_name=model_name
    )

    combined = nx.Graph()
    combined.add_nodes_from(cooccurrence.nodes(data=True))
    combined.add_nodes_from(semantic.nodes(data=True))

    # Add co-occurrence edges first.
    for u, v, data in cooccurrence.edges(data=True):
        combined.add_edge(u, v, edge_type="cooccurrence", jaccard=data.get("jaccard", 0.0), weight=data.get("weight", 0.0))

    # Merge semantic edges.
    for u, v, data in semantic.edges(data=True):
        sim = data.get("semantic_similarity", 0.0)
        if combined.has_edge(u, v):
            combined[u][v]["semantic_similarity"] = sim
            combined[u][v]["edge_type"] = "both"
            # Use the stronger signal as a unified Node2Vec weight.
            combined[u][v]["weight"] = max(combined[u][v]["weight"], sim)
        else:
            combined.add_edge(u, v, edge_type="semantic", semantic_similarity=sim, weight=sim)

    return combined


def attach_requirement_references(
    graph: nx.Graph,
    keyword_lists: list[list[str]],
    requirement_ids: list[str],
) -> nx.Graph:
    if len(keyword_lists) != len(requirement_ids):
        raise ValueError("keyword_lists and requirement_ids must have same length")

    keyword_to_requirement_ids: dict[str, set[str]] = {}
    for requirement_id, keywords in zip(requirement_ids, keyword_lists):
        requirement_id_text = str(requirement_id).strip()
        if not requirement_id_text:
            continue

        for keyword in keywords:
            keyword_to_requirement_ids.setdefault(keyword, set()).add(requirement_id_text)

    for keyword in graph.nodes:
        req_ids = _sort_requirement_ids(keyword_to_requirement_ids.get(keyword, set()))
        graph.nodes[keyword]["requirement_ids"] = ",".join(req_ids)
        graph.nodes[keyword]["requirement_count"] = len(req_ids)

    for left, right in graph.edges:
        left_ids = keyword_to_requirement_ids.get(left, set())
        right_ids = keyword_to_requirement_ids.get(right, set())
        shared_ids = _sort_requirement_ids(left_ids & right_ids)
        graph.edges[left, right]["shared_requirement_ids"] = ",".join(shared_ids)
        graph.edges[left, right]["shared_requirement_count"] = len(shared_ids)

    return graph


def remove_isolated_nodes(graph: nx.Graph) -> tuple[nx.Graph, int]:
    isolated_nodes = [node for node in graph.nodes if graph.degree(node) == 0]
    if not isolated_nodes:
        return graph, 0

    graph = graph.copy()
    graph.remove_nodes_from(isolated_nodes)
    return graph, len(isolated_nodes)


def style_nodes_by_degree(graph: nx.Graph) -> nx.Graph:
    for node in graph.nodes:
        degree = int(graph.degree(node))
        color_hex, color_bucket = _degree_color(degree)

        graph.nodes[node]["relation_count"] = degree
        graph.nodes[node]["relation_color"] = color_hex
        graph.nodes[node]["relation_bucket"] = color_bucket
        graph.nodes[node]["viz"] = {
            "color": {
                "r": int(color_hex[1:3], 16),
                "g": int(color_hex[3:5], 16),
                "b": int(color_hex[5:7], 16),
                "a": 1.0,
            }
        }

    return graph


def _degree_color(degree: int) -> tuple[str, str]:
    if degree == 0:
        return "#e53935", "0"
    if 1 <= degree <= 5:
        return "#fb8c00", "1-5"
    if 6 <= degree <= 10:
        return "#fdd835", "6-10"
    return "#43a047", ">10"


def _sort_requirement_ids(requirement_ids: set[str]) -> list[str]:
    def sort_key(value: str) -> tuple[int, str]:
        return (0, f"{int(value):012d}") if value.isdigit() else (1, value)

    return sorted(requirement_ids, key=sort_key)


def reduce_graph_nodes(
    graph: nx.Graph,
    max_degree_ratio: float = 0.15,
    min_hub_degree: int = 5,
    degree_threshold_low: int | None = None,
    degree_threshold_high: int | None = None,
) -> nx.Graph:
    """
    Reduce graph nodes by absorbing low-value nodes into hubs.

    Two modes are supported for backward compatibility:
    1) Legacy absolute-threshold mode (if `degree_threshold_low` and
       `degree_threshold_high` are provided): a node is absorbed if its
       degree <= `degree_threshold_low` and all its neighbours are hubs
       (degree > `degree_threshold_high`). This preserves previous
       behaviour used by older CLI flags.
    2) Relative-degree mode (default): a node is absorbed when its
       degree is small relative to a candidate hub (degree/hub_degree <=
       `max_degree_ratio`) and the candidate hub meets `min_hub_degree`.

    Args:
        graph: NetworkX graph with node attributes 'requirement_ids'.
        max_degree_ratio: relative threshold for satellite-to-hub absorption.
        min_hub_degree: minimum absolute degree for a node to be considered a hub.
        degree_threshold_low: (optional) legacy absolute low-degree threshold.
        degree_threshold_high: (optional) legacy absolute hub-degree threshold.

    Returns:
        Modified graph with reduced nodes.
    """
    graph = graph.copy()

    # If legacy absolute thresholds are provided, run the original algorithm
    if degree_threshold_low is not None and degree_threshold_high is not None:
        hub_nodes = {node for node in graph.nodes() if graph.degree(node) > degree_threshold_high}

        if not hub_nodes:
            return graph

        nodes_to_remove = set()
        node_absorption_map: dict[str, str] = {}

        for node in list(graph.nodes()):
            if node in hub_nodes or node in nodes_to_remove:
                continue

            degree = graph.degree(node)

            # Isolated nodes are not reassigned
            if degree == 0:
                continue

            if degree > degree_threshold_low:
                continue

            neighbors = list(graph.neighbors(node))
            hub_neighbors = [n for n in neighbors if n in hub_nodes]

            # Only absorb if ALL neighbors are hubs (no non-hub connections to lose).
            if len(hub_neighbors) != len(neighbors):
                continue

            best_hub = max(
                hub_neighbors,
                key=lambda h: graph[node][h].get("weight", 0.0) if graph.has_edge(node, h) else 0.0,
            )
            node_absorption_map[node] = best_hub
            nodes_to_remove.add(node)

        # Merge requirement references from absorbed nodes into target hubs.
        for absorbed_node, target_hub in node_absorption_map.items():
            absorbed_reqs = {
                r
                for r in graph.nodes[absorbed_node].get("requirement_ids", "").split(",")
                if r.strip()
            }
            current_reqs = {
                r
                for r in graph.nodes[target_hub].get("requirement_ids", "").split(",")
                if r.strip()
            }
            merged_reqs = _sort_requirement_ids(current_reqs | absorbed_reqs)
            graph.nodes[target_hub]["requirement_ids"] = ",".join(merged_reqs)
            graph.nodes[target_hub]["requirement_count"] = len(merged_reqs)

        graph.remove_nodes_from(nodes_to_remove)

        # Recompute shared_requirement_count on remaining edges after merges.
        for left, right in graph.edges():
            left_ids = {r for r in graph.nodes[left].get("requirement_ids", "").split(",") if r.strip()}
            right_ids = {r for r in graph.nodes[right].get("requirement_ids", "").split(",") if r.strip()}
            shared_ids = _sort_requirement_ids(left_ids & right_ids)
            graph.edges[left, right]["shared_requirement_ids"] = ",".join(shared_ids)
            graph.edges[left, right]["shared_requirement_count"] = len(shared_ids)

        return graph

    # Default: relative-degree absorption algorithm
    nodes_sorted = sorted(graph.nodes(), key=lambda n: graph.degree(n))

    for node in nodes_sorted:
        # Skip if the node was already absorbed in a previous iteration
        if node not in graph:
            continue

        degree = graph.degree(node)

        # Isolated nodes are not reassigned (they should already be removed).
        if degree == 0:
            continue

        neighbors = list(graph.neighbors(node))
        if not neighbors:
            continue

        # Find the neighbor with the highest absolute degree (candidate hub)
        best_hub = max(neighbors, key=lambda n: graph.degree(n))
        hub_degree = graph.degree(best_hub)

        # Ignore if the candidate hub is too small to act as a hub
        if hub_degree < min_hub_degree:
            continue

        # Relative check: e.g., 10 edges / 100 edges = 0.10 <= 0.15 -> Absorb
        if (degree / hub_degree) <= max_degree_ratio:

            # 1. Merge requirement references
            node_reqs = {r for r in graph.nodes[node].get("requirement_ids", "").split(",") if r.strip()}
            hub_reqs = {r for r in graph.nodes[best_hub].get("requirement_ids", "").split(",") if r.strip()}

            merged_reqs = _sort_requirement_ids(hub_reqs | node_reqs)
            graph.nodes[best_hub]["requirement_ids"] = ",".join(merged_reqs)
            graph.nodes[best_hub]["requirement_count"] = len(merged_reqs)

            # 2. Transfer edges to preserve topology
            for neighbor in list(graph.neighbors(node)):
                if neighbor == best_hub:
                    continue

                # Keep maximum weight if the hub already shares an edge with this neighbor
                edge_weight = graph[node][neighbor].get("weight", 0.0)
                if graph.has_edge(best_hub, neighbor):
                    graph[best_hub][neighbor]["weight"] = max(
                        graph[best_hub][neighbor]["weight"],
                        edge_weight,
                    )
                else:
                    graph.add_edge(best_hub, neighbor, weight=edge_weight)

            # 3. Physically remove the absorbed satellite node
            graph.remove_node(node)

    # Recompute shared requirements count on remaining edges after all merges
    for left, right in graph.edges():
        left_ids = {r for r in graph.nodes[left].get("requirement_ids", "").split(",") if r.strip()}
        right_ids = {r for r in graph.nodes[right].get("requirement_ids", "").split(",") if r.strip()}
        shared_ids = _sort_requirement_ids(left_ids & right_ids)
        graph.edges[left, right]["shared_requirement_ids"] = ",".join(shared_ids)
        graph.edges[left, right]["shared_requirement_count"] = len(shared_ids)

    return graph


def save_graph_gexf(graph: nx.Graph, output_path: str) -> str:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(graph, destination)
    return str(destination)