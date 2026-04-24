from __future__ import annotations

from itertools import combinations
from pathlib import Path

import networkx as nx
from sentence_transformers import SentenceTransformer, util
from src.keywords.normalize import parse_keyword_text
from transformers import logging as hf_logging


def parse_keywords(keywords_text: str) -> list[str]:
    return parse_keyword_text(keywords_text)


def build_cooccurrence_graph(keyword_lists: list[list[str]]) -> nx.Graph:
    graph = nx.Graph()

    for keywords in keyword_lists:
        if not keywords:
            continue

        if len(keywords) == 1:
            graph.add_node(keywords[0])
            continue

        for left, right in combinations(keywords, 2):
            if graph.has_edge(left, right):
                graph[left][right]["weight"] += 1
            else:
                graph.add_edge(left, right, weight=1)

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
    if 1 <= degree <= 3:
        return "#fb8c00", "1-3"
    if 4 <= degree <= 5:
        return "#fdd835", "4-5"
    return "#43a047", ">5"


def _sort_requirement_ids(requirement_ids: set[str]) -> list[str]:
    def sort_key(value: str) -> tuple[int, str]:
        return (0, f"{int(value):012d}") if value.isdigit() else (1, value)

    return sorted(requirement_ids, key=sort_key)


def reduce_graph_nodes(
    graph: nx.Graph,
    degree_threshold_low: int = 2,
    degree_threshold_high: int = 5,
) -> nx.Graph:
    """
    Reduce the number of graph nodes by absorbing low-degree nodes into high-degree hub nodes.
    
    Strategy:
    - Identify hub nodes: degree > degree_threshold_high
    - For each low-degree node (degree <= degree_threshold_low):
      - If connected only to hub nodes, absorb it into the hub with highest weight connection
      - Transfer requirement references to the absorbing hub
      - Remove the low-degree node from the graph
    
    Args:
        graph: NetworkX graph with node attributes 'requirement_ids', 'requirement_count'
        degree_threshold_low: Maximum degree for a node to be considered "low"
        degree_threshold_high: Minimum degree for a node to be considered "hub"
    
    Returns:
        Modified graph with reduced nodes
    """
    graph = graph.copy()
    
    # Identify hub nodes
    hub_nodes = set(node for node in graph.nodes() if graph.degree(node) > degree_threshold_high)
    
    if not hub_nodes:
        return graph
    
    nodes_to_remove = set()
    node_absorption_map = {}  # Maps absorbed node -> target hub node
    
    # For each low-degree node, check if it should be absorbed
    for node in list(graph.nodes()):
        if node in hub_nodes or node in nodes_to_remove:
            continue
        
        degree = graph.degree(node)
        if degree > degree_threshold_low:
            continue
        
        # Get neighbors and check if all are hubs
        neighbors = list(graph.neighbors(node))
        hub_neighbors = [n for n in neighbors if n in hub_nodes]
        
        # Only absorb if connected ONLY to hubs or is isolated
        if len(hub_neighbors) == len(neighbors) and (len(neighbors) > 0 or degree == 0):
            # Find the hub with the strongest edge (highest weight)
            if neighbors:
                best_hub = max(
                    hub_neighbors,
                    key=lambda h: graph[node][h].get("weight", 0)
                    if graph.has_edge(node, h)
                    else 0,
                )
            else:
                # Isolated node: attach to largest hub by requirement count
                best_hub = max(
                    hub_nodes,
                    key=lambda h: graph.nodes[h].get("requirement_count", 0),
                )
            
            node_absorption_map[node] = best_hub
            nodes_to_remove.add(node)
    
    # Merge requirement references from absorbed nodes to target hubs
    for absorbed_node, target_hub in node_absorption_map.items():
        absorbed_reqs = set(
            graph.nodes[absorbed_node]
            .get("requirement_ids", "")
            .split(",")
            if graph.nodes[absorbed_node].get("requirement_ids")
            else []
        )
        absorbed_reqs = {r for r in absorbed_reqs if r}  # Remove empty strings
        
        current_reqs = set(
            graph.nodes[target_hub]
            .get("requirement_ids", "")
            .split(",")
            if graph.nodes[target_hub].get("requirement_ids")
            else []
        )
        current_reqs = {r for r in current_reqs if r}  # Remove empty strings
        
        # Merge requirement sets
        merged_reqs = _sort_requirement_ids(current_reqs | absorbed_reqs)
        graph.nodes[target_hub]["requirement_ids"] = ",".join(merged_reqs)
        graph.nodes[target_hub]["requirement_count"] = len(merged_reqs)
    
    # Remove absorbed nodes from graph
    graph.remove_nodes_from(nodes_to_remove)
    
    # Update edge shared_requirement_count after node removal
    for left, right in graph.edges():
        left_ids = set(
            graph.nodes[left]
            .get("requirement_ids", "")
            .split(",")
            if graph.nodes[left].get("requirement_ids")
            else []
        )
        left_ids = {r for r in left_ids if r}
        
        right_ids = set(
            graph.nodes[right]
            .get("requirement_ids", "")
            .split(",")
            if graph.nodes[right].get("requirement_ids")
            else []
        )
        right_ids = {r for r in right_ids if r}
        
        shared_ids = _sort_requirement_ids(left_ids & right_ids)
        graph.edges[left, right]["shared_requirement_ids"] = ",".join(shared_ids)
        graph.edges[left, right]["shared_requirement_count"] = len(shared_ids)
    
    return graph


def save_graph_gexf(graph: nx.Graph, output_path: str) -> str:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(graph, destination)
    return str(destination)
