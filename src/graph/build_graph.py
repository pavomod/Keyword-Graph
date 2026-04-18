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


def save_graph_gexf(graph: nx.Graph, output_path: str) -> str:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(graph, destination)
    return str(destination)
