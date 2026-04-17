from __future__ import annotations

from itertools import combinations
from pathlib import Path

import networkx as nx
from sentence_transformers import SentenceTransformer, util


def parse_keywords(keywords_text: str) -> list[str]:
    normalized = [k.strip().lower() for k in str(keywords_text).split(",") if k.strip()]
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(normalized))


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

    encoder = SentenceTransformer(model_name)
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


def save_graph_gexf(graph: nx.Graph, output_path: str) -> str:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(graph, destination)
    return str(destination)
