# Phase 2: Graph and Traceability

This step converts keyword outputs into a semantic graph and attaches requirement traceability.

## Quick Purpose

Phase 2 answers two questions:
- Which keywords are semantically related?
- Which requirements justify each node and edge?

## What This Step Works On

Input from Phase 1:
- One normalized keyword list per requirement
- One requirement identifier (`source_id`) per row

Data source context:
- Requirements are taken from `data/crowd.csv`
- Reference `tags` are not used to build graph structure

## What This Step Produces

- A semantic keyword graph in GEXF format
- Node-level requirement traceability
- Edge-level shared-requirement metadata
- Degree-based visual attributes for graph tools

Default output:
- `results/keyword_graph.gexf`

## How This Step Works

### 1. Build the semantic graph

Graph type:
- Undirected `networkx.Graph`

Construction logic:
- Create one node per unique normalized keyword
- Embed keywords with SentenceTransformer (`all-MiniLM-L6-v2` by default)
- Compute cosine similarity for keyword pairs
- Create an edge only when similarity >= `--semantic-threshold`

Edge attributes set at creation:
- `weight`
- `semantic_similarity`

### 2. Attach requirement provenance to nodes

For each keyword node, Phase 2 stores:
- `requirement_ids`: requirements where the keyword appears
- `requirement_count`: number of linked requirements

This enables direct traceability from concept to source requirement rows.

### 3. Remove isolated nodes

Nodes with degree `0` are removed automatically.

Why:
- Isolated terms do not contribute to relational structure
- They can harm clustering quality and readability

### 4. Optional node reduction (enabled by default)

Goal:
- Reduce noisy low-value nodes while keeping traceability

Rules:
- Hub nodes: degree > high threshold (default 5)
- Low-degree nodes: degree <= low threshold (default 2)
- A low-degree node connected only to hubs is merged into the strongest hub
- An isolated node is merged into the hub with the richest requirement coverage
- Requirement references are transferred before node removal

Configuration:
- Enable: `--enable-node-reduction`
- Disable: `--disable-node-reduction`
- Low threshold: `--node-reduction-degree-threshold-low`
- High threshold: `--node-reduction-degree-threshold-high`

### 5. Attach shared requirement metadata to edges

Each edge gets overlap metadata between endpoint nodes:
- `shared_requirement_ids`
- `shared_requirement_count`

This shows whether connected concepts are also co-mentioned in the same requirements.

### 6. Add visual styling by relation count

Node color buckets by degree:
- Degree `0`: red
- Degree `1-3`: orange
- Degree `4-5`: yellow
- Degree `>5`: green

Added attributes:
- `relation_count`
- `relation_color`
- `relation_bucket`
- `viz.color` (GEXF-compatible)

### 7. Export graph

The final enriched graph is written to GEXF.

## Processing Order Summary

1. Parse normalized keywords
2. Build semantic graph
3. Attach node traceability
4. Remove isolated nodes
5. Apply optional node reduction
6. Attach edge overlap metadata
7. Add visual styling
8. Export GEXF

## Main Code Entry Points

- `src/run_pipeline.py`
- `src/graph/build_graph.py`
- `src/keywords/normalize.py`

## Next Step

[Phase 3 - Clustering and Requirement-Aware Features](explanation/PHASE_3_CLUSTERING_AND_REQUIREMENT_AWARE_FEATURES.md)
