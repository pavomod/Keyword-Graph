# Phase 2: Semantic Graph Construction

## Executive Summary

Phase 2 transforms extracted keywords into a structured semantic graph where nodes represent concepts and edges represent semantic relationships. The phase maintains complete traceability, linking every graph element back to the original requirements, enabling later analysis to understand not just *what* concepts are related, but *why* (through requirement evidence).

## What Phase 2 Achieves

**Input**: Keyword lists from Phase 1  
**Output**: A semantic graph in GEXF format with requirement traceability metadata

The phase answers:
- Which keywords are semantically related to each other?
- How strong is each relationship?
- Which requirements provide evidence for each relationship?

## Graph Construction Strategy

Phase 2 builds a unified graph combining two complementary relationship types:

### Relationship Type 1: Co-occurrence

**Definition**: Keywords that appear together in the same requirement

**How it works**:
- For each requirement, all keyword pairs that appear together are connected
- Edge weight = Jaccard similarity of their requirement coverage
- If keywords A and B appear in requirements {1,2,3} and {2,3,4} respectively:
  - Intersection: {2,3} (shared in 2 requirements)
  - Union: {1,2,3,4} (appear across 4 requirements total)
  - Jaccard = 2/4 = 0.5

**Why this matters**: Keywords mentioned together are likely describing related aspects of the same feature

### Relationship Type 2: Semantic Similarity

**Definition**: Keywords that are conceptually similar even if not mentioned together

**How it works**:
- Each keyword is converted to an embedding using a pre-trained language model (`all-MiniLM-L6-v2`)
- Cosine similarity is computed for all keyword pairs
- An edge is created only when similarity ≥ `--semantic-threshold` (default: 0.4)
- Examples of high similarity:
  - "light switch" vs "lighting control"
  - "temperature sensor" vs "thermometer"
  - "automated" vs "automation"

**Why this matters**: Semantic similarity captures synonyms and conceptually equivalent ideas that don't co-occur but mean similar things

### Combined Graph

The two relationship types are merged into a single graph where:
- **Nodes** = all unique keywords
- **Edges** combine both relationship types
- **Edge attributes** track the evidence:
  - `weight` — strongest signal (co-occurrence weight OR semantic similarity)
  - `edge_type` — whether edge came from "cooccurrence", "semantic", or "both"
  - Both individual metrics are preserved for detailed analysis

## Graph Refinement

### Step 1: Remove Isolated Nodes

Nodes with no connections (degree = 0) are removed because:
- They represent concepts mentioned in isolation
- They don't contribute to relational structure
- They can harm clustering quality and visualization clarity

### Step 2: Optional Node Reduction (Enabled by Default)

**Purpose**: Reduce noise from low-value singleton nodes while preserving traceability

**How it works**:

A node is considered "low-value" if:
- Its degree is ≤ `--node-reduction-degree-threshold-low` (default: 2)
- It is only connected to "hub" nodes (degree > `--node-reduction-degree-threshold-high`, default: 5)

Low-value nodes are merged into the strongest hub connected to them. Before removal:
- All requirement references from the low-value node are transferred to the hub
- The edge weight is updated to reflect the merge

**Example**:
```
Before:    "timer module" --0.6-- "automation hub" (degree 8)
           "timer module" --0.4-- "scheduling system" (degree 1)
           
After:     "timer module" is merged into "automation hub"
           Requirements that mentioned "timer module" are now attributed to "automation hub"
```

This reduces graph complexity while maintaining requirement linkage.

## Traceability Metadata

### Node Attributes

Each node stores:
- `requirement_ids` — List of all original requirements mentioning this keyword
- `requirement_count` — How many requirements mention this keyword

Example:
```
Node: "lighting control"
  requirement_ids: ["req_42", "req_103", "req_157"]
  requirement_count: 3
```

This enables analysts to instantly understand:
- How common is this concept across requirements?
- Which specific requirements are affected by this concept?

### Edge Attributes

Each edge stores:
- `shared_requirement_ids` — Requirements that mention BOTH connected keywords
- `shared_requirement_count` — How many requirements mention both concepts

Example:
```
Edge: "lighting" -- "remote control"
  shared_requirement_ids: ["req_42", "req_103"]
  shared_requirement_count: 2
  
Interpretation: These two concepts co-occur in 2 requirements,
suggesting they are genuinely related concepts for users.
```

## Visual Styling

Nodes are automatically colored based on their structural importance:

| Degree Range | Color | Meaning |
|--------------|-------|---------|
| 0 (isolated) | Red | Removed from final graph |
| 1–3 | Orange | Low connectivity, supporting concepts |
| 4–5 | Yellow | Moderate connectivity, important concepts |
| >5 | Green | Hub nodes, central to requirement structure |

These colors help analysts quickly identify the "star" concepts that appear frequently and in multiple contexts.

## Output Artifacts

Phase 2 produces:
- **GEXF file** (`results/keyword_graph.gexf`) — Importable into graph visualization tools (Gephi, Cytoscape)
- **Graph statistics** — Node count, edge count, density metrics
- **Visual attributes** — X/Y coordinates (computed later in Phase 3)

## Design Philosophy

Phase 2's key principle: **"Keep every requirement connected to every graph element"**

This design choice:
- Prevents loss of information during transformation
- Enables downstream phases to validate clustering against original requirements
- Allows inconsistency detection to reference source material
- Provides accountability: analysts can always see the evidence for any graph structure

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

[Phase 3 - Clustering and Requirement-Aware Features](PHASE_3_CLUSTERING_AND_REQUIREMENT_AWARE_FEATURES.md)
