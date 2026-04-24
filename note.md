# Workflow: Keyword Graph from Requirements Engineering

1) Use a model (optional fine-tuning) to extract keywords from a requirements dataset (crowd).  
[COMPLETED] Modify the prompt to avoid overly frequent and uninformative terms such as "HOME". [COMPLETED]

2) Build an algorithm that creates an N:N relationship graph among the keywords extracted from the model.  
[COMPLETED] Reduce the number of nodes → if a node has many relationships, keep only the most informative keyword. (How to define "many"? For example: a node with high degree can absorb/replace connected nodes with low degree.) also delete nodes with 0 degree.[COMPLETED]

3) Use an unsupervised approach such as K-means to generate clusters in the resulting graph.

4) [COMPLETED] From the clusters obtained in the previous step, map requirements to clusters in a many-to-many relationship (no duplicates within cluster). Each requirement can belong to multiple clusters if extracted keywords appear in different clusters. [COMPLETED]

5) [COMPLETED] Create a JSON system that permit me to evaluate whether this pipeline performs better than directly clustering the requirements. without the use of the keyword etc... [COMPLETED]

6) [COMPLETED] Label the clusters using FLAN-T5 model to generate descriptive 2-5 word labels for each cluster based on keywords and requirement count. [COMPLETED]

7) [NEW] Use an LLM or a pre-trained model to detect inconsistencies and suggest improvements for each cluster by providing:
- the cluster
- a mapping indicating which requirement each node belongs to  

The model must return:
- inconsistent requirements with explanations
- a suggested improved requirement  
[NEW]

-- Example of cluster inconsistency --
- The television cannot be turned on before 8 a.m.  
- The irrigation system must be turned on before 8 a.m.

## Pipeline Overview

```text
Crowd CSV Dataset
      |
      v
[PHASE 1] flan-t5-base fine-tuning      -> Keyword Extraction
      |
      v
[PHASE 2] N:N graph construction        -> NetworkX (co-occurrence + semantic similarity)
      |
      v
[PHASE 3] Node2Vec + K-Means            -> Keyword Clusters
      |
      v
[PHASE 4] LLM analysis                  -> Inconsistencies + Suggestions