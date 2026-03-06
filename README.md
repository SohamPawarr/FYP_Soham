# Graph-Based Fraud Detection System (Elliptic Data Set)

A comprehensive Python framework for detecting illicit transactions in the Elliptic Bitcoin dataset using advanced Graph Neural Networks, Community Detection, and Meta-Learning approaches.

## Setup & Execution

### Prerequisites
The project uses `uv` for dependency management. All datasets are dynamically downloaded via `kagglehub` on the first run, so no manual dataset fetching is required.

```bash
# Install dependencies
uv sync

# Run the benchmark pipeline
uv run main.py
```

## 🏆 The State-of-the-Art Architecture: Unified Multi-Stage Detection Framework (S11)

Through empirical benchmarking against 10 other architectural designs, our **Unified Multi-Stage Detection Framework** (Strategy 11) emerged as the definitive champion. This architecture leverages a multi-faceted ensemble of meta-learning concepts to achieve state-of-the-art anomaly detection on large-scale transaction networks.

### Architecture Topology

```mermaid
graph TD
    classDef nodeData fill:#252525,stroke:#fff,stroke-width:2px,color:#fff
    classDef extract fill:#0d47a1,stroke:#fff,stroke-width:2px,color:#fff
    classDef model fill:#1b5e20,stroke:#fff,stroke-width:2px,color:#fff
    classDef output fill:#b71c1c,stroke:#fff,stroke-width:2px,color:#fff

    A[Raw Elliptic Bitcoin Transactions]:::nodeData --> B{Graph Construction}:::extract
    
    B --> C[Structural Heuristics]:::extract
    C --> C1(Degree, In/Out-Flow, Reciprocity)
    
    B --> D[Community Partitioning]:::extract
    D --> D1(Louvain Modularity Partitions)
    D --> D2(Tarjan Strongly Connected Components)
    
    B --> E[Higher-Order Motifs]:::extract
    E --> E1(Higher-Order Triangles & k-Hop Paths)
    
    B --> F[Stochastic Risk Propagation]:::extract
    F --> F1(Chronological Personalized PageRank)
    F --> F2(Semi-Supervised Label Propagation)
    F --> F3(HITS Hub & Authority Vectorization)
    
    C1 --> G[[Unified Feature Integration Layer]]:::model
    D1 --> G
    D2 --> G
    E1 --> G
    F1 --> G
    F2 --> G
    F3 --> G
    
    G --> H[Heterogeneous Stacking Ensemble]:::model
    H --> H1(XGBoost Gradient Boosting)
    H --> H2(Random Forest Forest)
    H --> H3(Stochastic Gradient Boosting)
    
    H1 --> I[Logistic Regression Meta-Classifer]:::model
    H2 --> I
    H3 --> I
    
    I --> J((Final Illicit Transaction Prediction)):::output
```

### High-Level Component Explanations

1. **Topological Motifs:** Extracts the atomic structural roles (higher-order triangle formations) to identify structured tumbling and money laundering efforts.
2. **Community Partitions (Louvain & SCC):** Identifies macro-clusters and cyclic formations. Illicit actors often operate in dense rings to obfuscate flow. 
3. **Risk propagation (C-PPR & LP):** Utilizes seeds of known illicit entities to run a Modified Chronological Personalized PageRank. It calculates "risk-by-association" across multiple hops.
4. **Stacking Ensemble:** A Logistic Regression meta-classifier handles the final decision boundary, optimally weighting sub-predictions from XGBoost, Gradient Boosters, and Random Forests.

## Experimental Framework & Comparison Benchmarks

The framework executes 12 distinct isolated methodologies in parallel (using `multiprocessing`) to evaluate their relative effectiveness on Bitcoin transaction networks. Results are output to the `./results` directory.

### Strategies Evaluated:

* `s00_topological_baseline.py`: Basic I-hop graph heuristics.
* `s01_structural_feature_augmentation.py`: Classic Social Network Analysis (SNA) metrics. 
* `s02_multi_resolution_community_detection.py`: Unsupervised multi-scale clustering.
* `s03_associative_propagated_risk_2hop.py`: Multi-hop guilt-propagation algorithms.
* `s04_heterogeneous_stacking_ensemble.py`: Base ensemble learning implementations. 
* `s05_spatio_temporal_feature_fusion.py`: Spatio-temporal transactional heuristics. 
* `s06_latent_graph_embeddings_node2vec.py`: Latent graph-embedded transformations.
* `s07_chronological_risk_propagation.py`: PPR adjusted with chronological weights.
* `s08_higher_order_flow_motifs.py`: Complex flow-cycle topology.
* `s09_semi_supervised_identity_propagation.py`: Iterative label-inference algorithms.
* `s10_composite_multi_feature_ensemble.py`: Deep structural feature integration.
* `s11_unified_multistage_detection_framework.py`: The champion architectural framework.
