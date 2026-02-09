
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def test_metrics():
    print("Testing Metric Calculation...")
    
    # Create a dummy graph
    G = nx.erdos_renyi_graph(100, 0.1, directed=True)
    
    # Create dummy labels (imbalanced)
    # Nodes 0-4 are fraud (1), rest are legit (0)
    node_labels = {i: 0 for i in range(100)}
    for i in range(5):
        node_labels[i] = 1
        
    y_true = [node_labels[n] for n in G.nodes()]
    
    # Fake scores (perfect prediction)
    scores_perfect = {i: (0.9 if i < 5 else 0.1) for i in range(100)}
    y_scores_perfect = [scores_perfect[n] for n in G.nodes()]
    
    # Fake scores (random)
    scores_random = {i: np.random.random() for i in range(100)}
    y_scores_random = [scores_random[n] for n in G.nodes()]

    # Calculate
    auc_perf = roc_auc_score(y_true, y_scores_perfect)
    pr_perf = average_precision_score(y_true, y_scores_perfect)
    
    auc_rand = roc_auc_score(y_true, y_scores_random)
    pr_rand = average_precision_score(y_true, y_scores_random)
    
    print(f"Perfect - ROC-AUC: {auc_perf}, PR-AUC: {pr_perf}")
    print(f"Random  - ROC-AUC: {auc_rand}, PR-AUC: {pr_rand}")
    
    assert auc_perf == 1.0
    assert pr_perf == 1.0
    print("Metrics check passed.")

if __name__ == "__main__":
    test_metrics()
