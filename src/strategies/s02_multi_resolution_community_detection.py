import time
import networkx as nx
import numpy as np
import community.community_louvain as community_louvain
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from src.graph_utils import extract_enhanced_features, evaluate

def run(G, df, node_labels, fraud_train, legit_nodes, test_nodes, test_labels, dataset_name, src_col, label_col):
    print("\n🔬 Strategy 2: Multi-Resolution Community Detection")
    t0 = time.time()

    G_und = G.to_undirected()
    G_und_weighted = nx.Graph()
    for u, v, d in G_und.edges(data=True):
        w = d.get("weight", 1)
        if G_und_weighted.has_edge(u, v):
            G_und_weighted[u][v]["weight"] += w
        else:
            G_und_weighted.add_edge(u, v, weight=w)
            
    for n in G.nodes():
        if n not in G_und_weighted:
            G_und_weighted.add_node(n)

    resolutions = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    all_partitions = {}
    all_comm_sizes = {}
    for res in resolutions:
        part = community_louvain.best_partition(G_und_weighted, resolution=res, random_state=42)
        all_partitions[res] = part
        cs = defaultdict(int)
        for v in part.values():
            cs[v] += 1
        all_comm_sizes[res] = cs

    seeds = {n: 1.0 for n in fraud_train if n in G.nodes()}
    ppr_scores = nx.pagerank(G, personalization=seeds if seeds else None)

    rng = np.random.RandomState(42)
    legit_train = rng.choice(legit_nodes, size=min(len(fraud_train), len(legit_nodes)), replace=False).tolist()
    train_nodes = fraud_train + legit_train

    X_train_base = extract_enhanced_features(G, train_nodes, ppr_scores, all_partitions[1.0], all_comm_sizes[1.0])
    X_test_base = extract_enhanced_features(G, test_nodes, ppr_scores, all_partitions[1.0], all_comm_sizes[1.0])

    def multi_res_features(nodes):
        feats = []
        for node in nodes:
            row = []
            for res in resolutions:
                cid = all_partitions[res].get(node, -1)
                csize = all_comm_sizes[res].get(cid, 0)
                row.append(csize)
            feats.append(row)
        return np.array(feats, dtype=float)

    X_train_mr = multi_res_features(train_nodes)
    X_test_mr = multi_res_features(test_nodes)

    X_train = np.hstack([X_train_base, X_train_mr])
    X_test = np.hstack([X_test_base, X_test_mr])

    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    y_train = [node_labels[n] for n in train_nodes]
    y_test = [test_labels[n] for n in test_nodes]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_feats = X_train.shape[1]
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test, y_proba)
    runtime = time.time() - t0
    
    print(f"   ✅ ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  F1={metrics['f1']:.4f}  ({runtime:.1f}s)")
    
    return {
        "strategy": "S02_Multi_Resolution_Community_Detection",
        "dataset": dataset_name,
        "roc_auc": metrics.get("roc_auc", 0),
        "pr_auc": metrics.get("pr_auc", 0),
        "f1": metrics.get("f1", 0),
        "precision": metrics.get("precision", 0),
        "recall": metrics.get("recall", 0),
        "n_features": n_feats,
        "runtime_sec": runtime
    }
