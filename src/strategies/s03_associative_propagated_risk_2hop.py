import time
import networkx as nx
import numpy as np
import community.community_louvain as community_louvain
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from src.graph_utils import extract_enhanced_features, evaluate

def run(G, df, node_labels, fraud_train, legit_nodes, test_nodes, test_labels, dataset_name, src_col, label_col):
    print("\n🔬 Strategy 3: Associative Propagated Risk (2-Hop)")
    t0 = time.time()

    seeds = {n: 1.0 for n in fraud_train if n in G.nodes()}
    ppr_scores = nx.pagerank(G, personalization=seeds if seeds else None)

    ppr_weighted = nx.pagerank(G, personalization=seeds if seeds else None, weight="weight")

    def two_hop_score(node):
        if node not in G:
            return 0.0, 0.0
        neighbors_1 = list(G.successors(node)) + list(G.predecessors(node))
        if not neighbors_1:
            return 0.0, 0.0
        avg_1hop = np.mean([ppr_scores.get(n, 0) for n in neighbors_1])

        neighbors_2 = set()
        for n1 in neighbors_1:
            if n1 in G:
                neighbors_2.update(G.successors(n1))
                neighbors_2.update(G.predecessors(n1))
        neighbors_2.discard(node)
        avg_2hop = np.mean([ppr_scores.get(n, 0) for n in neighbors_2]) if neighbors_2 else 0.0
        return avg_1hop, avg_2hop

    rng = np.random.RandomState(42)
    legit_train = rng.choice(legit_nodes, size=min(len(fraud_train), len(legit_nodes)), replace=False).tolist()
    train_nodes = fraud_train + legit_train

    G_und = G.to_undirected()
    partition = community_louvain.best_partition(G_und)
    comm_counts = defaultdict(int)
    for v in partition.values():
        comm_counts[v] += 1

    X_train_base = extract_enhanced_features(G, train_nodes, ppr_scores, partition, comm_counts)
    X_test_base = extract_enhanced_features(G, test_nodes, ppr_scores, partition, comm_counts)

    def ppr_extra_features(nodes):
        feats = []
        for node in nodes:
            ppr_w = ppr_weighted.get(node, 0)
            a1, a2 = two_hop_score(node)
            ppr_std = ppr_scores.get(node, 0)
            combined = 0.4 * ppr_std + 0.3 * a1 + 0.3 * ppr_w
            feats.append([ppr_w, a1, a2, combined])
        return np.array(feats, dtype=float)

    X_train_ppr = ppr_extra_features(train_nodes)
    X_test_ppr = ppr_extra_features(test_nodes)

    X_train = np.hstack([X_train_base, X_train_ppr])
    X_test = np.hstack([X_test_base, X_test_ppr])

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
        "strategy": "S03_Associative_Propagated_Risk_2hop",
        "dataset": dataset_name,
        "roc_auc": metrics.get("roc_auc", 0),
        "pr_auc": metrics.get("pr_auc", 0),
        "f1": metrics.get("f1", 0),
        "precision": metrics.get("precision", 0),
        "recall": metrics.get("recall", 0),
        "n_features": n_feats,
        "runtime_sec": runtime
    }
