import time
import networkx as nx
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from src.graph_utils import compute_common_features, evaluate

def run(G, df, node_labels, fraud_train, legit_nodes, test_nodes, test_labels, dataset_name, src_col, label_col):
    print("\n🔬 Strategy 8: Higher Order Flow Motifs")
    t0 = time.time()

    train_nodes, y_train, y_test, X_train_base, X_test_base, _, _, _ = \
        compute_common_features(G, fraud_train, legit_nodes, test_nodes, node_labels, test_labels)

    G_und = G.to_undirected()
    try:
        triangles = nx.triangles(G_und)
    except Exception:
        triangles = {n: 0 for n in G.nodes()}

    scc_map = {}
    for scc in nx.strongly_connected_components(G):
        sz = len(scc)
        for n in scc:
            scc_map[n] = sz

    def motif_features(nodes):
        feats = []
        for node in nodes:
            if node not in G:
                feats.append([0, 0, 0, 0, 0])
                continue
            tri = triangles.get(node, 0)
            scc_sz = scc_map.get(node, 1)
            preds = list(G.predecessors(node))
            succs = list(G.successors(node))
            two_paths = len(preds) * len(succs)
            reciprocal = sum(1 for s in succs if G.has_edge(s, node))
            fan_ratio = len(succs) / (len(preds) + 1e-10)
            feats.append([tri, scc_sz, two_paths, reciprocal, fan_ratio])
        return np.array(feats, dtype=float)

    X_train_motif = motif_features(train_nodes)
    X_test_motif = motif_features(test_nodes)

    X_train = np.hstack([X_train_base, X_train_motif])
    X_test = np.hstack([X_test_base, X_test_motif])

    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_feats = X_train.shape[1]
    clf = XGBClassifier(
        n_estimators=200, random_state=42, eval_metric="logloss",
        use_label_encoder=False, verbosity=0, n_jobs=-1, max_depth=8
    )
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test, y_proba)
    runtime = time.time() - t0
    
    print(f"   ✅ ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  F1={metrics['f1']:.4f}  ({runtime:.1f}s)")
    
    return {
        "strategy": "S08_Higher_Order_Flow_Motifs",
        "dataset": dataset_name,
        "roc_auc": metrics.get("roc_auc", 0),
        "pr_auc": metrics.get("pr_auc", 0),
        "f1": metrics.get("f1", 0),
        "precision": metrics.get("precision", 0),
        "recall": metrics.get("recall", 0),
        "n_features": n_feats,
        "runtime_sec": runtime
    }
