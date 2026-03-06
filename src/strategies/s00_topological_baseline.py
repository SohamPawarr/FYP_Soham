import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from src.graph_utils import evaluate

def extract_basic_features(G, nodes):
    features = []
    for node in nodes:
        if node not in G:
            features.append([0] * 5)
            continue
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        total_deg = in_deg + out_deg
        in_flow = sum(G[u][node].get("weight", 0) for u in G.predecessors(node))
        out_flow = sum(G[node][v].get("weight", 0) for v in G.successors(node))
        features.append([total_deg, in_deg, out_deg, in_flow, out_flow])
    return np.array(features, dtype=float)

def run(G, df, node_labels, fraud_train, legit_nodes, test_nodes, test_labels, dataset_name, src_col, label_col):
    print("\n🔬 Strategy 0: Topological Baseline")
    t0 = time.time()

    rng = np.random.RandomState(42)
    legit_train = rng.choice(legit_nodes, size=min(len(fraud_train), len(legit_nodes)), replace=False).tolist()
    train_nodes = fraud_train + legit_train

    X_train = extract_basic_features(G, train_nodes)
    y_train = [node_labels[n] for n in train_nodes]

    X_test = extract_basic_features(G, test_nodes)
    y_test = [test_labels[n] for n in test_nodes]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_feats = X_train.shape[1]
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test, y_proba)
    runtime = time.time() - t0
    
    print(f"   ✅ ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  F1={metrics['f1']:.4f}  ({runtime:.1f}s)")
    
    return {
        "strategy": "S00_Topological_Baseline",
        "dataset": dataset_name,
        "roc_auc": metrics.get("roc_auc", 0),
        "pr_auc": metrics.get("pr_auc", 0),
        "f1": metrics.get("f1", 0),
        "precision": metrics.get("precision", 0),
        "recall": metrics.get("recall", 0),
        "n_features": n_feats,
        "runtime_sec": runtime
    }
