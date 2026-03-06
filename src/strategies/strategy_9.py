import time
import networkx as nx
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from src.graph_utils import compute_common_features, evaluate

def run(G, df, node_labels, fraud_train, legit_nodes, test_nodes, test_labels, dataset_name, src_col, label_col):
    print("\n🔬 Strategy 9: Label Propagation Features")
    t0 = time.time()

    train_nodes, y_train, y_test, X_train_base, X_test_base, _, _, _ = \
        compute_common_features(G, fraud_train, legit_nodes, test_nodes, node_labels, test_labels)

    scores = {}
    for n in G.nodes():
        if n in node_labels:
            if node_labels[n] == 1 and n in fraud_train:
                scores[n] = 1.0
            else:
                scores[n] = 0.0
        else:
            scores[n] = 0.5

    for iteration in range(10):
        new_scores = {}
        for n in G.nodes():
            if n in fraud_train:
                new_scores[n] = scores[n]
                continue
            neighbors = list(G.predecessors(n)) + list(G.successors(n))
            if neighbors:
                new_scores[n] = 0.3 * scores[n] + 0.7 * np.mean([scores.get(nb, 0.5) for nb in neighbors])
            else:
                new_scores[n] = scores[n]
        scores = new_scores

    def lp_features(nodes):
        feats = []
        for node in nodes:
            s = scores.get(node, 0.5)
            neighbors = list(G.predecessors(node)) + list(G.successors(node)) if node in G else []
            if neighbors:
                nb_scores = [scores.get(nb, 0.5) for nb in neighbors]
                nb_mean = np.mean(nb_scores)
                nb_max = np.max(nb_scores)
                nb_std = np.std(nb_scores)
            else:
                nb_mean, nb_max, nb_std = 0.5, 0.5, 0.0
            feats.append([s, nb_mean, nb_max, nb_std, s - nb_mean])
        return np.array(feats, dtype=float)

    X_train_lp = lp_features(train_nodes)
    X_test_lp = lp_features(test_nodes)

    X_train = np.hstack([X_train_base, X_train_lp])
    X_test = np.hstack([X_test_base, X_test_lp])

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
        "strategy": "S9_Label_Propagation",
        "dataset": dataset_name,
        "roc_auc": metrics.get("roc_auc", 0),
        "pr_auc": metrics.get("pr_auc", 0),
        "f1": metrics.get("f1", 0),
        "precision": metrics.get("precision", 0),
        "recall": metrics.get("recall", 0),
        "n_features": n_feats,
        "runtime_sec": runtime
    }
