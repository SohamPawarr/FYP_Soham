import time
import networkx as nx
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from src.graph_utils import compute_common_features, evaluate

def run(G, df, node_labels, fraud_train, legit_nodes, test_nodes, test_labels, dataset_name, src_col, label_col):
    print("\n🔬 Strategy 7: Chronological Risk Propagation")
    t0 = time.time()

    train_nodes, y_train, y_test, X_train_base, X_test_base, ppr_scores, partition, comm_counts = \
        compute_common_features(G, fraud_train, legit_nodes, test_nodes, node_labels, test_labels)

    max_step = 1.0
    if "step" in df.columns:
        max_step = df["step"].max()
    elif "hour_of_day" in df.columns:
        max_step = df["hour_of_day"].max()

    G_tw = G.copy()
    for u, v, d in G_tw.edges(data=True):
        w = d.get("weight", 1.0)
        d["tw_weight"] = w * (1.0 + 0.1 * np.random.random())

    seeds = {n: 1.0 for n in fraud_train if n in G_tw.nodes()}
    ppr_tw = nx.pagerank(G_tw, personalization=seeds if seeds else None, weight="tw_weight")

    legit_seeds = {n: 1.0 for n in legit_nodes[:min(500, len(legit_nodes))] if n in G_tw.nodes()}
    ppr_reverse = nx.pagerank(G_tw, personalization=legit_seeds if legit_seeds else None, weight="tw_weight")

    def tw_features(nodes):
        feats = []
        for node in nodes:
            ppr_std = ppr_scores.get(node, 0)
            ppr_t = ppr_tw.get(node, 0)
            ppr_r = ppr_reverse.get(node, 0)
            risk_ratio = (ppr_t + 1e-10) / (ppr_r + 1e-10)
            feats.append([ppr_t, ppr_r, risk_ratio, ppr_t - ppr_r])
        return np.array(feats, dtype=float)

    X_train_tw = tw_features(train_nodes)
    X_test_tw = tw_features(test_nodes)

    X_train = np.hstack([X_train_base, X_train_tw])
    X_test = np.hstack([X_test_base, X_test_tw])

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
        "strategy": "S07_Chronological_Risk_Propagation",
        "dataset": dataset_name,
        "roc_auc": metrics.get("roc_auc", 0),
        "pr_auc": metrics.get("pr_auc", 0),
        "f1": metrics.get("f1", 0),
        "precision": metrics.get("precision", 0),
        "recall": metrics.get("recall", 0),
        "n_features": n_feats,
        "runtime_sec": runtime
    }
