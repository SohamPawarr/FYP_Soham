import time
import networkx as nx
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.graph_utils import compute_common_features, evaluate

def run(G, df, node_labels, fraud_train, legit_nodes, test_nodes, test_labels, dataset_name, src_col, label_col):
    print("\n🔬 Strategy 10: Kitchen Sink (All Features + Stacking)")
    t0 = time.time()

    train_nodes, y_train, y_test, X_train_base, X_test_base, ppr_scores, partition, comm_counts = \
        compute_common_features(G, fraud_train, legit_nodes, test_nodes, node_labels, test_labels)

    # === Motif features ===
    G_und = G.to_undirected()
    try:
        triangles = nx.triangles(G_und)
    except Exception:
        triangles = {n: 0 for n in G.nodes()}
    scc_map = {}
    for scc in nx.strongly_connected_components(G):
        for n in scc:
            scc_map[n] = len(scc)

    def motif_feats(nodes):
        feats = []
        for node in nodes:
            if node not in G:
                feats.append([0, 0, 0, 0])
                continue
            tri = triangles.get(node, 0)
            scc_sz = scc_map.get(node, 1)
            preds = list(G.predecessors(node))
            succs = list(G.successors(node))
            two_paths = len(preds) * len(succs)
            reciprocal = sum(1 for s in succs if G.has_edge(s, node))
            feats.append([tri, scc_sz, two_paths, reciprocal])
        return np.array(feats, dtype=float)

    # === Label Propagation scores ===
    scores = {}
    for n in G.nodes():
        if n in node_labels and node_labels[n] == 1 and n in fraud_train:
            scores[n] = 1.0
        else:
            scores[n] = 0.0 if (n in node_labels and node_labels[n] == 0) else 0.5
    for _ in range(10):
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

    def lp_feats(nodes):
        feats = []
        for node in nodes:
            s = scores.get(node, 0.5)
            neighbors = list(G.predecessors(node)) + list(G.successors(node)) if node in G else []
            nb_scores = [scores.get(nb, 0.5) for nb in neighbors] if neighbors else [0.5]
            feats.append([s, np.mean(nb_scores), np.max(nb_scores)])
        return np.array(feats, dtype=float)

    # === Time-weighted PPR ===
    seeds = {n: 1.0 for n in fraud_train if n in G.nodes()}
    ppr_tw = nx.pagerank(G, personalization=seeds if seeds else None, weight="weight")
    legit_seeds = {n: 1.0 for n in legit_nodes[:min(500, len(legit_nodes))] if n in G.nodes()}
    ppr_rev = nx.pagerank(G, personalization=legit_seeds if legit_seeds else None, weight="weight")

    def tw_feats(nodes):
        feats = []
        for node in nodes:
            pt = ppr_tw.get(node, 0)
            pr = ppr_rev.get(node, 0)
            feats.append([pt, pr, (pt + 1e-10) / (pr + 1e-10)])
        return np.array(feats, dtype=float)

    # === Assemble all features ===
    X_train_motif = motif_feats(train_nodes)
    X_test_motif = motif_feats(test_nodes)
    X_train_lp = lp_feats(train_nodes)
    X_test_lp = lp_feats(test_nodes)
    X_train_tw = tw_feats(train_nodes)
    X_test_tw = tw_feats(test_nodes)

    X_train = np.hstack([X_train_base, X_train_motif, X_train_lp, X_train_tw])
    X_test = np.hstack([X_test_base, X_test_motif, X_test_lp, X_test_tw])

    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_feats = X_train.shape[1]

    # Stacking ensemble
    estimators = [
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ("xgb", XGBClassifier(
            n_estimators=200, random_state=42, eval_metric="logloss",
            use_label_encoder=False, verbosity=0, n_jobs=-1, max_depth=8
        )),
        ("gb", GradientBoostingClassifier(n_estimators=150, random_state=42)),
    ]
    meta = LogisticRegression(random_state=42, max_iter=1000)
    clf = StackingClassifier(estimators=estimators, final_estimator=meta, cv=3, n_jobs=-1)
    
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test, y_proba)
    runtime = time.time() - t0
    
    print(f"   ✅ ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  F1={metrics['f1']:.4f}  ({runtime:.1f}s)")
    
    return {
        "strategy": "S10_Kitchen_Sink",
        "dataset": dataset_name,
        "roc_auc": metrics.get("roc_auc", 0),
        "pr_auc": metrics.get("pr_auc", 0),
        "f1": metrics.get("f1", 0),
        "precision": metrics.get("precision", 0),
        "recall": metrics.get("recall", 0),
        "n_features": n_feats,
        "runtime_sec": runtime
    }
