import time
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from src.graph_utils import compute_common_features, evaluate

def _node2vec_embeddings(G, nodes, dimensions=64, walk_length=15, num_walks=8, p=1, q=2):
    """Generate Node2Vec embeddings for a list of nodes."""
    from node2vec import Node2Vec
    G_for_n2v = G.to_undirected()
    if G_for_n2v.number_of_nodes() == 0:
        return np.zeros((len(nodes), dimensions))

    n2v = Node2Vec(
        G_for_n2v, dimensions=dimensions, walk_length=walk_length,
        num_walks=num_walks, p=p, q=q, workers=1, quiet=True
    )
    model = n2v.fit(window=5, min_count=1, batch_words=4)
    embeddings = []
    for node in nodes:
        try:
            embeddings.append(model.wv[str(node)])
        except KeyError:
            embeddings.append(np.zeros(dimensions))
    return np.array(embeddings, dtype=float)

def run(G, df, node_labels, fraud_train, legit_nodes, test_nodes, test_labels, dataset_name, src_col, label_col):
    print("\n🔬 Strategy 6: Node2Vec Embeddings")
    t0 = time.time()

    train_nodes, y_train, y_test, X_train_base, X_test_base, _, _, _ = \
        compute_common_features(G, fraud_train, legit_nodes, test_nodes, node_labels, test_labels)

    all_ordered = train_nodes + test_nodes
    print("   Computing Node2Vec embeddings (this may take a minute)...")
    emb_all = _node2vec_embeddings(G, all_ordered, dimensions=64, walk_length=15, num_walks=8)
    X_train_emb = emb_all[:len(train_nodes)]
    X_test_emb = emb_all[len(train_nodes):]

    X_train = np.hstack([X_train_base, X_train_emb])
    X_test = np.hstack([X_test_base, X_test_emb])

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
        "strategy": "S6_Node2Vec",
        "dataset": dataset_name,
        "roc_auc": metrics.get("roc_auc", 0),
        "pr_auc": metrics.get("pr_auc", 0),
        "f1": metrics.get("f1", 0),
        "precision": metrics.get("precision", 0),
        "recall": metrics.get("recall", 0),
        "n_features": n_feats,
        "runtime_sec": runtime
    }
