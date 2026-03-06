import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from src.graph_utils import compute_common_features, evaluate

def run(G, df, node_labels, fraud_train, legit_nodes, test_nodes, test_labels, dataset_name, src_col, label_col):
    print("\n🔬 Strategy 1: Enhanced Features (15 features)")
    t0 = time.time()

    train_nodes, y_train, y_test, X_train_base, X_test_base, _, _, _ = \
        compute_common_features(G, fraud_train, legit_nodes, test_nodes, node_labels, test_labels)

    X_train = np.nan_to_num(X_train_base, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test_base, nan=0, posinf=0, neginf=0)

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
        "strategy": "S1_Enhanced_Features",
        "dataset": dataset_name,
        "roc_auc": metrics.get("roc_auc", 0),
        "pr_auc": metrics.get("pr_auc", 0),
        "f1": metrics.get("f1", 0),
        "precision": metrics.get("precision", 0),
        "recall": metrics.get("recall", 0),
        "n_features": n_feats,
        "runtime_sec": runtime
    }
