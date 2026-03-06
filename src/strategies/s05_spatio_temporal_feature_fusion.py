import time
import numpy as np
import networkx as nx
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import community.community_louvain as community_louvain
from src.graph_utils import extract_enhanced_features, evaluate

def extract_temporal_features(df, src_col, label_col, nodes):
    node_feats = {}

    cols_needed = [src_col, "amount"]
    for c in ["step", "type", "hour_of_day", "timestamp", "transaction_type",
              "oldbalanceOrg", "newbalanceOrig", "login_frequency", "session_duration",
              "risk_score"]:
        if c in df.columns:
            cols_needed.append(c)
    cols_needed = list(set(cols_needed))
    grouped = df[cols_needed].groupby(src_col)

    for node, group in grouped:
        feats = {}

        if "step" in group.columns:
            steps = group["step"].values.astype(float)
            feats["step_mean"] = float(np.mean(steps))
            feats["step_std"] = float(np.std(steps)) if len(steps) > 1 else 0.0
            feats["step_range"] = float(np.ptp(steps))
            feats["n_active_steps"] = float(len(np.unique(steps)))
        elif "hour_of_day" in group.columns:
            hours = group["hour_of_day"].values.astype(float)
            feats["step_mean"] = float(np.mean(hours))
            feats["step_std"] = float(np.std(hours)) if len(hours) > 1 else 0.0
            feats["step_range"] = float(np.ptp(hours))
            feats["n_active_steps"] = float(len(np.unique(hours)))
        else:
            feats["step_mean"] = 0.0
            feats["step_std"] = 0.0
            feats["step_range"] = 0.0
            feats["n_active_steps"] = 0.0

        if "type" in group.columns:
            type_counts = group["type"].value_counts(normalize=True)
            feats["frac_TRANSFER"] = float(type_counts.get("TRANSFER", 0))
            feats["frac_CASH_OUT"] = float(type_counts.get("CASH_OUT", 0))
            feats["frac_PAYMENT"] = float(type_counts.get("PAYMENT", 0))
            feats["n_tx_types"] = float(len(type_counts))
        elif "transaction_type" in group.columns:
            type_counts = group["transaction_type"].value_counts(normalize=True)
            feats["frac_TRANSFER"] = float(type_counts.iloc[0]) if len(type_counts) > 0 else 0.0
            feats["frac_CASH_OUT"] = float(type_counts.iloc[1]) if len(type_counts) > 1 else 0.0
            feats["frac_PAYMENT"] = float(type_counts.iloc[2]) if len(type_counts) > 2 else 0.0
            feats["n_tx_types"] = float(len(type_counts))
        else:
            feats["frac_TRANSFER"] = 0.0
            feats["frac_CASH_OUT"] = 0.0
            feats["frac_PAYMENT"] = 0.0
            feats["n_tx_types"] = 0.0

        amounts = group["amount"].values.astype(float)
        feats["amt_mean"] = float(np.mean(amounts))
        feats["amt_std"] = float(np.std(amounts)) if len(amounts) > 1 else 0.0
        feats["amt_max"] = float(np.max(amounts))
        feats["amt_total"] = float(np.sum(amounts))
        feats["n_transactions"] = float(len(group))

        if "oldbalanceOrg" in group.columns and "newbalanceOrig" in group.columns:
            balance_diff = (group["oldbalanceOrg"] - group["newbalanceOrig"]).values.astype(float)
            amount_vals = group["amount"].values.astype(float)
            discrepancy = np.abs(balance_diff - amount_vals)
            feats["balance_discrepancy_mean"] = float(np.mean(discrepancy))
            feats["balance_discrepancy_max"] = float(np.max(discrepancy))
        else:
            feats["balance_discrepancy_mean"] = 0.0
            feats["balance_discrepancy_max"] = 0.0

        if "login_frequency" in group.columns:
            feats["login_freq"] = float(group["login_frequency"].mean())
        else:
            feats["login_freq"] = 0.0
        if "session_duration" in group.columns:
            feats["session_dur"] = float(group["session_duration"].mean())
        else:
            feats["session_dur"] = 0.0
        if "risk_score" in group.columns:
            rs = pd.to_numeric(group["risk_score"], errors="coerce").fillna(0)
            feats["risk_score_mean"] = float(rs.mean())
        else:
            feats["risk_score_mean"] = 0.0

        if feats.get("step_range", 0) > 0:
            feats["velocity"] = feats["n_transactions"] / (feats["step_range"] + 1)
        else:
            feats["velocity"] = feats["n_transactions"]

        node_feats[node] = feats

    feature_names = list(next(iter(node_feats.values())).keys()) if node_feats else []
    feature_matrix = []
    for node in nodes:
        if node in node_feats:
            feature_matrix.append([node_feats[node].get(f, 0) for f in feature_names])
        else:
            feature_matrix.append([0] * len(feature_names))

    return np.array(feature_matrix, dtype=float), feature_names

def run(G, df, node_labels, fraud_train, legit_nodes, test_nodes, test_labels, dataset_name, src_col, label_col):
    print("\n🔬 Strategy 5: Spatio-Temporal Feature Fusion")
    t0 = time.time()

    seeds = {n: 1.0 for n in fraud_train if n in G.nodes()}
    ppr_scores = nx.pagerank(G, personalization=seeds if seeds else None)
    G_und = G.to_undirected()
    partition = community_louvain.best_partition(G_und)
    comm_counts = defaultdict(int)
    for v in partition.values():
        comm_counts[v] += 1

    rng = np.random.RandomState(42)
    legit_train = rng.choice(legit_nodes, size=min(len(fraud_train), len(legit_nodes)), replace=False).tolist()
    train_nodes = fraud_train + legit_train

    X_train_graph = extract_enhanced_features(G, train_nodes, ppr_scores, partition, comm_counts)
    X_test_graph = extract_enhanced_features(G, test_nodes, ppr_scores, partition, comm_counts)

    all_nodes_for_temporal = train_nodes + test_nodes
    X_temporal_all, feat_names = extract_temporal_features(df, src_col, label_col, all_nodes_for_temporal)
    X_train_temporal = X_temporal_all[:len(train_nodes)]
    X_test_temporal = X_temporal_all[len(train_nodes):]

    X_train = np.hstack([X_train_graph, X_train_temporal])
    X_test = np.hstack([X_test_graph, X_test_temporal])

    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    y_train = [node_labels[n] for n in train_nodes]
    y_test = [test_labels[n] for n in test_nodes]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_feats = X_train.shape[1]

    clf = XGBClassifier(
        n_estimators=200, random_state=42, eval_metric="logloss",
        use_label_encoder=False, verbosity=0, n_jobs=-1,
        max_depth=8, learning_rate=0.1
    )
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test, y_proba)
    runtime = time.time() - t0
    
    print(f"   ✅ ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  F1={metrics['f1']:.4f}  ({runtime:.1f}s)")
    
    return {
        "strategy": "S05_Spatio_Temporal_Feature_Fusion",
        "dataset": dataset_name,
        "roc_auc": metrics.get("roc_auc", 0),
        "pr_auc": metrics.get("pr_auc", 0),
        "f1": metrics.get("f1", 0),
        "precision": metrics.get("precision", 0),
        "recall": metrics.get("recall", 0),
        "n_features": n_feats,
        "runtime_sec": runtime
    }
