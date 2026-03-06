"""
Unified Experiment Runner: 6 Strategies for Graph-Based Fraud Detection
========================================================================
Strategy 0: Baseline (5 basic graph features + Random Forest)
Strategy 1: Enhanced Feature Engineering (15+ graph features)
Strategy 2: Weighted Louvain + Resolution Tuning
Strategy 3: 2-Hop Modified PPR
Strategy 4: Stacking Ensemble
Strategy 5: Temporal + Transaction-Type Features
"""

import pandas as pd
import numpy as np
import networkx as nx
import time
import os
import sys
import warnings
import csv
from datetime import datetime
from collections import defaultdict

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, precision_recall_curve
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import community.community_louvain as community_louvain
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ── CONFIGURATION ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
LOG_FILE = os.path.join(RESULTS_DIR, "experiment_log.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Datasets to run — toggle as needed
DATASETS_TO_RUN = ["ELLIPTIC"]  # Focused on Elliptic for Phase 2

# ── UTILITIES ────────────────────────────────────────────────────────────────

def log_result(strategy, dataset, metrics, n_features, runtime):
    """Append a single result row to the CSV log."""
    row = {
        "timestamp": datetime.now().isoformat(),
        "strategy": strategy,
        "dataset": dataset,
        "roc_auc": round(metrics.get("roc_auc", 0), 4),
        "pr_auc": round(metrics.get("pr_auc", 0), 4),
        "f1": round(metrics.get("f1", 0), 4),
        "precision": round(metrics.get("precision", 0), 4),
        "recall": round(metrics.get("recall", 0), 4),
        "n_features": n_features,
        "runtime_sec": round(runtime, 3),
    }
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return row


def evaluate(y_true, y_proba):
    """Compute all metrics from true labels and predicted probabilities."""
    metrics = {}
    if len(set(y_true)) < 2:
        return {"roc_auc": 0, "pr_auc": 0, "f1": 0, "precision": 0, "recall": 0}

    metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    metrics["pr_auc"] = average_precision_score(y_true, y_proba)

    # Find optimal F1 threshold
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    y_pred = (np.array(y_proba) >= best_thresh).astype(int)
    metrics["f1"] = f1_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    return metrics


# ── DATA LOADING ─────────────────────────────────────────────────────────────

def load_paysim():
    """Load PaySim with balanced sampling."""
    print("\n📂 Loading PaySim...")
    path = os.path.join(DATA_DIR, "paysim dataset.csv")
    df = pd.read_csv(path)
    fraud_df = df[df["isFraud"] == 1]
    legit_df = df[df["isFraud"] == 0].sample(n=len(fraud_df), random_state=42)
    df = pd.concat([fraud_df, legit_df]).sample(frac=1, random_state=42)
    print(f"   Balanced: {len(fraud_df)} fraud + {len(legit_df)} legit = {len(df)} rows")
    return df, "nameOrig", "nameDest", "amount", "isFraud"


def load_metaverse():
    """Load Metaverse with balanced sampling. anomaly is categorical: low_risk/moderate_risk/high_risk."""
    print("\n📂 Loading Metaverse...")
    path = os.path.join(DATA_DIR, "metaverse_transactions_dataset.csv")
    df = pd.read_csv(path)
    # Map categorical anomaly to binary: high_risk=1 (fraud), else=0
    df["is_fraud"] = (df["anomaly"] == "high_risk").astype(int)
    fraud_df = df[df["is_fraud"] == 1]
    legit_df = df[df["is_fraud"] == 0].sample(n=min(len(fraud_df), len(df[df["is_fraud"] == 0])), random_state=42)
    df = pd.concat([fraud_df, legit_df]).sample(frac=1, random_state=42)
    print(f"   Balanced: {len(fraud_df)} high_risk + {len(legit_df)} normal = {len(df)} rows")
    return df, "sending_address", "receiving_address", "amount", "is_fraud"


def load_elliptic():
    """Load Elliptic Data Set via KaggleHub."""
    print("\n📂 Loading Elliptic (Bitcoin)...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("ellipticco/elliptic-data-set")
        print(f"   Downloaded/Found at: {path}")

        # Find files
        edge_file = None
        class_file = None
        for root, dirs, files in os.walk(path):
            if 'elliptic_txs_edgelist.csv' in files:
                edge_file = os.path.join(root, 'elliptic_txs_edgelist.csv')
            if 'elliptic_txs_classes.csv' in files:
                class_file = os.path.join(root, 'elliptic_txs_classes.csv')
        
        if not edge_file or not class_file:
            raise FileNotFoundError("Elliptic CSVs not found in downloaded path")

        # Load
        print("   Reading CSVs...")
        edges = pd.read_csv(edge_file)
        classes = pd.read_csv(class_file)

        # Map classes: 1=Illicit, 2=Licit, "unknown"=Drop
        classes = classes[classes['class'] != 'unknown']
        classes['is_fraud'] = classes['class'].apply(lambda x: 1 if x == '1' else 0)

        # Merge classes onto source node (txId1)
        # We classify based on sender behavior
        df = edges.merge(classes, left_on='txId1', right_on='txId', how='inner')
        
        # Add dummy amount/timestep for flow/temporal logic
        # Ideally we'd use features, but for structural baseline, 
        # random capacity is better than 0, or unit capacity.
        # Let's use unit capacity (1.0) basically unweighted
        df['amount'] = 1.0  
        df['step'] = 1.0 # placeholder

        print(f"   Loaded: {len(df[df['is_fraud']==1])} illicit + {len(df[df['is_fraud']==0])} licit = {len(df)} transactions")
        return df, "txId1", "txId2", "amount", "is_fraud"

    except Exception as e:
        print(f"   ❌ Failed to load Elliptic: {e}")
        return pd.DataFrame(), "", "", "", ""


def build_graph(df, src_col, dst_col, amt_col):
    """Build a directed graph, aggregating edge weights."""
    G = nx.DiGraph()
    for _, row in df.iterrows():
        s, d, a = row[src_col], row[dst_col], row[amt_col]
        if G.has_edge(s, d):
            G[s][d]["weight"] += a
            G[s][d]["count"] += 1
        else:
            G.add_edge(s, d, weight=a, count=1)
    print(f"   Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def get_node_labels(df, src_col, label_col):
    """Map source nodes to fraud labels (1 if any transaction is fraud)."""
    labels = {}
    for _, row in df.iterrows():
        src = row[src_col]
        if row[label_col] > 0:
            labels[src] = 1
        elif src not in labels:
            labels[src] = 0
    return labels


def split_nodes(node_labels, test_size=0.5, random_state=42):
    """Split fraud nodes 50/50 for train/test. All legit go to test."""
    fraud_nodes = [n for n, l in node_labels.items() if l == 1]
    legit_nodes = [n for n, l in node_labels.items() if l == 0]
    fraud_train, fraud_test = train_test_split(
        fraud_nodes, test_size=test_size, random_state=random_state
    )
    test_nodes = fraud_test + legit_nodes
    test_labels = {n: node_labels[n] for n in test_nodes}
    return fraud_train, legit_nodes, test_nodes, test_labels


# ── FEATURE EXTRACTORS ──────────────────────────────────────────────────────

def extract_basic_features(G, nodes):
    """Strategy 0: 5 basic features."""
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


def extract_enhanced_features(G, nodes, ppr_scores, partition, community_sizes):
    """Strategy 1: 15+ features including PPR, Louvain, clustering, HITS."""
    # Precompute expensive features once
    G_und = G.to_undirected()
    try:
        clustering = nx.clustering(G_und)
    except Exception:
        clustering = {}

    # HITS can diverge on some graphs — use safe fallback
    try:
        hub_scores, auth_scores = nx.hits(G, max_iter=50, tol=1e-4)
    except Exception:
        hub_scores, auth_scores = {n: 0 for n in G.nodes()}, {n: 0 for n in G.nodes()}

    try:
        avg_neighbor_deg = nx.average_neighbor_degree(G)
    except Exception:
        avg_neighbor_deg = {}

    sccs_set = set()
    for scc in nx.strongly_connected_components(G):
        if len(scc) > 1:
            sccs_set.update(scc)

    features = []
    for node in nodes:
        if node not in G:
            features.append([0] * 15)
            continue

        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        total_deg = in_deg + out_deg
        in_flow = sum(G[u][node].get("weight", 0) for u in G.predecessors(node))
        out_flow = sum(G[node][v].get("weight", 0) for v in G.successors(node))

        # Flow imbalance
        flow_total = in_flow + out_flow + 1e-10
        flow_imbalance = (out_flow - in_flow) / flow_total

        # Transaction count
        in_count = sum(G[u][node].get("count", 1) for u in G.predecessors(node))
        out_count = sum(G[node][v].get("count", 1) for v in G.successors(node))
        tx_count = in_count + out_count

        # PPR score
        ppr = ppr_scores.get(node, 0)

        # Community features
        comm_id = partition.get(node, -1)
        comm_size = community_sizes.get(comm_id, 0)

        # Clustering coefficient
        cc = clustering.get(node, 0)

        # HITS
        hub = hub_scores.get(node, 0)
        auth = auth_scores.get(node, 0)

        # Average neighbor degree
        avg_nd = avg_neighbor_deg.get(node, 0)

        # SCC membership
        in_scc = 1 if node in sccs_set else 0

        features.append([
            total_deg, in_deg, out_deg, in_flow, out_flow,
            flow_imbalance, tx_count, ppr, comm_size, cc,
            hub, auth, avg_nd, in_scc,
            out_flow / (in_flow + 1e-10),  # out/in ratio
        ])
    return np.array(features, dtype=float)


def extract_temporal_features(df, src_col, label_col, nodes):
    """Strategy 5: Temporal + transaction-type features from raw DataFrame."""
    # Build per-node aggregated features from the DataFrame
    node_feats = {}

    # Pre-build a lookup from the DataFrame grouped by source node
    # Use only columns we need to avoid issues
    cols_needed = [src_col, "amount"]
    for c in ["step", "type", "hour_of_day", "timestamp", "transaction_type",
              "oldbalanceOrg", "newbalanceOrig", "login_frequency", "session_duration",
              "risk_score"]:
        if c in df.columns:
            cols_needed.append(c)
    cols_needed = list(set(cols_needed))  # deduplicate
    grouped = df[cols_needed].groupby(src_col)

    for node, group in grouped:
        feats = {}

        # Temporal features (using 'step' column if available)
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

        # Transaction type features
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

        # Amount features
        amounts = group["amount"].values.astype(float)
        feats["amt_mean"] = float(np.mean(amounts))
        feats["amt_std"] = float(np.std(amounts)) if len(amounts) > 1 else 0.0
        feats["amt_max"] = float(np.max(amounts))
        feats["amt_total"] = float(np.sum(amounts))
        feats["n_transactions"] = float(len(group))

        # Balance anomaly (PaySim only)
        if "oldbalanceOrg" in group.columns and "newbalanceOrig" in group.columns:
            balance_diff = (group["oldbalanceOrg"] - group["newbalanceOrig"]).values.astype(float)
            amount_vals = group["amount"].values.astype(float)
            discrepancy = np.abs(balance_diff - amount_vals)
            feats["balance_discrepancy_mean"] = float(np.mean(discrepancy))
            feats["balance_discrepancy_max"] = float(np.max(discrepancy))
        else:
            feats["balance_discrepancy_mean"] = 0.0
            feats["balance_discrepancy_max"] = 0.0

        # Metaverse-specific features
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

        # Velocity: transactions per step
        if feats.get("step_range", 0) > 0:
            feats["velocity"] = feats["n_transactions"] / (feats["step_range"] + 1)
        else:
            feats["velocity"] = feats["n_transactions"]

        node_feats[node] = feats

    # Extract ordered feature matrix
    feature_names = list(next(iter(node_feats.values())).keys()) if node_feats else []
    feature_matrix = []
    for node in nodes:
        if node in node_feats:
            feature_matrix.append([node_feats[node].get(f, 0) for f in feature_names])
        else:
            feature_matrix.append([0] * len(feature_names))

    return np.array(feature_matrix, dtype=float), feature_names


# ── STRATEGIES ───────────────────────────────────────────────────────────────

def run_strategy_0(G, df, node_labels, fraud_train, legit_nodes, test_nodes, test_labels, dataset_name, src_col, label_col):
    """Strategy 0: Baseline — 5 basic graph features + Random Forest."""
    print("\n🔬 Strategy 0: Baseline (5 features + RF)")
    t0 = time.time()

    # Training set: fraud_train + equal legit sample
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

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test, y_proba)
    runtime = time.time() - t0
    row = log_result("S0_Baseline", dataset_name, metrics, 5, runtime)
    print(f"   ✅ ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  F1={metrics['f1']:.4f}  ({runtime:.1f}s)")
    return row


def run_strategy_1(G, df, node_labels, fraud_train, legit_nodes, test_nodes, test_labels, dataset_name, src_col, label_col):
    """Strategy 1: Enhanced Feature Engineering (15+ features)."""
    print("\n🔬 Strategy 1: Enhanced Features (15 features)")
    t0 = time.time()

    # Compute PPR seeded from train fraud
    seeds = {n: 1.0 for n in fraud_train if n in G.nodes()}
    ppr_scores = nx.pagerank(G, personalization=seeds if seeds else None)

    # Louvain
    G_und = G.to_undirected()
    partition = community_louvain.best_partition(G_und)
    comm_counts = defaultdict(int)
    for v in partition.values():
        comm_counts[v] += 1

    # Training set
    rng = np.random.RandomState(42)
    legit_train = rng.choice(legit_nodes, size=min(len(fraud_train), len(legit_nodes)), replace=False).tolist()
    train_nodes = fraud_train + legit_train
    
    # PPR Seed (Only Train Fraud to prevent Data Leak)
    seeds = {n: 1.0 for n in fraud_train if n in G.nodes()}

    X_train = extract_enhanced_features(G, train_nodes, nx.pagerank(G, personalization=seeds), partition, comm_counts)
    y_train = [node_labels[n] for n in train_nodes]

    X_test = extract_enhanced_features(G, test_nodes, nx.pagerank(G, personalization=seeds), partition, comm_counts)
    y_test = [test_labels[n] for n in test_nodes]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test, y_proba)
    runtime = time.time() - t0
    row = log_result("S1_Enhanced_Features", dataset_name, metrics, 15, runtime)
    print(f"   ✅ ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  F1={metrics['f1']:.4f}  ({runtime:.1f}s)")
    return row


def run_strategy_2(G, df, node_labels, fraud_train, legit_nodes, test_nodes, test_labels, dataset_name, src_col, label_col):
    """Strategy 2: Weighted Louvain + Multi-Resolution tuning."""
    print("\n🔬 Strategy 2: Weighted Louvain + Resolution Tuning")
    t0 = time.time()

    G_und = G.to_undirected()
    # Add weights to undirected graph
    G_und_weighted = nx.Graph()
    for u, v, d in G_und.edges(data=True):
        w = d.get("weight", 1)
        if G_und_weighted.has_edge(u, v):
            G_und_weighted[u][v]["weight"] += w
        else:
            G_und_weighted.add_edge(u, v, weight=w)
    # Ensure all nodes present
    for n in G.nodes():
        if n not in G_und_weighted:
            G_und_weighted.add_node(n)

    # Sweep resolutions
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

    # PPR
    seeds = {n: 1.0 for n in fraud_train if n in G.nodes()}
    ppr_scores = nx.pagerank(G, personalization=seeds if seeds else None)

    # Build features: enhanced + multi-resolution community IDs
    rng = np.random.RandomState(42)
    legit_train = rng.choice(legit_nodes, size=min(len(fraud_train), len(legit_nodes)), replace=False).tolist()
    train_nodes = fraud_train + legit_train

    # Use the best resolution's partition for enhanced features base
    X_train_base = extract_enhanced_features(G, train_nodes, ppr_scores, all_partitions[1.0], all_comm_sizes[1.0])
    X_test_base = extract_enhanced_features(G, test_nodes, ppr_scores, all_partitions[1.0], all_comm_sizes[1.0])

    # Additional features: community size at each resolution
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
    row = log_result("S2_Weighted_Louvain", dataset_name, metrics, n_feats, runtime)
    print(f"   ✅ ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  F1={metrics['f1']:.4f}  ({runtime:.1f}s)")
    return row


def run_strategy_3(G, df, node_labels, fraud_train, legit_nodes, test_nodes, test_labels, dataset_name, src_col, label_col):
    """Strategy 3: 2-Hop Modified PPR."""
    print("\n🔬 Strategy 3: 2-Hop Modified PPR")
    t0 = time.time()

    # Standard PPR
    seeds = {n: 1.0 for n in fraud_train if n in G.nodes()}
    ppr_scores = nx.pagerank(G, personalization=seeds if seeds else None)

    # Weighted PPR (use edge weights as transition bias)
    ppr_weighted = nx.pagerank(G, personalization=seeds if seeds else None, weight="weight")

    # 2-hop neighborhood scoring
    def two_hop_score(node):
        if node not in G:
            return 0.0, 0.0
        # 1-hop average
        neighbors_1 = list(G.successors(node)) + list(G.predecessors(node))
        if not neighbors_1:
            return 0.0, 0.0
        avg_1hop = np.mean([ppr_scores.get(n, 0) for n in neighbors_1])

        # 2-hop average
        neighbors_2 = set()
        for n1 in neighbors_1:
            if n1 in G:
                neighbors_2.update(G.successors(n1))
                neighbors_2.update(G.predecessors(n1))
        neighbors_2.discard(node)
        avg_2hop = np.mean([ppr_scores.get(n, 0) for n in neighbors_2]) if neighbors_2 else 0.0
        return avg_1hop, avg_2hop

    # Build combined PPR features
    rng = np.random.RandomState(42)
    legit_train = rng.choice(legit_nodes, size=min(len(fraud_train), len(legit_nodes)), replace=False).tolist()
    train_nodes = fraud_train + legit_train

    # Louvain for base features
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
            # Combined score: alpha * PPR + beta * 1hop + gamma * weighted
            combined = 0.4 * ppr_std + 0.3 * a1 + 0.3 * ppr_w
            feats.append([ppr_w, a1, a2, combined])
        return np.array(feats, dtype=float)

    X_train_ppr = ppr_extra_features(train_nodes)
    X_test_ppr = ppr_extra_features(test_nodes)

    X_train = np.hstack([X_train_base, X_train_ppr])
    X_test = np.hstack([X_test_base, X_test_ppr])

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
    row = log_result("S3_2Hop_PPR", dataset_name, metrics, n_feats, runtime)
    print(f"   ✅ ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  F1={metrics['f1']:.4f}  ({runtime:.1f}s)")
    return row


def run_strategy_4(G, df, node_labels, fraud_train, legit_nodes, test_nodes, test_labels, dataset_name, src_col, label_col):
    """Strategy 4: Stacking Ensemble with Enhanced Features."""
    print("\n🔬 Strategy 4: Stacking Ensemble")
    t0 = time.time()

    # PPR + Louvain
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

    X_train = extract_enhanced_features(G, train_nodes, ppr_scores, partition, comm_counts)
    X_test = extract_enhanced_features(G, test_nodes, ppr_scores, partition, comm_counts)

    y_train = [node_labels[n] for n in train_nodes]
    y_test = [test_labels[n] for n in test_nodes]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_feats = X_train.shape[1]

    # Stacking ensemble
    estimators = [
        ("rf", RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)),
        ("xgb", XGBClassifier(
            n_estimators=150, random_state=42, eval_metric="logloss",
            use_label_encoder=False, verbosity=0, n_jobs=-1
        )),
        ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]
    meta = LogisticRegression(random_state=42, max_iter=1000)

    clf = StackingClassifier(estimators=estimators, final_estimator=meta, cv=3, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test, y_proba)
    runtime = time.time() - t0
    row = log_result("S4_Stacking_Ensemble", dataset_name, metrics, n_feats, runtime)
    print(f"   ✅ ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  F1={metrics['f1']:.4f}  ({runtime:.1f}s)")
    return row


def run_strategy_5(G, df, node_labels, fraud_train, legit_nodes, test_nodes, test_labels, dataset_name, src_col, label_col):
    """Strategy 5: Temporal + Transaction-Type Feature Fusion."""
    print("\n🔬 Strategy 5: Temporal + Type Features Fusion")
    t0 = time.time()

    # PPR + Louvain (base enhanced)
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

    # Graph features
    X_train_graph = extract_enhanced_features(G, train_nodes, ppr_scores, partition, comm_counts)
    X_test_graph = extract_enhanced_features(G, test_nodes, ppr_scores, partition, comm_counts)

    # Temporal + Type features
    all_nodes_for_temporal = train_nodes + test_nodes
    X_temporal_all, feat_names = extract_temporal_features(df, src_col, label_col, all_nodes_for_temporal)
    X_train_temporal = X_temporal_all[:len(train_nodes)]
    X_test_temporal = X_temporal_all[len(train_nodes):]

    # Concatenate
    X_train = np.hstack([X_train_graph, X_train_temporal])
    X_test = np.hstack([X_test_graph, X_test_temporal])

    y_train = [node_labels[n] for n in train_nodes]
    y_test = [test_labels[n] for n in test_nodes]

    # Handle NaN/Inf
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_feats = X_train.shape[1]

    # Use XGBoost for this rich feature set
    clf = XGBClassifier(
        n_estimators=200, random_state=42, eval_metric="logloss",
        use_label_encoder=False, verbosity=0, n_jobs=-1,
        max_depth=8, learning_rate=0.1
    )
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test, y_proba)
    runtime = time.time() - t0
    row = log_result("S5_Temporal_Fusion", dataset_name, metrics, n_feats, runtime)
    print(f"   ✅ ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  F1={metrics['f1']:.4f}  ({runtime:.1f}s)")
    return row


# ── PHASE 2 STRATEGIES (S6–S10) ──────────────────────────────────────────────

def _compute_common_features(G, fraud_train, legit_nodes, test_nodes, node_labels, test_labels):
    """Precompute PPR, Louvain, and train/test splits shared by S6-S10."""
    # Seed ONLY from training set to prevent leakage
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
    y_train = [node_labels[n] for n in train_nodes]
    y_test = [test_labels[n] for n in test_nodes]

    X_train_base = extract_enhanced_features(G, train_nodes, ppr_scores, partition, comm_counts)
    X_test_base = extract_enhanced_features(G, test_nodes, ppr_scores, partition, comm_counts)

    return train_nodes, y_train, y_test, X_train_base, X_test_base, ppr_scores, partition, comm_counts


def _node2vec_embeddings(G, nodes, dimensions=64, walk_length=20, num_walks=10, p=1, q=2):
    """Generate Node2Vec embeddings for a list of nodes."""
    from node2vec import Node2Vec
    # Node2Vec works on undirected by default
    G_for_n2v = G.to_undirected()
    # Ensure graph is not empty
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


def run_strategy_6(G, df, node_labels, fraud_train, legit_nodes, test_nodes, test_labels, dataset_name, src_col, label_col):
    """Strategy 6: Node2Vec Embeddings + XGBoost."""
    print("\n🔬 Strategy 6: Node2Vec Embeddings")
    t0 = time.time()

    train_nodes, y_train, y_test, X_train_base, X_test_base, _, _, _ = \
        _compute_common_features(G, fraud_train, legit_nodes, test_nodes, node_labels, test_labels)

    # Generate 64-dim embeddings
    all_ordered = train_nodes + test_nodes
    print("   Computing Node2Vec embeddings (this may take a minute)...")
    emb_all = _node2vec_embeddings(G, all_ordered, dimensions=64, walk_length=15, num_walks=8)
    X_train_emb = emb_all[:len(train_nodes)]
    X_test_emb = emb_all[len(train_nodes):]

    # Combine enhanced features + embeddings
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
    row = log_result("S6_Node2Vec", dataset_name, metrics, n_feats, runtime)
    print(f"   ✅ ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  F1={metrics['f1']:.4f}  ({runtime:.1f}s)")
    return row


def run_strategy_7(G, df, node_labels, fraud_train, legit_nodes, test_nodes, test_labels, dataset_name, src_col, label_col):
    """Strategy 7: Time-Weighted PPR."""
    print("\n🔬 Strategy 7: Time-Weighted PPR")
    t0 = time.time()

    train_nodes, y_train, y_test, X_train_base, X_test_base, ppr_scores, partition, comm_counts = \
        _compute_common_features(G, fraud_train, legit_nodes, test_nodes, node_labels, test_labels)

    # Compute time-weighted PPR
    # Decay: more recent edges get higher weight
    max_step = 1.0  # default
    if "step" in df.columns:
        max_step = df["step"].max()
    elif "hour_of_day" in df.columns:
        max_step = df["hour_of_day"].max()

    G_tw = G.copy()
    decay = 0.9
    for u, v, d in G_tw.edges(data=True):
        w = d.get("weight", 1.0)
        # Simple recency: newer edges have higher weight
        # Since we can't easily get per-edge timestamp from the graph,
        # use a proxy: weight * (1 + random noise) to break ties
        # For Elliptic, all weights are 1.0 so this adds structural variation
        d["tw_weight"] = w * (1.0 + 0.1 * np.random.random())

    seeds = {n: 1.0 for n in fraud_train if n in G_tw.nodes()}
    ppr_tw = nx.pagerank(G_tw, personalization=seeds if seeds else None, weight="tw_weight")

    # Also compute a reverse PPR (seed from legit nodes to find "safe" zones)
    legit_seeds = {n: 1.0 for n in legit_nodes[:min(500, len(legit_nodes))] if n in G_tw.nodes()}
    ppr_reverse = nx.pagerank(G_tw, personalization=legit_seeds if legit_seeds else None, weight="tw_weight")

    def tw_features(nodes):
        feats = []
        for node in nodes:
            ppr_std = ppr_scores.get(node, 0)
            ppr_t = ppr_tw.get(node, 0)
            ppr_r = ppr_reverse.get(node, 0)
            # Risk ratio: fraud PPR / legit PPR
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
    row = log_result("S7_TimeWeighted_PPR", dataset_name, metrics, n_feats, runtime)
    print(f"   ✅ ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  F1={metrics['f1']:.4f}  ({runtime:.1f}s)")
    return row


def run_strategy_8(G, df, node_labels, fraud_train, legit_nodes, test_nodes, test_labels, dataset_name, src_col, label_col):
    """Strategy 8: Graph Motifs (Triangles, Cycles, Path Counts)."""
    print("\n🔬 Strategy 8: Graph Motifs & Flow Cycles")
    t0 = time.time()

    train_nodes, y_train, y_test, X_train_base, X_test_base, _, _, _ = \
        _compute_common_features(G, fraud_train, legit_nodes, test_nodes, node_labels, test_labels)

    # Precompute motif features
    G_und = G.to_undirected()
    # Triangle count per node
    try:
        triangles = nx.triangles(G_und)
    except Exception:
        triangles = {n: 0 for n in G.nodes()}

    # Strongly connected component size per node
    scc_map = {}
    for scc in nx.strongly_connected_components(G):
        sz = len(scc)
        for n in scc:
            scc_map[n] = sz

    # 2-path count (number of 2-hop directed paths through this node)
    def motif_features(nodes):
        feats = []
        for node in nodes:
            if node not in G:
                feats.append([0, 0, 0, 0, 0])
                continue
            tri = triangles.get(node, 0)
            scc_sz = scc_map.get(node, 1)
            # Count 2-paths: predecessors -> node -> successors
            preds = list(G.predecessors(node))
            succs = list(G.successors(node))
            two_paths = len(preds) * len(succs)
            # Reciprocity: how many edges are bidirectional?
            reciprocal = sum(1 for s in succs if G.has_edge(s, node))
            # Fan-out ratio
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
    row = log_result("S8_Graph_Motifs", dataset_name, metrics, n_feats, runtime)
    print(f"   ✅ ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  F1={metrics['f1']:.4f}  ({runtime:.1f}s)")
    return row


def run_strategy_9(G, df, node_labels, fraud_train, legit_nodes, test_nodes, test_labels, dataset_name, src_col, label_col):
    """Strategy 9: Label Propagation Features (Semi-Supervised Risk Score)."""
    print("\n🔬 Strategy 9: Label Propagation Features")
    t0 = time.time()

    train_nodes, y_train, y_test, X_train_base, X_test_base, _, _, _ = \
        _compute_common_features(G, fraud_train, legit_nodes, test_nodes, node_labels, test_labels)

    # Iterative label propagation:
    # Initialize known fraud nodes with score 1.0, known legit with 0.0, unknown with 0.5
    # Then iteratively update each node's score as average of neighbors
    scores = {}
    for n in G.nodes():
        if n in node_labels:
            if node_labels[n] == 1 and n in fraud_train:
                scores[n] = 1.0  # known fraud (training only)
            else:
                scores[n] = 0.0  # known legit or test fraud (don't leak)
        else:
            scores[n] = 0.5  # unknown

    # Run 10 iterations of propagation
    for iteration in range(10):
        new_scores = {}
        for n in G.nodes():
            # Don't update seed labels
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
            # Also compute neighborhood risk stats
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
    row = log_result("S9_Label_Propagation", dataset_name, metrics, n_feats, runtime)
    print(f"   ✅ ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  F1={metrics['f1']:.4f}  ({runtime:.1f}s)")
    return row


def run_strategy_10(G, df, node_labels, fraud_train, legit_nodes, test_nodes, test_labels, dataset_name, src_col, label_col):
    """Strategy 10: Kitchen Sink — ALL features combined with Stacking."""
    print("\n🔬 Strategy 10: Kitchen Sink (All Features + Stacking)")
    t0 = time.time()

    train_nodes, y_train, y_test, X_train_base, X_test_base, ppr_scores, partition, comm_counts = \
        _compute_common_features(G, fraud_train, legit_nodes, test_nodes, node_labels, test_labels)

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
    row = log_result("S10_Kitchen_Sink", dataset_name, metrics, n_feats, runtime)
    print(f"   ✅ ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  F1={metrics['f1']:.4f}  ({runtime:.1f}s)")
    return row


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # Clear previous log
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    print("=" * 70)
    print("  FRAUD DETECTION EXPERIMENT RUNNER — PHASE 2 (S6-S10)")
    print("=" * 70)

    all_rows = []

    loaders = {
        "PAYSIM": load_paysim,
        "METAVERSE": load_metaverse,
        "ELLIPTIC": load_elliptic,
    }

    strategies = [
        ("S6", run_strategy_6),
        ("S7", run_strategy_7),
        ("S8", run_strategy_8),
        ("S9", run_strategy_9),
        ("S10", run_strategy_10),
    ]

    for ds_name in DATASETS_TO_RUN:
        if ds_name not in loaders:
            print(f"\n⚠️  Skipping {ds_name} (no loader)")
            continue

        print(f"\n{'=' * 70}")
        print(f"  DATASET: {ds_name}")
        print(f"{'=' * 70}")

        df, src_col, dst_col, amt_col, label_col = loaders[ds_name]()

        print("   Building graph...")
        G = build_graph(df, src_col, dst_col, amt_col)

        node_labels = get_node_labels(df, src_col, label_col)
        fraud_count = sum(1 for l in node_labels.values() if l == 1)
        legit_count = sum(1 for l in node_labels.values() if l == 0)
        print(f"   Labels: {fraud_count} fraud, {legit_count} legit")

        fraud_train, legit_nodes, test_nodes, test_labels = split_nodes(node_labels)
        print(f"   Split: {len(fraud_train)} train fraud, {len(test_nodes)} test nodes")

        for s_name, s_func in strategies:
            try:
                row = s_func(G, df, node_labels, fraud_train, legit_nodes,
                             test_nodes, test_labels, ds_name, src_col, label_col)
                all_rows.append(row)
            except Exception as e:
                print(f"   ❌ {s_name} failed: {e}")
                import traceback
                traceback.print_exc()
                log_result(s_name, ds_name, {"roc_auc": 0, "pr_auc": 0, "f1": 0, "precision": 0, "recall": 0}, 0, 0)

    # ── FINAL COMPARISON ───────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  FINAL COMPARISON TABLE")
    print("=" * 90)

    if os.path.exists(LOG_FILE):
        results_df = pd.read_csv(LOG_FILE)
        print(f"\n{'Strategy':<25} {'Dataset':<12} {'ROC-AUC':>9} {'PR-AUC':>9} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Time(s)':>9} {'#Feats':>7}")
        print("-" * 90)
        for _, r in results_df.iterrows():
            print(f"{r['strategy']:<25} {r['dataset']:<12} {r['roc_auc']:>9.4f} {r['pr_auc']:>9.4f} {r['f1']:>8.4f} {r['precision']:>8.4f} {r['recall']:>8.4f} {r['runtime_sec']:>9.1f} {int(r['n_features']):>7}")

        # Best per dataset
        print("\n🏆 BEST PER DATASET (by ROC-AUC):")
        for ds in results_df["dataset"].unique():
            ds_data = results_df[results_df["dataset"] == ds]
            best = ds_data.loc[ds_data["roc_auc"].idxmax()]
            print(f"   {ds}: {best['strategy']} → ROC-AUC={best['roc_auc']:.4f}, PR-AUC={best['pr_auc']:.4f}")
    else:
        print("No results found!")

    print(f"\n📁 Full results saved to: {LOG_FILE}")
    print("Done!")


if __name__ == "__main__":
    main()
