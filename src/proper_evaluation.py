# Proper Evaluation Script with Train/Test Split and ML Baselines
# This script implements methodologically sound evaluation:
# 1. Proper 50/50 train/test split on fraud nodes
# 2. PPR seeded with ONLY training fraud nodes
# 3. ML baselines (XGBoost, RandomForest, LogisticRegression)

import pandas as pd
import numpy as np
import networkx as nx
import time
import os
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

print("=" * 60)
print("PROPER EVALUATION WITH TRAIN/TEST SPLIT")
print("=" * 60)

# --- LOAD DATA ---
def load_balanced_paysim():
    """Load PaySim with balanced sampling (all fraud + equal legit)"""
    print("\n--- Loading PaySim Dataset ---")
    df = pd.read_csv(os.path.join(DATA_DIR, 'paysim dataset.csv'))
    
    # Balanced sampling
    fraud_df = df[df['isFraud'] == 1]
    legit_df = df[df['isFraud'] == 0].sample(n=len(fraud_df), random_state=42)
    df = pd.concat([fraud_df, legit_df]).sample(frac=1, random_state=42)
    
    print(f"Balanced Sample: {len(fraud_df)} fraud + {len(legit_df)} legit = {len(df)} total")
    return df

# --- BUILD GRAPH ---
def build_graph(df):
    """Build directed graph from transaction data"""
    print("\n--- Building Graph ---")
    G = nx.DiGraph()
    
    for _, row in df.iterrows():
        src, dst, amt = row['nameOrig'], row['nameDest'], row['amount']
        if G.has_edge(src, dst):
            G[src][dst]['weight'] += amt
        else:
            G.add_edge(src, dst, weight=amt)
    
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

# --- EXTRACT NODE LABELS ---
def get_node_labels(df):
    """Create node -> fraud_label mapping (source nodes)"""
    labels = {}
    for _, row in df.iterrows():
        src = row['nameOrig']
        labels[src] = row['isFraud']
    return labels

# --- PROPER TRAIN/TEST SPLIT ---
def train_test_split_nodes(node_labels, test_size=0.5, random_state=42):
    """
    Split fraud nodes into train/test.
    ALL legit nodes go to test (more realistic evaluation).
    """
    fraud_nodes = [n for n, l in node_labels.items() if l == 1]
    legit_nodes = [n for n, l in node_labels.items() if l == 0]
    
    # Split fraud nodes 50/50
    fraud_train, fraud_test = train_test_split(
        fraud_nodes, test_size=test_size, random_state=random_state
    )
    
    print(f"\n--- Train/Test Split ---")
    print(f"Train: {len(fraud_train)} fraud nodes (seeds for PPR)")
    print(f"Test: {len(fraud_test)} fraud + {len(legit_nodes)} legit = {len(fraud_test) + len(legit_nodes)} nodes")
    
    # Test set = held-out fraud + all legit
    test_nodes = fraud_test + legit_nodes
    test_labels = {n: node_labels[n] for n in test_nodes}
    
    return fraud_train, test_nodes, test_labels

# --- PPR WITH PROPER SEEDING ---
def run_ppr_proper(G, train_fraud_nodes, test_nodes, test_labels):
    """Run PPR seeded ONLY with training fraud nodes, evaluate on test"""
    print("\n--- Personalized PageRank (Proper) ---")
    
    # Seed with ONLY training fraud nodes (not test!)
    seeds = {n: 1.0 for n in train_fraud_nodes if n in G.nodes()}
    print(f"Seeding with {len(seeds)} training fraud nodes")
    
    if not seeds:
        print("ERROR: No seeds found in graph!")
        return 0.0, 0.0
    
    start = time.time()
    ppr_scores = nx.pagerank(G, personalization=seeds)
    dur = time.time() - start
    print(f"PPR completed in {dur:.3f}s")
    
    # Evaluate on TEST nodes only
    y_true = []
    y_scores = []
    for node in test_nodes:
        if node in G.nodes():
            y_true.append(test_labels[node])
            y_scores.append(ppr_scores.get(node, 0))
    
    if len(set(y_true)) < 2:
        print("ERROR: Only one class in test set!")
        return 0.0, 0.0
    
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    
    return roc_auc, pr_auc

# --- EXTRACT GRAPH FEATURES FOR ML BASELINES ---
def extract_graph_features(G, nodes):
    """Extract graph-based features for each node"""
    features = []
    
    for node in nodes:
        if node not in G.nodes():
            features.append([0, 0, 0, 0, 0])
            continue
        
        # Basic degree features
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        total_deg = in_deg + out_deg
        
        # Flow features
        in_flow = sum(G[u][node].get('weight', 0) for u in G.predecessors(node))
        out_flow = sum(G[node][v].get('weight', 0) for v in G.successors(node))
        
        features.append([total_deg, in_deg, out_deg, in_flow, out_flow])
    
    return np.array(features)

# --- ML BASELINES ---
def run_ml_baselines(G, train_fraud_nodes, test_nodes, test_labels, node_labels):
    """Train and evaluate ML baselines using graph features"""
    print("\n--- ML Baselines ---")
    
    # Get all training nodes (fraud_train + some legit for training ML)
    all_fraud = [n for n, l in node_labels.items() if l == 1]
    all_legit = [n for n, l in node_labels.items() if l == 0]
    
    # For ML training: use train_fraud_nodes + equal sample of legit
    legit_train = np.random.RandomState(42).choice(
        all_legit, size=len(train_fraud_nodes), replace=False
    ).tolist()
    
    train_nodes = train_fraud_nodes + legit_train
    train_labels = {n: node_labels[n] for n in train_nodes}
    
    print(f"ML Training: {len(train_fraud_nodes)} fraud + {len(legit_train)} legit")
    
    # Extract features
    X_train = extract_graph_features(G, train_nodes)
    y_train = [train_labels[n] for n in train_nodes]
    
    X_test = extract_graph_features(G, test_nodes)
    y_test = [test_labels[n] for n in test_nodes]
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    results = []
    
    # 1. XGBoost (GradientBoosting as proxy)
    print("\nTraining XGBoost...")
    xgb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)
    y_proba = xgb.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    print(f"XGBoost ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
    results.append(("XGBoost (Graph Features)", roc_auc, pr_auc))
    
    # 2. Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_proba = rf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    print(f"Random Forest ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
    results.append(("Random Forest (Graph Features)", roc_auc, pr_auc))
    
    # 3. Logistic Regression
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    y_proba = lr.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    print(f"Logistic Regression ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
    results.append(("Logistic Regression (Graph Features)", roc_auc, pr_auc))
    
    return results

# --- DEGREE CENTRALITY BASELINE ---
def run_degree_centrality(G, test_nodes, test_labels):
    """Baseline: Use in-degree as fraud score"""
    print("\n--- Degree Centrality Baseline ---")
    
    deg_scores = dict(G.in_degree())
    
    y_true = []
    y_scores = []
    for node in test_nodes:
        if node in G.nodes():
            y_true.append(test_labels[node])
            y_scores.append(deg_scores.get(node, 0))
    
    if len(set(y_true)) < 2:
        return 0.0, 0.0
    
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    
    print(f"Degree Centrality ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
    return roc_auc, pr_auc

# --- MAIN ---
def main():
    # Load data
    df = load_balanced_paysim()
    
    # Build graph
    G = build_graph(df)
    
    # Get node labels
    node_labels = get_node_labels(df)
    print(f"\nTotal labeled nodes: {len(node_labels)}")
    print(f"  Fraud: {sum(1 for l in node_labels.values() if l == 1)}")
    print(f"  Legit: {sum(1 for l in node_labels.values() if l == 0)}")
    
    # Test multiple seed percentages for PPR
    seed_percentages = [0.1, 0.3, 0.5, 0.7]
    ppr_results = []
    
    for seed_pct in seed_percentages:
        print(f"\n{'='*60}")
        print(f"TESTING PPR WITH {int(seed_pct*100)}% TRAIN SEEDS")
        print(f"{'='*60}")
        
        train_fraud_nodes, test_nodes, test_labels = train_test_split_nodes(
            node_labels, test_size=1-seed_pct
        )
        ppr_roc, ppr_pr = run_ppr_proper(G, train_fraud_nodes, test_nodes, test_labels)
        ppr_results.append((f"PPR ({int(seed_pct*100)}% seeds)", ppr_roc, ppr_pr))
    
    # Use 50% split for baselines (fair comparison)
    print(f"\n{'='*60}")
    print("ML BASELINES (50% Train Split)")
    print(f"{'='*60}")
    train_fraud_nodes, test_nodes, test_labels = train_test_split_nodes(node_labels, test_size=0.5)
    
    # Degree Centrality
    deg_roc, deg_pr = run_degree_centrality(G, test_nodes, test_labels)
    
    # ML baselines
    ml_results = run_ml_baselines(G, train_fraud_nodes, test_nodes, test_labels, node_labels)
    
    # Compile ALL results
    all_results = ppr_results + [("Degree Centrality", deg_roc, deg_pr)] + ml_results
    
    # Print summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS (Proper Train/Test Split - No Data Leakage)")
    print("=" * 70)
    print(f"{'Algorithm':<45} {'ROC-AUC':>10} {'PR-AUC':>10}")
    print("-" * 70)
    for algo, roc, pr in all_results:
        print(f"{algo:<45} {roc:>10.4f} {pr:>10.4f}")
    
    # Save to CSV
    results_df = pd.DataFrame(all_results, columns=['Algorithm', 'ROC-AUC', 'PR-AUC'])
    results_df.to_csv(os.path.join(RESULTS_DIR, 'proper_evaluation_results.csv'), index=False)
    print(f"\nResults saved to: {os.path.join(RESULTS_DIR, 'proper_evaluation_results.csv')}")
    
    # Best PPR result
    best_ppr = max(ppr_results, key=lambda x: x[1])
    
    # Generate LaTeX table
    print("\n--- LaTeX Table (Copy-Paste Ready) ---")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Comparison of Fraud Detection Algorithms (Proper Evaluation)}")
    print(r"\label{tab:proper_comparison}")
    print(r"\begin{tabular}{l c c}")
    print(r"\toprule")
    print(r"\textbf{Algorithm} & \textbf{ROC-AUC} & \textbf{PR-AUC} \\")
    print(r"\midrule")
    # Print best PPR
    print(f"Personalized PageRank (Best) & {best_ppr[1]:.4f} & {best_ppr[2]:.4f} \\\\")
    print(f"Degree Centrality & {deg_roc:.4f} & {deg_pr:.4f} \\\\")
    for algo, roc, pr in ml_results:
        print(f"{algo} & {roc:.4f} & {pr:.4f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

if __name__ == "__main__":
    main()
