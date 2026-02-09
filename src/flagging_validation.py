# Algorithmic Flagging Validation Script
# Focus: Pattern identification and risk propagation (NOT ML prediction)
#
# Use Case: Given SOME known fraud accounts (from external reports),
# use graph algorithms to FLAG connected suspicious accounts for investigation.

import pandas as pd
import numpy as np
import networkx as nx
import time
import os
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")

print("=" * 70)
print("ALGORITHMIC FLAGGING VALIDATION")
print("Use Case: Risk propagation from known fraud → flag suspicious accounts")
print("=" * 70)

# --- LOAD DATA ---
def load_balanced_paysim():
    """Load PaySim with balanced sampling"""
    print("\n--- Loading PaySim Dataset ---")
    df = pd.read_csv(os.path.join(DATA_DIR, 'paysim dataset.csv'))
    
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
    """Create node -> fraud_label mapping"""
    labels = {}
    for _, row in df.iterrows():
        src = row['nameOrig']
        labels[src] = row['isFraud']
    return labels

# --- METRIC: PRECISION @ K ---
def precision_at_k(ranked_nodes, node_labels, k):
    """What fraction of top-K flagged nodes are actually fraud?"""
    top_k = ranked_nodes[:k]
    fraud_in_top_k = sum(1 for n in top_k if node_labels.get(n, 0) == 1)
    return fraud_in_top_k / k if k > 0 else 0

# --- METRIC: RECALL @ K ---
def recall_at_k(ranked_nodes, node_labels, k):
    """What fraction of all fraud nodes are in top-K?"""
    top_k = set(ranked_nodes[:k])
    total_fraud = sum(1 for n, l in node_labels.items() if l == 1)
    fraud_in_top_k = sum(1 for n in top_k if node_labels.get(n, 0) == 1)
    return fraud_in_top_k / total_fraud if total_fraud > 0 else 0

# --- PERSONALIZED PAGERANK (FLAGGING MODE) ---
def run_ppr_flagging(G, seed_nodes, all_nodes, node_labels):
    """
    PPR as a flagging tool: 
    - Seeds = KNOWN fraud (e.g., 10% from external reports)
    - Output = Ranked list of suspicious accounts
    """
    seeds = {n: 1.0 for n in seed_nodes if n in G.nodes()}
    
    start = time.time()
    ppr_scores = nx.pagerank(G, personalization=seeds if seeds else None)
    dur = time.time() - start
    
    # Rank all nodes by PPR score (excluding seeds - already known)
    non_seed_nodes = [n for n in all_nodes if n not in seed_nodes and n in G.nodes()]
    ranked = sorted(non_seed_nodes, key=lambda n: ppr_scores.get(n, 0), reverse=True)
    
    return ranked, dur

# --- LOUVAIN COMMUNITY DETECTION (FLAGGING MODE) ---
def run_louvain_flagging(G, seed_nodes, node_labels):
    """
    Louvain as a flagging tool:
    - Find communities containing known fraud
    - Flag all accounts in those communities
    """
    import community.community_louvain as community_louvain
    
    G_undirected = G.to_undirected()
    
    start = time.time()
    partition = community_louvain.best_partition(G_undirected)
    dur = time.time() - start
    
    # Find communities with seeds
    seed_communities = set()
    for seed in seed_nodes:
        if seed in partition:
            seed_communities.add(partition[seed])
    
    # Flag all nodes in those communities
    flagged = [n for n, comm in partition.items() 
               if comm in seed_communities and n not in seed_nodes]
    
    return flagged, len(seed_communities), dur

# --- MAIN VALIDATION ---
def main():
    df = load_balanced_paysim()
    G = build_graph(df)
    node_labels = get_node_labels(df)
    
    all_fraud = [n for n, l in node_labels.items() if l == 1]
    all_legit = [n for n, l in node_labels.items() if l == 0]
    all_nodes = list(node_labels.keys())
    
    print(f"\nTotal: {len(all_fraud)} fraud, {len(all_legit)} legit")
    
    # Test with different seed percentages (simulating known fraud from external sources)
    seed_percentages = [0.05, 0.10, 0.20, 0.30]  # 5%, 10%, 20%, 30% known
    
    results = []
    
    for seed_pct in seed_percentages:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {int(seed_pct*100)}% of fraud accounts are KNOWN (from external reports)")
        print(f"{'='*70}")
        
        # Simulate: some fraud accounts are "known" (seeds)
        known_fraud, unknown_fraud = train_test_split(
            all_fraud, test_size=1-seed_pct, random_state=42
        )
        
        print(f"Known fraud (seeds): {len(known_fraud)}")
        print(f"Unknown fraud (to find): {len(unknown_fraud)}")
        print(f"Legit accounts: {len(all_legit)}")
        
        # Unknowns = fraud we're trying to find + all legit
        unknown_nodes = unknown_fraud + all_legit
        
        # --- PPR Flagging ---
        print("\n--- Personalized PageRank Flagging ---")
        ranked_ppr, ppr_time = run_ppr_flagging(G, known_fraud, unknown_nodes, node_labels)
        
        # Calculate Precision@K for various K
        k_values = [100, 500, 1000, 2000]
        for k in k_values:
            p_at_k = precision_at_k(ranked_ppr, node_labels, k)
            r_at_k = recall_at_k(ranked_ppr, node_labels, k)
            print(f"  Top-{k}: Precision={p_at_k:.2%}, Recall={r_at_k:.2%}")
            
            if k == 1000:  # Store main result
                results.append({
                    'Seed %': f"{int(seed_pct*100)}%",
                    'Algorithm': 'PPR',
                    'Precision@1000': f"{p_at_k:.2%}",
                    'Recall@1000': f"{r_at_k:.2%}",
                    'Time (s)': f"{ppr_time:.3f}"
                })
        
        # --- Louvain Flagging ---
        print("\n--- Louvain Community Flagging ---")
        flagged_louvain, n_communities, louv_time = run_louvain_flagging(
            G, known_fraud, node_labels
        )
        
        # Precision = fraud in flagged / total flagged
        fraud_flagged = sum(1 for n in flagged_louvain if node_labels.get(n, 0) == 1)
        precision_louv = fraud_flagged / len(flagged_louvain) if flagged_louvain else 0
        recall_louv = fraud_flagged / len(unknown_fraud) if unknown_fraud else 0
        
        print(f"  Communities with known fraud: {n_communities}")
        print(f"  Total flagged: {len(flagged_louvain)}")
        print(f"  Precision: {precision_louv:.2%}")
        print(f"  Recall: {recall_louv:.2%}")
        
        results.append({
            'Seed %': f"{int(seed_pct*100)}%",
            'Algorithm': 'Louvain',
            'Precision@1000': f"{precision_louv:.2%}",
            'Recall@1000': f"{recall_louv:.2%}",
            'Time (s)': f"{louv_time:.3f}"
        })
    
    # Summary Table
    print("\n" + "=" * 70)
    print("SUMMARY: ALGORITHMIC FLAGGING PERFORMANCE")
    print("=" * 70)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    results_df.to_csv(os.path.join(RESULTS_DIR, 'flagging_validation_results.csv'), index=False)
    print(f"\nSaved to: {os.path.join(RESULTS_DIR, 'flagging_validation_results.csv')}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
Use Case: Banks receive fraud reports (e.g., 10% of fraud is reported).
Graph algorithms help find the REMAINING 90% of connected fraud.

Key Insight: These algorithms are for FLAGGING & INVESTIGATION, not prediction.
- PPR propagates risk from known bad actors to connected accounts
- Louvain identifies fraud RINGS (all accounts in same community as known fraud)

Precision@K = "If investigators check top-K flagged accounts, how many are fraud?"
Recall@K = "What % of all fraud was caught in top-K?"
""")

if __name__ == "__main__":
    main()
