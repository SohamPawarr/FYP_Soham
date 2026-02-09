# Deep Dive: Honest Metrics Analysis
# Goal: Understand exactly what's happening and produce defensible numbers

import pandas as pd
import numpy as np
import networkx as nx
import os
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")

print("=" * 70)
print("DEEP DIVE: HONEST METRICS ANALYSIS")
print("=" * 70)

# --- LOAD FULL DATASET (not balanced) ---
print("\n--- Loading FULL PaySim Dataset ---")
df = pd.read_csv(os.path.join(DATA_DIR, 'paysim dataset.csv'))
print(f"Total transactions: {len(df)}")
print(f"Fraud transactions: {df['isFraud'].sum()} ({100*df['isFraud'].mean():.3f}%)")

# --- GRAPH STRUCTURE ANALYSIS ---
print("\n--- Graph Structure Analysis ---")
G = nx.DiGraph()
for _, row in df.iterrows():
    src, dst, amt = row['nameOrig'], row['nameDest'], row['amount']
    if G.has_edge(src, dst):
        G[src][dst]['weight'] += amt
    else:
        G.add_edge(src, dst, weight=amt)

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")

# Get fraud source nodes
fraud_sources = set(df[df['isFraud'] == 1]['nameOrig'].unique())
print(f"\nFraud SOURCE nodes: {len(fraud_sources)}")

# How many fraud sources are connected to OTHER fraud sources?
fraud_to_fraud = 0
fraud_to_legit = 0
for src in fraud_sources:
    if src in G:
        for dst in G.successors(src):
            if dst in fraud_sources:
                fraud_to_fraud += 1
            else:
                fraud_to_legit += 1

print(f"Fraud → Fraud edges: {fraud_to_fraud}")
print(f"Fraud → Non-fraud edges: {fraud_to_legit}")
print(f"Fraud sources connect to OTHER fraud sources: {100*fraud_to_fraud/(fraud_to_fraud+fraud_to_legit):.2f}%")

# --- NODE DEGREE ANALYSIS ---
print("\n--- Degree Analysis ---")
fraud_degrees = [G.out_degree(n) for n in fraud_sources if n in G]
all_degrees = [G.out_degree(n) for n in G.nodes()]

print(f"Fraud node avg out-degree: {np.mean(fraud_degrees):.2f}")
print(f"All nodes avg out-degree: {np.mean(all_degrees):.2f}")

# --- BALANCED SUBGRAPH (Paper's Approach) ---
print("\n" + "=" * 70)
print("BALANCED SUBGRAPH (As described in paper)")
print("=" * 70)

# Take ALL fraud + equal legit (8213 + 8213)
fraud_txns = df[df['isFraud'] == 1]
legit_txns = df[df['isFraud'] == 0].sample(n=len(fraud_txns), random_state=42)
balanced_df = pd.concat([fraud_txns, legit_txns])

print(f"Balanced transactions: {len(balanced_df)}")
print(f"Fraud: {balanced_df['isFraud'].sum()}")
print(f"Legit: {len(balanced_df) - balanced_df['isFraud'].sum()}")

# Build balanced graph
G_bal = nx.DiGraph()
for _, row in balanced_df.iterrows():
    src, dst, amt = row['nameOrig'], row['nameDest'], row['amount']
    if G_bal.has_edge(src, dst):
        G_bal[src][dst]['weight'] += amt
    else:
        G_bal.add_edge(src, dst, weight=amt)

print(f"Balanced graph: {G_bal.number_of_nodes()} nodes, {G_bal.number_of_edges()} edges")

# Node labels (source nodes only)
node_labels = {}
for _, row in balanced_df.iterrows():
    node_labels[row['nameOrig']] = row['isFraud']

labeled_nodes = list(node_labels.keys())
fraud_nodes = [n for n, l in node_labels.items() if l == 1]
legit_nodes = [n for n, l in node_labels.items() if l == 0]

print(f"Labeled SOURCE nodes: {len(labeled_nodes)}")
print(f"  Fraud: {len(fraud_nodes)}")
print(f"  Legit: {len(legit_nodes)}")

# --- RIGOROUS PPR EVALUATION ---
print("\n" + "=" * 70)
print("RIGOROUS PPR EVALUATION (50% Train / 50% Test)")
print("=" * 70)

# Split: 50% fraud as seeds, 50% fraud + all legit as test
fraud_train, fraud_test = train_test_split(fraud_nodes, test_size=0.5, random_state=42)
print(f"Train seeds: {len(fraud_train)} fraud nodes")
print(f"Test set: {len(fraud_test)} fraud + {len(legit_nodes)} legit = {len(fraud_test)+len(legit_nodes)} nodes")

# PPR with train seeds
seeds = {n: 1.0 for n in fraud_train if n in G_bal.nodes()}
ppr_scores = nx.pagerank(G_bal, personalization=seeds)

# Evaluate on TEST nodes only
test_nodes = fraud_test + legit_nodes
y_true = [node_labels[n] for n in test_nodes]
y_scores = [ppr_scores.get(n, 0) for n in test_nodes]

# Find optimal threshold
precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores[:-1])
best_threshold = thresholds[best_idx]

print(f"\nBest threshold: {best_threshold:.6f}")
print(f"At best threshold:")
print(f"  Precision: {precisions[best_idx]:.4f}")
print(f"  Recall: {recalls[best_idx]:.4f}")
print(f"  F1-Score: {f1_scores[best_idx]:.4f}")

# ROC-AUC and PR-AUC on test set
roc_auc = roc_auc_score(y_true, y_scores)
pr_auc = average_precision_score(y_true, y_scores)

print(f"\nROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")

# --- NOW THE PAPER'S SETUP (100% train, same test) ---
print("\n" + "=" * 70)
print("PAPER'S SETUP (100% fraud as seeds, same nodes as test)")
print("NOTE: This is the data leakage scenario")
print("=" * 70)

seeds_all = {n: 1.0 for n in fraud_nodes if n in G_bal.nodes()}
ppr_all = nx.pagerank(G_bal, personalization=seeds_all)

y_true_all = [node_labels[n] for n in labeled_nodes]
y_scores_all = [ppr_all.get(n, 0) for n in labeled_nodes]

roc_auc_all = roc_auc_score(y_true_all, y_scores_all)
pr_auc_all = average_precision_score(y_true_all, y_scores_all)

print(f"ROC-AUC (with leakage): {roc_auc_all:.4f}")
print(f"PR-AUC (with leakage): {pr_auc_all:.4f}")

# --- COMMUNITY DETECTION ---
print("\n" + "=" * 70)
print("COMMUNITY DETECTION (Louvain)")
print("=" * 70)

import community.community_louvain as community_louvain
from sklearn.metrics import homogeneity_score

G_und = G_bal.to_undirected()
partition = community_louvain.best_partition(G_und)

n_communities = len(set(partition.values()))
print(f"Communities detected: {n_communities}")

# Homogeneity score (how well do communities align with fraud labels?)
nodes_with_both = [n for n in partition.keys() if n in node_labels]
true_labels = [node_labels[n] for n in nodes_with_both]
comm_labels = [partition[n] for n in nodes_with_both]

if len(set(true_labels)) > 1:
    homogeneity = homogeneity_score(true_labels, comm_labels)
    print(f"Homogeneity score: {homogeneity:.4f}")
else:
    homogeneity = 0
    print("Cannot compute homogeneity (single class)")

# Community density analysis
comm_fraud_density = {}
for comm_id in set(partition.values()):
    members = [n for n, c in partition.items() if c == comm_id and n in node_labels]
    if len(members) >= 5:
        fraud_in_comm = sum(1 for n in members if node_labels[n] == 1)
        comm_fraud_density[comm_id] = fraud_in_comm / len(members)

high_risk_comms = sum(1 for density in comm_fraud_density.values() if density > 0.9)
print(f"High-risk communities (>90% fraud): {high_risk_comms}")

# --- SCC DETECTION ---
print("\n" + "=" * 70)
print("STRONGLY CONNECTED COMPONENTS (Tarjan)")
print("=" * 70)

sccs = list(nx.strongly_connected_components(G_bal))
non_trivial_sccs = [scc for scc in sccs if len(scc) > 1]

print(f"Total SCCs: {len(sccs)}")
print(f"Non-trivial SCCs (size > 1): {len(non_trivial_sccs)}")

if non_trivial_sccs:
    largest = max(non_trivial_sccs, key=len)
    print(f"Largest SCC size: {len(largest)}")
else:
    print("No non-trivial SCCs (graph is DAG-like)")

# --- FINAL HONEST METRICS ---
print("\n" + "=" * 70)
print("FINAL HONEST METRICS FOR PAPER")
print("=" * 70)

print("""
RECOMMENDED PAPER METRICS (Defensible):

1. PPR Risk Scoring (50% train / 50% test split):
   - ROC-AUC: {:.4f}
   - PR-AUC: {:.4f}
   
2. Community Detection (Louvain):
   - Communities: {}
   - Homogeneity: {:.4f}
   - High-risk communities: {}

3. Cycle Detection (Tarjan):
   - Non-trivial SCCs: {} (DAG structure confirms linear fraud chains)

4. Dataset Info (Balanced Sub-graph):
   - 8,213 fraud + 8,213 legit transactions
   - 32,661 nodes, 16,426 edges
""".format(roc_auc, pr_auc, n_communities, homogeneity, high_risk_comms, len(non_trivial_sccs)))

# Save results
results = {
    'Metric': ['PPR ROC-AUC', 'PPR PR-AUC', 'Communities', 'Homogeneity', 'High-Risk Communities', 'Non-trivial SCCs'],
    'Value': [round(roc_auc, 4), round(pr_auc, 4), n_communities, round(homogeneity, 4), high_risk_comms, len(non_trivial_sccs)]
}
pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, 'honest_metrics.csv'), index=False)
print(f"\nSaved to: {os.path.join(RESULTS_DIR, 'honest_metrics.csv')}")
