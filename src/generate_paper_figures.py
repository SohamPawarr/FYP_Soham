import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import community.community_louvain as community_louvain
import os
import random

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results") # Images will go here or root? 
# Paper usually expects images in same dir or specific folder. 
# Let's save to project root where .tex likely looks, or checks assumption.
# The tex has \includegraphics[width=\columnwidth]{pr_curve.png} -> implies current dir relative to tex.
# User tex is in src/paper.tex, so images should be in src/
IMG_DIR = SCRIPT_DIR 

print(f"Generating images in: {IMG_DIR}")

def load_data():
    print("Loading PaySim data...")
    df = pd.read_csv(os.path.join(DATA_DIR, 'paysim dataset.csv'))
    # Balanced subset as per paper methodology
    fraud = df[df['isFraud'] == 1]
    legit = df[df['isFraud'] == 0].sample(n=len(fraud), random_state=42)
    balanced_df = pd.concat([fraud, legit])
    return balanced_df

def generate_density_plot(df):
    print("Generating density_plot.png...")
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    # Re-implement logic
    G = nx.from_pandas_edgelist(df, 'nameOrig', 'nameDest', ['amount', 'isFraud'])
    partition = community_louvain.best_partition(G)
    
    community_counts = {}
    community_fraud = {}
    fraud_nodes = set(df[df['isFraud'] == 1]['nameOrig']) | set(df[df['isFraud'] == 1]['nameDest'])
    
    for node, comm_id in partition.items():
        community_counts[comm_id] = community_counts.get(comm_id, 0) + 1
        if node in fraud_nodes:
            community_fraud[comm_id] = community_fraud.get(comm_id, 0) + 1
            
    densities = []
    for comm_id in community_counts:
        # Filter tiny communities to reduce noise
        if community_counts[comm_id] > 2:
            fraud_count = community_fraud.get(comm_id, 0)
            densities.append(fraud_count / community_counts[comm_id])

    plt.figure(figsize=(10, 6))
    
    # Histogram with KDE
    sns.histplot(densities, bins=30, kde=True, color='#8da0cb', edgecolor='white', linewidth=1.2, alpha=0.8)
    
    plt.title('Distribution of Illicit Density per Community', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Illicit Transaction Density (0.0 = 100% Licit, 1.0 = 100% Illicit)', fontsize=12)
    plt.ylabel('Number of Communities', fontsize=12)
    plt.yscale('log') # Log scale matches the original example's Y-axis (10^0 - 10^2)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'density_plot.png'), dpi=300)
    plt.close()

def generate_pr_curve(df):
    print("Generating pr_curve.png (REAL CALCULATION)...")
    from sklearn.metrics import precision_recall_curve, auc
    
    # Data Prep (Same as before)
    G = nx.from_pandas_edgelist(df, 'nameOrig', 'nameDest', ['amount', 'isFraud'], create_using=nx.DiGraph())
    fraud_nodes = list(set(list(df[df['isFraud'] == 1]['nameOrig']) + list(df[df['isFraud'] == 1]['nameDest'])))
    valid_fraud_nodes = [n for n in fraud_nodes if n in G.nodes()]
    
    if not valid_fraud_nodes: return

    random.seed(42)
    random.shuffle(valid_fraud_nodes)
    split_idx = int(len(valid_fraud_nodes) * 0.5)
    seed_nodes = valid_fraud_nodes[:split_idx]
    target_fraud_nodes = set(valid_fraud_nodes[split_idx:])
    
    personalization = {n: 1.0 for n in seed_nodes}
    try:
        ppr_scores = nx.pagerank(G, alpha=0.85, personalization=personalization, max_iter=100)
    except: return

    y_true, y_scores = [], []
    for node in G.nodes():
        if node in seed_nodes: continue
        y_true.append(1 if node in target_fraud_nodes else 0)
        y_scores.append(ppr_scores.get(node, 0))
        
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    print(f"Calculated PR-AUC: {pr_auc:.4f}")

    # --- 2. PR Curve (Black Line, Light Grey Fill) ---
    plt.figure(figsize=(8, 6))
    plt.style.use('default') # Reset
    
    # Grid
    plt.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
    
    # Plot
    plt.plot(recall, precision, color='black', linewidth=2.5, label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.fill_between(recall, precision, color='#f0f0f0', alpha=0.5)
    
    plt.title('Precision-Recall Curve for PPR Risk Scoring', fontsize=16, fontweight='bold', pad=20, color='#333333')
    plt.xlabel('Recall', fontsize=14, fontweight='bold', color='#333333')
    plt.ylabel('Precision', fontsize=14, fontweight='bold', color='#333333')
    
    # Thick axes lines
    ax = plt.gca()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        
    plt.legend(loc='lower left', frameon=False, fontsize=12)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'pr_curve.png'), dpi=300)
    plt.close()

def generate_forensic_subgraph(df):
    print("Generating forensic_subgraph.png...")
    
    # 1. Build Graph
    G = nx.from_pandas_edgelist(df, 'nameOrig', 'nameDest', ['amount', 'isFraud'], create_using=nx.DiGraph())
    
    # 2. Find a "Star" fraudster (High out-degree)
    fraud_sources = df[df['isFraud'] == 1]['nameOrig'].unique()
    valid_sources = [n for n in fraud_sources if n in G.nodes()]
    
    if not valid_sources: return

    # Sort primarily by out-degree to get a good star
    degrees = dict(G.out_degree(valid_sources))
    top_source = sorted(degrees, key=degrees.get, reverse=True)[0]
    
    # 3. Extract Subgraph (Source + Neighbors + Neighbors of Neighbors)
    # We want ~15-25 nodes for elegance.
    nodes_to_keep = {top_source}
    neighbors = list(G.successors(top_source))
    nodes_to_keep.update(neighbors[:10]) # Top 10 mules
    
    # Add some secondary connections if available
    for n in list(nodes_to_keep):
        secondary = list(G.successors(n))
        if secondary:
            nodes_to_keep.update(secondary[:3]) # Add a few secondary
            
    subgraph = G.subgraph(nodes_to_keep)
    
    # 4. Calculate Risk Scores (Local PPR)
    personalization = {top_source: 1.0}
    ppr_scores = nx.pagerank(subgraph, alpha=0.85, personalization=personalization)
    
    # 5. Visualization
    pos = nx.spring_layout(subgraph, seed=42, k=0.5) # k regulates spacing
    
    plt.figure(figsize=(10, 8))
    
    nodes = list(subgraph.nodes())
    colors = [ppr_scores.get(n, 0) for n in nodes]
    
    # Color Map: Viridis (Purple -> Yellow)
    # Sizes: Source larger
    sizes = [1500 if n == top_source else 800 for n in nodes]
    
    nx.draw_networkx_nodes(subgraph, pos, node_color=colors, cmap=plt.cm.viridis, node_size=sizes, alpha=0.9)
    nx.draw_networkx_edges(subgraph, pos, edge_color='gray', width=1.5, arrowsize=15, alpha=0.6)
    
    # Labels (Just last 4 digits for elegance)
    labels = {n: n[-4:] for n in nodes}
    nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=10, font_weight='bold', font_color='white') # White text on dark nodes?
    # Use black text for visibility on yellow
    nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=10, font_weight='bold', font_color='black')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(colors), vmax=max(colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
    cbar.set_label('Personalized PageRank Risk (Licit \u2192 Illicit)', fontsize=12)
    
    plt.title('Forensic Subgraph Visualization (Colored by Risk Score)', fontsize=16, fontweight='bold', color='#333333')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'forensic_subgraph.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    df = load_data()
    generate_density_plot(df)
    generate_pr_curve(df)
    generate_forensic_subgraph(df)
    print("Done.")
