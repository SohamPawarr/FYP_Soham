# Research Validation Pipeline: Multi-Dataset Benchmark
# Supports: PaySim, Metaverse, Elliptic
# Metrics: Runtime, ROC-AUC, F1-Score, Homogeneity

import pandas as pd
import networkx as nx
import numpy as np
import time
import os
import kagglehub
import community.community_louvain as community_louvain
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, homogeneity_score, average_precision_score

# --- CONFIGURATION ---
DATASETS_TO_RUN = ["PAYSIM", "METAVERSE", "ELLIPTIC"]

# Robust path handling relative to this script file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

print("=== STARTING COMPREHENSIVE BENCHMARK SUITE (WITH QUALITY METRICS) ===")
print(f"Data Directory: {DATA_DIR}")
print(f"Results Directory: {RESULTS_DIR}")

all_results = []
dataset_stats = []

def get_dataset(name):
    print(f"\n--- Loading Dataset: {name} ---")
    if name == "PAYSIM":
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, 'paysim dataset.csv'))
            # BALANCED SAMPLING: All fraud + equal legit (as per paper Section 4.1)
            fraud_df = df[df['isFraud'] == 1]
            legit_df = df[df['isFraud'] == 0].sample(n=len(fraud_df), random_state=42)
            df = pd.concat([fraud_df, legit_df]).sample(frac=1, random_state=42)  # Shuffle
            print(f"Balanced Sample: {len(fraud_df)} fraud + {len(legit_df)} legit = {len(df)} total")
            return df, 'nameOrig', 'nameDest', 'amount', 'isFraud'
        except FileNotFoundError:
            print(f"PaySim not found at {os.path.join(DATA_DIR, 'paysim dataset.csv')}")
            return None, None, None, None, None
            
    elif name == "METAVERSE":
        try:
            # Corrected filename based on directory check
            df = pd.read_csv(os.path.join(DATA_DIR, 'metaverse_transactions_dataset.csv')) 
            # BALANCED SAMPLING: All anomaly + equal normal
            fraud_df = df[df['anomaly'] == 1]
            legit_df = df[df['anomaly'] == 0].sample(n=len(fraud_df), random_state=42)
            df = pd.concat([fraud_df, legit_df]).sample(frac=1, random_state=42)  # Shuffle
            print(f"Balanced Sample: {len(fraud_df)} anomaly + {len(legit_df)} normal = {len(df)} total")
            return df, 'sending_address', 'receiving_address', 'amount', 'anomaly'
        except FileNotFoundError:
            print(f"Metaverse data not found at {os.path.join(DATA_DIR, 'metaverse_transactions_dataset.csv')}")
            return None, None, None, None, None

    elif name == "ELLIPTIC":
        print("Checking/Downloading Elliptic Dataset via KaggleHub...")
        try:
            path = kagglehub.dataset_download("ellipticco/elliptic-data-set")
            elliptic_path = path 
        except Exception as e:
            print(f"Failed to download Elliptic: {e}")
            return None, None, None, None, None
        
        try:
            # Recursive search for files
            edge_file = None
            class_file = None
            for root, dirs, files in os.walk(elliptic_path):
                if 'elliptic_txs_edgelist.csv' in files:
                    edge_file = os.path.join(root, 'elliptic_txs_edgelist.csv')
                if 'elliptic_txs_classes.csv' in files:
                    class_file = os.path.join(root, 'elliptic_txs_classes.csv')
            
            if not edge_file or not class_file:
                 print("Elliptic CSVs not found.")
                 return None, None, None, None, None
            
            print(f"Reading {edge_file}...")
            edgelist = pd.read_csv(edge_file)
            classes = pd.read_csv(class_file)
            
            # Map classes: 1=Illicit, 2=Licit, "unknown"=Unknown
            # For benchmark, we map 1 -> 1 (Fraud), 2 -> 0 (Legit), Unknown -> 0
            classes['isFraud'] = classes['class'].apply(lambda x: 1 if x == '1' else 0)
            
            # Merge class info onto the SOURCE node of the edge (txId1)
            # This allows us to treat the edge as "High Risk" if the source is Illicit
            df = edgelist.merge(classes, left_on='txId1', right_on='txId', how='inner')
            
            # BALANCED SAMPLING: All illicit + equal licit
            fraud_df = df[df['isFraud'] == 1]
            legit_df = df[df['isFraud'] == 0].sample(n=len(fraud_df), random_state=42)
            df = pd.concat([fraud_df, legit_df]).sample(frac=1, random_state=42)  # Shuffle
            print(f"Balanced Sample: {len(fraud_df)} illicit + {len(legit_df)} licit = {len(df)} total")
            df['amount'] = np.random.uniform(1, 10000, size=len(df)) # Dummy amount
            return df, 'txId1', 'txId2', 'amount', 'isFraud'
        except Exception as e:
            print(f"Error loading Elliptic: {e}")
            return None, None, None, None, None

    return None, None, None, None, None


def log_result(dataset, task, algo_name, time_taken, result_val, quality_metric, quality_val, description):
    all_results.append({
        "Dataset": dataset,
        "Task": task,
        "Algorithm": algo_name,
        "Time (s)": round(time_taken, 4),
        "Result Value": result_val,
        "Quality Metric": quality_metric,
        "Quality Score": round(quality_val, 4) if isinstance(quality_val, (int, float)) else quality_val,
        "Description": description
    })

# --- ALGORITHM IMPLEMENTATIONS ---

def run_community_detection(G, dataset_name, node_labels):
    G_undirected = G.to_undirected()
    
    # 1. Louvain
    start = time.time()
    try:
        partition = community_louvain.best_partition(G_undirected)
        dur = time.time() - start
        
        # Quality: Homogeneity (Do communities align with fraud labels?)
        # Only feasible if we have ground truth for nodes
        if node_labels:
            # Create list of true labels and pred labels (community id) aligned by node
            y_true = []
            y_pred = []
            for node, comm_id in partition.items():
                if node in node_labels:
                    y_true.append(node_labels[node])
                    y_pred.append(comm_id)
            
            score = homogeneity_score(y_true, y_pred)
            metric_name = "Homogeneity Score"
        else:
            score = 0.0
            metric_name = "N/A (No Labels)"

        log_result(dataset_name, "Community", "Louvain", dur, len(set(partition.values())), metric_name, score, "High Modularity Clustering")
    except Exception as e:
        log_result(dataset_name, "Community", "Louvain", 0, "Error", "Error", 0, str(e))

    # 2. WCC
    start = time.time()
    comps = list(nx.weakly_connected_components(G))
    dur = time.time() - start
    log_result(dataset_name, "Community", "WCC", dur, len(comps), "Connectivity", "N/A", "Basic Weakly Connected Components")

    # 3. Label Propagation
    start = time.time()
    lp_communities = list(nx.community.label_propagation_communities(G_undirected))
    dur = time.time() - start
    
    # Calculate Homogeneity for LP
    if node_labels:
        y_true = []
        y_pred = []
        for i, comm in enumerate(lp_communities):
            for node in comm:
                if node in node_labels:
                    y_true.append(node_labels[node])
                    y_pred.append(i)
        score = homogeneity_score(y_true, y_pred) if y_true else 0.0
        metric_name = "Homogeneity Score"
    else:
        score = 0.0
        metric_name = "N/A"

    log_result(dataset_name, "Community", "Label Propagation", dur, len(lp_communities), metric_name, score, "Structure-based Propagation")

def run_risk_scoring(G, dataset_name, node_labels):
    # Prepare Labels for ROC-AUC
    # We need a binary vector for all nodes in G
    if not node_labels:
        print("No labels for risk scoring verification.")
        y_true = None
    else:
        # Filter nodes that have labels (some datasets might have unlabeled nodes)
        nodes_with_labels = [n for n in G.nodes() if n in node_labels]
        y_true = [node_labels[n] for n in nodes_with_labels]
        
    def eval_risk(scores_dict, name, desc):
        start = time.time() # This timing is just for eval, logic timing is passed in
        y_scores = [scores_dict.get(n, 0) for n in nodes_with_labels] if y_true else []
        
        if y_true and len(set(y_true)) > 1: # Need both classes
            try:
                auc = roc_auc_score(y_true, y_scores)
                pr_auc = average_precision_score(y_true, y_scores)
            except:
                auc = 0.0
                pr_auc = 0.0
        else:
            auc = 0.0
            pr_auc = 0.0
        return auc, pr_auc

    # 1. PPR
    # Personalization seeds: ALL Known fraud nodes (as per paper Section 4.4)
    if node_labels:
        fraud_nodes = [n for n, l in node_labels.items() if l == 1 and n in G.nodes()]
        seeds = {n: 1.0 for n in fraud_nodes}  # Use ALL known fraud as seeds
        print(f"PPR Seeded with {len(fraud_nodes)} known fraud nodes")
        if not seeds: 
            seeds = None  # No personalization if no fraud nodes found
    else:
        seeds = None

    start = time.time()
    ppr_scores = nx.pagerank(G, personalization=seeds if seeds else None)
    dur = time.time() - start
    auc, pr = eval_risk(ppr_scores, "PPR", "Context-Aware")
    log_result(dataset_name, "Risk Scoring", "Personalized PageRank (ROC-AUC)", dur, "N/A", "ROC-AUC", auc, "Context-aware propagation from known fraud")
    log_result(dataset_name, "Risk Scoring", "Personalized PageRank (PR-AUC)", dur, "N/A", "PR-AUC", pr, "Precision-Recall Area Under Curve")

    # 2. Degree
    start = time.time()
    deg_scores = nx.degree_centrality(G)
    dur = time.time() - start
    auc, pr = eval_risk(deg_scores, "Degree", "Local")
    log_result(dataset_name, "Risk Scoring", "Degree Centrality (ROC-AUC)", dur, "N/A", "ROC-AUC", auc, "Local connectivity count")
    log_result(dataset_name, "Risk Scoring", "Degree Centrality (PR-AUC)", dur, "N/A", "PR-AUC", pr, "Precision-Recall Area Under Curve")

    # 3. Component-wise PageRank (Tarjan + PR)
    start = time.time()
    sccs = list(nx.strongly_connected_components(G))
    if not sccs:
        log_result(dataset_name, "Risk Scoring", "Component-wise PageRank (ROC-AUC)", 0, "No SCCs", "ROC-AUC", 0.0, "No SCCs found")
        log_result(dataset_name, "Risk Scoring", "Component-wise PageRank (PR-AUC)", 0, "No SCCs", "PR-AUC", 0.0, "No SCCs found")
        return
    largest_scc = max(sccs, key=len)
    subgraph = G.subgraph(largest_scc)
    try:
        sub_scores = nx.pagerank(subgraph)
    except:
        sub_scores = {}
    
    # Map back to full graph
    cw_scores = {n: sub_scores.get(n, 0) for n in G.nodes()}
    dur = time.time() - start
    auc, pr = eval_risk(cw_scores, "CW-PR", "Structure")
    log_result(dataset_name, "Risk Scoring", "Component-wise PageRank (ROC-AUC)", dur, f"Largest Comp: {len(largest_scc)}", "ROC-AUC", auc, "PageRank on Largest Strongly Connected Component")
    log_result(dataset_name, "Risk Scoring", "Component-wise PageRank (PR-AUC)", dur, f"Largest Comp: {len(largest_scc)}", "PR-AUC", pr, "Precision-Recall Area Under Curve")

def run_flow_analysis(G, dataset_name, amount_col):
    if G.number_of_edges() == 0: return
    u, v = list(G.edges())[0] 
    
    algos = [
        ("Edmonds-Karp", nx.algorithms.flow.edmonds_karp, "BFS-based, good for general graphs"),
        ("Shortest Augmenting Path", nx.algorithms.flow.shortest_augmenting_path, "Good for small capacity networks"),
        ("Preflow-Push", nx.algorithms.flow.preflow_push, "Local optimized push-relabel"),
        ("Dinitz", nx.algorithms.flow.dinitz, "Level graph optimization")
    ]
    
    for name, func, desc in algos:
        start = time.time()
        try:
            val = nx.maximum_flow_value(G, u, v, capacity=amount_col, flow_func=func)
            dur = time.time() - start
            log_result(dataset_name, "Flow", name, dur, val, "Max Flow Value", val, desc)
        except Exception as e:
            log_result(dataset_name, "Flow", name, 0, "Error", "Error", 0, str(e))

def run_structural(G, dataset_name):
    # 1. Tarjan's
    start = time.time()
    sccs = list(nx.strongly_connected_components(G))
    dur = time.time() - start
    cyclic_sccs = [c for c in sccs if len(c) > 1]
    
    # Quality: Did we find cycles? (Binary yes/no implies structure type)
    quality = "Cyclic" if cyclic_sccs else "Acyclic"
    
    log_result(dataset_name, "Structure", "Tarjan's SCC", dur, len(cyclic_sccs), "Graph Topology", quality, "Exact Strongly Connected Components")


# --- MAIN LOOP ---

for dataset_name in DATASETS_TO_RUN:
    df, src, dst, amt, label_col = get_dataset(dataset_name)
    if df is not None:
        print(f"Building Graph for {dataset_name}...")
        G = nx.from_pandas_edgelist(df, source=src, target=dst, edge_attr=amt, create_using=nx.DiGraph())
        
        # Build node labels dict for metrics
        if label_col and label_col in df.columns:
            # We only have labels for edges (transactions) usually, or source nodes?
            # PaySim: nameOrig isFraud? No, isFraud is for transaction.
            # But we can attribute fraud to nameOrig for node classification.
            # Group by source
            # Ensure label is numeric (0/1)
            try:
                # Force numeric conversion
                df[label_col] = pd.to_numeric(df[label_col], errors='coerce').fillna(0).astype(int)
                
                # Manual aggregation fallback - NO PANDAS GROUPBY to avoid crashes
                fraud_map = {}
                sources = df[src].values
                labels = df[label_col].values
                
                for s, l in zip(sources, labels):
                    if l > 0:
                        fraud_map[s] = 1
                    elif s not in fraud_map:
                        fraud_map[s] = 0
                
                node_labels = {n: fraud_map.get(n, 0) for n in G.nodes()} 
                fraud_count = sum(node_labels.values())
                print(f"Labels loaded: {fraud_count} Fraudulent Nodes identified.")
            except Exception as e:
                print(f"Warning: Failed to load node labels: {e}")
                node_labels = {}
        else:
            node_labels = {}
            print("No labels found for metrics.")

        print(f"Running Benchmarks for {dataset_name} ({G.number_of_nodes()} Nodes)...")
        dataset_stats.append({"Dataset": dataset_name, "Nodes": G.number_of_nodes(), "Edges": G.number_of_edges(), "Fraud Nodes": sum(node_labels.values())})
        
        print("DEBUG: Starting Community Detection...")
        run_community_detection(G, dataset_name, node_labels)
        print("DEBUG: Starting Risk Scoring...")
        run_risk_scoring(G, dataset_name, node_labels)
        print("DEBUG: Starting Structural Analysis...")
        run_structural(G, dataset_name)
        print("DEBUG: Starting Flow Analysis...")
        run_flow_analysis(G, dataset_name, amt)
        print(f"Finished {dataset_name}")

# --- EXPORT ---
if not all_results:
    print("No results to export. Check dataset paths.")
    exit(1)

print("Exporting Comprehensive Results with Metrics...")
results_df = pd.DataFrame(all_results)
stats_df = pd.DataFrame(dataset_stats)

try:
    with pd.ExcelWriter(os.path.join(RESULTS_DIR, 'comprehensive_benchmark_results.xlsx'), engine='openpyxl') as writer:
        # 1. Summary Sheet
        stats_df.to_excel(writer, sheet_name='Dataset Overview', index=False)
        
        # 2. Full Results
        results_df.to_excel(writer, sheet_name='Detailed Results', index=False)
        
        # 3. Pivot: Runtime
        try:
            pivot_time = results_df.pivot_table(index="Algorithm", columns="Dataset", values="Time (s)")
            pivot_time.to_excel(writer, sheet_name='Runtime Comparison')
        except: pass
        
        # 4. Pivot: Quality
        try:
            quality_df = results_df[results_df["Task"].isin(["Risk Scoring", "Community"])]
            pivot_quality = quality_df.pivot_table(index="Algorithm", columns="Dataset", values="Quality Score", aggfunc='first')
            pivot_quality.to_excel(writer, sheet_name='Quality Comparison')
        except: pass
except Exception as e:
    print(f"Excel Export Failed: {e}")
    # Fallback CSV
    results_df.to_csv(os.path.join(RESULTS_DIR, 'comprehensive_benchmark_results.csv'), index=False)

print("Generating Improved Plots...")
try:
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Runtime
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x="Algorithm", y="Time (s)", hue="Dataset")
    plt.xticks(rotation=45, ha='right')
    plt.title("Runtime Efficiency Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "benchmark_runtime.png"))
    
    # Plot 2: Quality (AUC)
    plt.figure(figsize=(12, 6))
    quality_data = results_df[results_df["Quality Metric"] == "ROC-AUC"]
    if not quality_data.empty:
        sns.barplot(data=quality_data, x="Algorithm", y="Quality Score", hue="Dataset")
        plt.title("Fraud Detection Accuracy (ROC-AUC)")
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "benchmark_quality_auc.png"))
except Exception as e:
    print(f"Plotting Failed: {e}")

print("Done! Check ../results/")
