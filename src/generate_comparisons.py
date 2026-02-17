import pandas as pd
import networkx as nx
import community.community_louvain as community_louvain
import time
import os
import random
from sklearn.metrics import precision_recall_curve, auc

# Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(os.path.dirname(script_dir), 'datasets', 'paysim dataset.csv')

def load_data():
    print("Loading PaySim...")
    df = pd.read_csv(data_path)
    fraud = df[df['isFraud'] == 1]
    legit = df[df['isFraud'] == 0].sample(n=len(fraud), random_state=42)
    return pd.concat([fraud, legit])

def benchmark_community(G, df):
    print("\n--- Community Detection ---")
    
    # 1. Louvain (Ours)
    start = time.time()
    partition_louvain = community_louvain.best_partition(G.to_undirected())
    time_louvain = time.time() - start
    
    # Homogeneity
    fraud_nodes = set(df[df['isFraud'] == 1]['nameOrig']) | set(df[df['isFraud'] == 1]['nameDest'])
    comm_counts_l = {}
    comm_fraud_l = {}
    for n, c in partition_louvain.items():
        comm_counts_l[c] = comm_counts_l.get(c, 0) + 1
        if n in fraud_nodes: comm_fraud_l[c] = comm_fraud_l.get(c, 0) + 1
    
    # Weighted average homogeneity
    total_homogeneity_l = 0
    total_nodes = len(G.nodes())
    for c, count in comm_counts_l.items():
        p = comm_fraud_l.get(c, 0) / count
        homogeneity = max(p, 1-p) # Purity (either mostly fraud or mostly legit)
        total_homogeneity_l += homogeneity * count
    avg_homogeneity_l = total_homogeneity_l / total_nodes

    print(f"Louvain: {time_louvain:.4f}s, Homogeneity: {avg_homogeneity_l:.4f}")

    # 2. Label Propagation (Competitor)
    start = time.time()
    communities_lpa = list(nx.community.label_propagation_communities(G.to_undirected()))
    time_lpa = time.time() - start
    
    # Calculate Homogeneity for LPA
    # LPA returns list of sets
    partition_lpa = {}
    for idx, comm in enumerate(communities_lpa):
        for n in comm: partition_lpa[n] = idx
        
    comm_counts_p = {}
    comm_fraud_p = {}
    for n, c in partition_lpa.items():
        comm_counts_p[c] = comm_counts_p.get(c, 0) + 1
        if n in fraud_nodes: comm_fraud_p[c] = comm_fraud_p.get(c, 0) + 1
        
    total_homogeneity_p = 0
    generated_nodes = len(partition_lpa) # LPA might not cover all nodes? usually does.
    if generated_nodes == 0: avg_homogeneity_p = 0
    else:
        for c, count in comm_counts_p.items():
            p = comm_fraud_p.get(c, 0) / count
            homogeneity = max(p, 1-p)
            total_homogeneity_p += homogeneity * count
        avg_homogeneity_p = total_homogeneity_p / generated_nodes

    print(f"LPA:     {time_lpa:.4f}s, Homogeneity: {avg_homogeneity_p:.4f}")
    return (time_louvain, avg_homogeneity_l), (time_lpa, avg_homogeneity_p)

def benchmark_scc(G):
    print("\n--- SCC Detection ---")
    
    # 1. Tarjan (Ours, nx.strongly_connected_components usually uses Tarjan)
    # Actually nx uses Tarjan's by default or local equivalent.
    start = time.time()
    sccs_t = list(nx.strongly_connected_components(G))
    time_tarjan = time.time() - start
    print(f"Tarjan:   {time_tarjan:.4f}s, Count: {len(sccs_t)}")
    
    # 2. Kosaraju (Competitor)
    start = time.time()
    sccs_k = list(nx.kosaraju_strongly_connected_components(G))
    time_kosaraju = time.time() - start
    print(f"Kosaraju: {time_kosaraju:.4f}s, Count: {len(sccs_k)}")
    
    return time_tarjan, time_kosaraju

def benchmark_risk(G, df):
    print("\n--- Risk Scoring ---")
    
    # Setup Ground Truth
    fraud_nodes = list(df[df['isFraud'] == 1]['nameOrig']) + list(df[df['isFraud'] == 1]['nameDest'])
    fraud_nodes = list(set(fraud_nodes))
    valid_fraud = [n for n in fraud_nodes if n in G.nodes()]
    random.seed(42)
    random.shuffle(valid_fraud)
    split = int(len(valid_fraud)*0.5)
    seeds = valid_fraud[:split]
    targets = set(valid_fraud[split:])
    
    # 1. PPR (Ours)
    start = time.time()
    pers = {n: 1.0 for n in seeds}
    ppr = nx.pagerank(G, alpha=0.85, personalization=pers)
    time_ppr = time.time() - start
    
    # AUC
    y_true = [1 if n in targets else 0 for n in G.nodes() if n not in seeds]
    y_score_p = [ppr.get(n, 0) for n in G.nodes() if n not in seeds]
    precision, recall, _ = precision_recall_curve(y_true, y_score_p)
    auc_ppr = auc(recall, precision)
    print(f"PPR:        {time_ppr:.4f}s, PR-AUC: {auc_ppr:.4f}")
    
    # 2. Degree Centrality (Competitor - Heuristic)
    # Simple In-Degree is often used as a baseline for popularity/risk
    start = time.time()
    deg = nx.in_degree_centrality(G)
    time_deg = time.time() - start
    
    y_score_d = [deg.get(n, 0) for n in G.nodes() if n not in seeds]
    precision, recall, _ = precision_recall_curve(y_true, y_score_d)
    auc_deg = auc(recall, precision)
    print(f"In-Degree:  {time_deg:.4f}s, PR-AUC: {auc_deg:.4f}")
    
    return (time_ppr, auc_ppr), (time_deg, auc_deg)

def benchmark_flow(G):
    print("\n--- Max Flow ---")
    # Finding a valid source/sink pair with flow
    # Just picking first two nodes? No, need connected.
    # Try to find a path first.
    
    try:
        path = nx.shortest_path(G) # Dictionary of paths
        # Iterate to find a long path or specific fraud path
        src, sink = None, None
        for s in path:
            for t in path[s]:
                if len(path[s][t]) > 3: # length 3
                    src, sink = s, t
                    break
            if src: break
    except: src, sink = list(G.nodes())[0], list(G.nodes())[1]

    if not src:
        print("No path found for flow.")
        return 0, 0
        
    print(f"Flow from {src} to {sink}")
    # Capacity? Use 'amount'? 
    # Edges have 'amount' attribute.
    
    # 1. Edmonds-Karp (Ours)
    start = time.time()
    try:
        flow_ek = nx.maximum_flow_value(G, src, sink, capacity='amount', flow_func=nx.algorithms.flow.edmonds_karp)
    except: flow_ek = 0
    time_ek = time.time() - start
    print(f"Edmonds-Karp: {time_ek:.4f}s")
    
    # 2. Preflow-Push (Competitor)
    start = time.time()
    try:
        flow_pp = nx.maximum_flow_value(G, src, sink, capacity='amount', flow_func=nx.algorithms.flow.preflow_push)
    except: flow_pp = 0
    time_pp = time.time() - start
    print(f"Preflow-Push: {time_pp:.4f}s")
    
    return time_ek, time_pp

if __name__ == "__main__":
    df = load_data()
    G = nx.from_pandas_edgelist(df, 'nameOrig', 'nameDest', ['amount', 'isFraud'], create_using=nx.DiGraph())
    
    benchmark_community(G, df)
    benchmark_scc(G)
    benchmark_risk(G, df)
    benchmark_flow(G)
