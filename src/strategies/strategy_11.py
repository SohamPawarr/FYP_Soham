import time
import numpy as np
import networkx as nx
import copy
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import community.community_louvain as community_louvain
from src.graph_utils import evaluate

def run(G, df, node_labels, fraud_train, legit_nodes, test_nodes, test_labels, dataset_name, src_col, label_col):
    print("\n🔬 Strategy 11: Integrated Framework Bible")
    t0 = time.time()
    
    fraud_train_set = set(fraud_train)
    # We can infer legit_train strictly from train_nodes to prevent leaking test labels.
    rng = np.random.RandomState(42)
    legit_train = rng.choice(legit_nodes, size=min(len(fraud_train), len(legit_nodes)), replace=False).tolist()
    legit_train_set = set(legit_train)
    
    train_nodes = fraud_train + legit_train

    # Pre-compute heavy graph features
    G_und = G.to_undirected()
    try:
        clustering = nx.clustering(G_und)
    except Exception:
        clustering = {}
        
    try:
        hub_scores, auth_scores = nx.hits(G, max_iter=50, tol=1e-4)
    except Exception:
        hub_scores, auth_scores = {n: 0 for n in G.nodes()}, {n: 0 for n in G.nodes()}
        
    try:
        avg_neighbor_deg = nx.average_neighbor_degree(G)
    except Exception:
        avg_neighbor_deg = {}
    
    # 2A. Louvain Communities
    partition = community_louvain.best_partition(G_und)
    comm_sizes = defaultdict(int)
    for comm_id in partition.values():
        comm_sizes[comm_id] += 1
        
    # 2B. Tarjan's SCCs
    scc_sizes = {}
    scc_flags = {}
    for scc in nx.strongly_connected_components(G):
        sz = len(scc)
        for node in scc:
            scc_sizes[node] = sz
            scc_flags[node] = 1 if sz > 1 else 0
            
    # 2C. Strategy 8 Motifs
    motifs = {}
    try:
        triangles = nx.triangles(G_und)
    except Exception:
        triangles = {n: 0 for n in G.nodes()}
        
    for node in G.nodes():
        preds = list(G.predecessors(node))
        succs = list(G.successors(node))
        tri = triangles.get(node, 0)
        two_paths = len(preds) * len(succs)
        reciprocal = sum(1 for s in succs if G.has_edge(s, node))
        fan_ratio = len(succs) / (len(preds) + 1e-10)
        motifs[node] = {
            "triangles": tri, "two_paths": two_paths, 
            "reciprocity": reciprocal, "fan_ratio": fan_ratio
        }
    
    # PPR
    seeds = {n: 1.0 for n in fraud_train_set if n in G.nodes()}
    ppr_scores = nx.pagerank(G, personalization=seeds if seeds else None)

    # Label Propagation
    scores = {}
    for n in G.nodes():
        if n in fraud_train_set:
            scores[n] = 1.0 # Known Bad
        elif n in legit_train_set:
            scores[n] = 0.0 # Known Legit 
        else:
            scores[n] = 0.5 # Unknown / Test
            
    for _ in range(10):
        new_scores = {}
        for n in G.nodes():
            if n in fraud_train_set or n in legit_train_set:
                new_scores[n] = scores[n]
                continue
            neighbors = list(G.predecessors(n)) + list(G.successors(n))
            if neighbors:
                new_scores[n] = 0.3 * scores[n] + 0.7 * np.mean([scores.get(nb, 0.5) for nb in neighbors])
            else:
                new_scores[n] = scores[n]
        scores = new_scores

    # Time-Weighted PageRanks
    ppr_tw = nx.pagerank(G, personalization=seeds if seeds else None, weight="weight")
    
    legit_train_list = list(legit_train_set)
    legit_seeds = {n: 1.0 for n in legit_train_list[:min(500, len(legit_train_list))] if n in G.nodes()}
    ppr_rev = nx.pagerank(G, personalization=legit_seeds if legit_seeds else None, weight="weight")

    def assemble_features(nodes):
        X, y = [], []
        for node in nodes:
            if node not in G: continue
            
            in_deg = G.in_degree(node)
            out_deg = G.out_degree(node)
            total_deg = in_deg + out_deg
            
            in_flow = sum(G[u][node].get("weight", 0) for u in G.predecessors(node))
            out_flow = sum(G[node][v].get("weight", 0) for v in G.successors(node))
            flow_imbalance = (out_flow - in_flow) / (in_flow + out_flow + 1e-10)
            
            in_count = sum(G[u][node].get("count", 1) for u in G.predecessors(node))
            out_count = sum(G[node][v].get("count", 1) for v in G.successors(node))
            tx_count = in_count + out_count

            tri = motifs[node]["triangles"]
            two_paths = motifs[node]["two_paths"]
            reciprocal = motifs[node]["reciprocity"]
            
            s = scores.get(node, 0.5)
            neighbors = list(G.predecessors(node)) + list(G.successors(node))
            nb_scores = [scores.get(nb, 0.5) for nb in neighbors] if neighbors else [0.5]
            
            pt = ppr_tw.get(node, 0)
            pr = ppr_rev.get(node, 0)
            
            vector = [
                total_deg, in_deg, out_deg, in_flow, out_flow,              
                flow_imbalance, tx_count,                                   
                ppr_scores.get(node, 0.0),                                  
                comm_sizes.get(partition.get(node, -1), 0),                 
                clustering.get(node, 0),                                    
                hub_scores.get(node, 0), auth_scores.get(node, 0),          
                avg_neighbor_deg.get(node, 0),                              
                scc_flags.get(node, 0),                                     
                out_flow / (in_flow + 1e-10),                               
                tri, scc_sizes.get(node, 1), two_paths, reciprocal,
                s, np.mean(nb_scores), np.max(nb_scores),
                pt, pr, (pt + 1e-10) / (pr + 1e-10)
            ]
            X.append(vector)
            y.append(node_labels.get(node, 0) if node in node_labels else (test_labels.get(node, 0) if node in test_labels else 0))
        return np.array(X, dtype=float), np.array(y)

    X_train, y_train = assemble_features(train_nodes)
    X_test, y_test   = assemble_features(test_nodes)
    
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test  = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    n_feats = X_train.shape[1]

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
    
    print(f"   ✅ ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  F1={metrics['f1']:.4f}  ({runtime:.1f}s)")
    
    return {
        "strategy": "S11_Bible_Framework",
        "dataset": dataset_name,
        "roc_auc": metrics.get("roc_auc", 0),
        "pr_auc": metrics.get("pr_auc", 0),
        "f1": metrics.get("f1", 0),
        "precision": metrics.get("precision", 0),
        "recall": metrics.get("recall", 0),
        "n_features": n_feats,
        "runtime_sec": runtime
    }
