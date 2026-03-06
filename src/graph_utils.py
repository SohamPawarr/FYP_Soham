import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, precision_recall_curve

def build_graph(df, src_col, dst_col, amt_col):
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
    labels = {}
    for _, row in df.iterrows():
        src = row[src_col]
        if row[label_col] > 0:
            labels[src] = 1
        elif src not in labels:
            labels[src] = 0
    return labels

def split_nodes(node_labels, test_size=0.5, random_state=42):
    fraud_nodes = [n for n, l in node_labels.items() if l == 1]
    legit_nodes = [n for n, l in node_labels.items() if l == 0]
    fraud_train, fraud_test = train_test_split(
        fraud_nodes, test_size=test_size, random_state=random_state
    )
    test_nodes = fraud_test + legit_nodes
    test_labels = {n: node_labels[n] for n in test_nodes}
    return fraud_train, legit_nodes, test_nodes, test_labels

def evaluate(y_true, y_proba):
    metrics = {}
    if len(set(y_true)) < 2:
        return {"roc_auc": 0, "pr_auc": 0, "f1": 0, "precision": 0, "recall": 0}

    metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    metrics["pr_auc"] = average_precision_score(y_true, y_proba)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    y_pred = (np.array(y_proba) >= best_thresh).astype(int)
    metrics["f1"] = f1_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    return metrics

def extract_enhanced_features(G, nodes, ppr_scores, partition, community_sizes):
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

        flow_total = in_flow + out_flow + 1e-10
        flow_imbalance = (out_flow - in_flow) / flow_total

        in_count = sum(G[u][node].get("count", 1) for u in G.predecessors(node))
        out_count = sum(G[node][v].get("count", 1) for v in G.successors(node))
        tx_count = in_count + out_count

        ppr = ppr_scores.get(node, 0)
        comm_id = partition.get(node, -1)
        comm_size = community_sizes.get(comm_id, 0)
        cc = clustering.get(node, 0)
        hub = hub_scores.get(node, 0)
        auth = auth_scores.get(node, 0)
        avg_nd = avg_neighbor_deg.get(node, 0)
        in_scc = 1 if node in sccs_set else 0

        features.append([
            total_deg, in_deg, out_deg, in_flow, out_flow,
            flow_imbalance, tx_count, ppr, comm_size, cc,
            hub, auth, avg_nd, in_scc,
            out_flow / (in_flow + 1e-10),
        ])
    return np.array(features, dtype=float)

def compute_common_features(G, fraud_train, legit_nodes, test_nodes, node_labels, test_labels):
    import community.community_louvain as community_louvain
    from collections import defaultdict

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
