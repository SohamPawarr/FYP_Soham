import os
import sys
import csv
import importlib
from datetime import datetime
import warnings
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')

# Add the src package to path to make absolute imports work
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_elliptic
from src.graph_utils import build_graph, get_node_labels, split_nodes

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOG_FILE = os.path.join(RESULTS_DIR, "experiment_log.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

def initialize_log():
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "strategy", "dataset", "roc_auc", "pr_auc", 
            "f1", "precision", "recall", "n_features", "runtime_sec"
        ])

def log_result(row):
    row["timestamp"] = datetime.now().isoformat()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "strategy", "dataset", "roc_auc", "pr_auc", 
            "f1", "precision", "recall", "n_features", "runtime_sec"
        ])
        writer.writerow(row)

def run_strategy(args):
    """Worker function for multiprocessing."""
    module_name, G, df, node_labels, fraud_train, legit_nodes, test_nodes, test_labels, src_col, label_col = args
    print(f"🔄 Starting [{module_name}] in worker process...")
    try:
        mod = importlib.import_module(module_name)
        result = mod.run(
            G=G, df=df, node_labels=node_labels, fraud_train=fraud_train, 
            legit_nodes=legit_nodes, test_nodes=test_nodes, test_labels=test_labels, 
            dataset_name="ELLIPTIC", src_col=src_col, label_col=label_col
        )
        return result
    except Exception as e:
        print(f"❌ {module_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("=" * 70)
    print("  FRAUD DETECTION EXPERIMENT RUNNER — BIBLE PIPELINE (PARALLEL)")
    print("=" * 70)

    initialize_log()

    # Get data
    df, src_col, dst_col, amt_col, label_col = load_elliptic()
    if df.empty:
        print("❌ Could not load dataset. Exiting.")
        return

    print("\n   Building graph...")
    G = build_graph(df, src_col, dst_col, amt_col)
    
    # Prune isolated nodes for memory & speed, though connected components logic handles it
    node_labels = get_node_labels(df, src_col, label_col)
    fraud_count = sum(1 for l in node_labels.values() if l == 1)
    legit_count = sum(1 for l in node_labels.values() if l == 0)
    print(f"   Labels: {fraud_count} fraud, {legit_count} legit")

    fraud_train, legit_nodes, test_nodes, test_labels = split_nodes(node_labels)
    print(f"   Split: {len(fraud_train)} train fraud, {len(test_nodes)} test nodes")

    strategy_modules = [
        "src.strategies.s00_topological_baseline",
        "src.strategies.s01_structural_feature_augmentation",
        "src.strategies.s02_multi_resolution_community_detection",
        "src.strategies.s03_associative_propagated_risk_2hop",
        "src.strategies.s04_heterogeneous_stacking_ensemble",
        "src.strategies.s05_spatio_temporal_feature_fusion",
        "src.strategies.s06_latent_graph_embeddings_node2vec",
        "src.strategies.s07_chronological_risk_propagation",
        "src.strategies.s08_higher_order_flow_motifs",
        "src.strategies.s09_semi_supervised_identity_propagation",
        "src.strategies.s10_composite_multi_feature_ensemble",
        "src.strategies.s11_unified_multistage_detection_framework",
    ]

    # Package arguments for multiprocessing
    tasks = [(mod, G, df, node_labels, fraud_train, legit_nodes, test_nodes, test_labels, src_col, label_col) for mod in strategy_modules]

    num_cores = max(1, min(cpu_count() - 1, len(strategy_modules)))
    print(f"\n🚀 Launching {len(tasks)} strategies across {num_cores} parallel cores...\n")

    # Run in parallel
    results = []
    with Pool(processes=num_cores) as pool:
        for res in pool.imap_unordered(run_strategy, tasks):
            if res:
                log_result(res)
                results.append(res)
                print(f"   ✅ Logged result for {res['strategy']}")

    print("\n" + "=" * 90)
    print("  EXPERIMENTS COMPLETE. GENERATING WINNER COMPARISON GRAPHS...")
    print("=" * 90)

    try:
        from src import generate_paper_figures
        # The main code will run plotting logic
        generate_paper_figures.generate_winner_comparison()
    except Exception as e:
        print(f"⚠️ Could not generate graphs automatically: {e}")

if __name__ == "__main__":
    main()
