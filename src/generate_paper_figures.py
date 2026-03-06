import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOG_FILE = os.path.join(RESULTS_DIR, "experiment_log.csv")

def generate_winner_comparison():
    print("\n📊 Generating High-Resolution Academic Figures...")
    
    if not os.path.exists(LOG_FILE):
        print("   ❌ experiment_log.csv not found! Run main.py first.")
        return

    df = pd.read_csv(LOG_FILE)
    if df.empty:
        print("   ❌ No data in experiment_log.csv")
        return

    # Standardize strategy labeling for publication
    # Extract index (S00) and cleaned name
    df['Strategy_ID'] = df['strategy'].apply(lambda x: x.split('_')[0])
    df['Strategy_Name'] = df['strategy'].apply(lambda x: x.split('_', 1)[1].replace('_', ' '))
    df['Label'] = df['Strategy_ID'] + ": " + df['Strategy_Name']

    # Set Academic Theme
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.titlesize": 18
    })
    sns.set_theme(style="ticks", context="paper")

    # --- FIGURE 1: COMPREHENSIVE PERFORMANCE BAR CHART ---
    # Metrics to compare
    metrics = ['roc_auc', 'pr_auc', 'f1']
    df_melted = df.melt(id_vars=['Strategy_ID', 'Strategy_Name'], value_vars=metrics, 
                        var_name='Metric', value_name='Score')
    
    # Map metric names to academic labels
    df_melted['Metric'] = df_melted['Metric'].replace({
        'roc_auc': 'ROC-AUC', 'pr_auc': 'PR-AUC', 'f1': 'F1-Score'
    })

    plt.figure(figsize=(14, 8))
    g = sns.barplot(
        data=df_melted,
        x='Strategy_ID',
        y='Score',
        hue='Metric',
        palette='viridis',
        edgecolor='0.2',
        linewidth=0.8
    )
    
    plt.title('Performance Metric Comparison Across Detection Strategies', fontweight='bold', pad=20)
    plt.xlabel('Strategy ID', fontweight='bold')
    plt.ylabel('Performance Score', fontweight='bold')
    plt.ylim(0, 1.05)
    plt.legend(title='Metric', bbox_to_anchor=(1, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'performance_comparison_matrix.png'), dpi=400)
    plt.close()

    # --- FIGURE 2: EFFICIENCY FRONTIER (AUC VS RUNTIME) ---
    plt.figure(figsize=(12, 8))
    
    # Use log scale for runtime to handle differences between simple vs complex models
    scatter = sns.scatterplot(
        data=df,
        x='runtime_sec',
        y='roc_auc',
        size='n_features',
        hue='Strategy_Name',
        sizes=(100, 500),
        alpha=0.8,
        palette='Set1',
        edgecolor='black',
        linewidth=1
    )
    
    # Annotate points with IDs
    for i in range(df.shape[0]):
        plt.text(
            df.runtime_sec.iloc[i], 
            df.roc_auc.iloc[i] + 0.01, 
            df.Strategy_ID.iloc[i],
            horizontalalignment='center', 
            size='small', weight='bold'
        )

    plt.xscale('log')
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    
    plt.title('Efficiency Frontier: Predictive Power vs Computational Latency', fontweight='bold', pad=20)
    plt.xlabel('Execution Runtime (seconds, log-scale)', fontweight='bold')
    plt.ylabel('ROC-AUC Score', fontweight='bold')
    plt.grid(True, which="both", linestyle='--', alpha=0.3)
    plt.legend(title='Strategy Module', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
    # Highlight the winner
    winner = df.loc[df['roc_auc'].idxmax()]
    plt.annotate('SOTA Champion', 
                 xy=(winner.runtime_sec, winner.roc_auc),
                 xytext=(winner.runtime_sec * 1.5, winner.roc_auc - 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                 fontweight='bold', color='darkred')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'efficiency_frontier_auc_runtime.png'), dpi=400)
    plt.close()

    # --- FIGURE 3: STRATEGY RANKING HEATMAP ---
    pivot_df = df.set_index('Strategy_ID')[metrics].sort_values('roc_auc', ascending=False)
    pivot_df.columns = ['ROC-AUC', 'PR-AUC', 'F1-Score']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.3f', linewidths=.5, cbar_kws={'label': 'Metric Value'})
    plt.title('Benchmark Summary: Strategy Performance Rankings', fontweight='bold', pad=20)
    plt.ylabel('Strategy Identifier', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'strategy_performance_heatmap.png'), dpi=400)
    plt.close()

    print(f"   ✅ Processed 3 Academic Figures in {RESULTS_DIR}")

if __name__ == "__main__":
    generate_winner_comparison()
