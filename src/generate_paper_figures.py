import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOG_FILE = os.path.join(RESULTS_DIR, "experiment_log.csv")

def generate_winner_comparison():
    print("\n📊 Generating Strategy Winner Comparison Graphs...")
    
    if not os.path.exists(LOG_FILE):
        print("   ❌ experiment_log.csv not found! Run main.py first.")
        return

    df = pd.read_csv(LOG_FILE)
    if df.empty:
        print("   ❌ No data in experiment_log.csv")
        return

    # Clean up strategy names for the plot
    df['Strategy_Clean'] = df['strategy'].apply(lambda x: x.split('_', 1)[1].replace('_', ' '))

    # Set Seaborn theme
    sns.set_theme(style="whitegrid", context="talk")

    # 1. ROC-AUC bar chart
    plt.figure(figsize=(14, 8))
    
    # Sort by ROC-AUC to highlight winner
    df_sorted_roc = df.sort_values('roc_auc', ascending=False)
    winner_idx = df_sorted_roc.index[0]
    
    # Choose a color palette: Winner gets distinct color
    colors = ['#ff7f0e' if i == winner_idx else '#1f77b4' for i in range(len(df_sorted_roc))]
    
    ax = sns.barplot(
        data=df_sorted_roc, 
        x='roc_auc', 
        y='Strategy_Clean', 
        palette=colors,
        edgecolor='black'
    )

    # Annotate with the exact scores
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        ax.annotate(f'{width:.3f}', 
                    (width, p.get_y() + p.get_height() / 2.), 
                    ha='right', va='center', 
                    xytext=(-5, 0), textcoords='offset points',
                    color='white' if i == 0 else 'black',
                    fontweight='bold')

    plt.title('Strategy Performance Breakdown (ROC-AUC)\nOrange: Winning Architecture', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('ROC-AUC Score (Higher is Better)', fontsize=14)
    plt.ylabel('Modeling Strategy', fontsize=14)
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'winner_roc_auc_comparison.png'), dpi=300)
    plt.close()

    # 2. Scatter Plot: Performance vs Runtime
    plt.figure(figsize=(12, 8))
    
    # Using F1 Score on Y and Runtime on X
    scatter = sns.scatterplot(
        data=df, 
        x='runtime_sec', 
        y='f1', 
        hue='Strategy_Clean',
        s=200, 
        palette='husl',
        edgecolor='black',
        linewidth=1.5
    )
    
    # Add labels to points
    for line in range(0, df.shape[0]):
        scatter.text(
            df.runtime_sec.iloc[line] + (df.runtime_sec.max() * 0.01), 
            df.f1.iloc[line], 
            df.strategy.iloc[line].split('_')[0], # Just S0, S1, etc
            horizontalalignment='left', 
            size='medium', color='black', weight='semibold'
        )

    plt.title('Computation Cost vs. Detection Accuracy (F1-Score)', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Execution Time (Seconds)', fontsize=14)
    plt.ylabel('F1-Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Move legend outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'runtime_vs_f1_scatter.png'), dpi=300)
    plt.close()
    
    print(f"   ✅ Saved winner_roc_auc_comparison.png and runtime_vs_f1_scatter.png to {RESULTS_DIR}")

if __name__ == "__main__":
    generate_winner_comparison()
