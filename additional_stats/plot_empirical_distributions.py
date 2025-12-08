#!/usr/bin/env python3
"""
Plot empirical distributions of returns for each environment.

For each environment, creates a histogram showing the distribution of returns
across all algorithms and all seeds. Each algorithm is shown as a separate
histogram on the same plot for comparison.

Outputs:
    - empirical_distributions/: Directory containing one plot per environment
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for scripts
import matplotlib.pyplot as plt

# Set base directory
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

# Configuration
BASE_DIR = os.path.join(BASE, "rl_experiments")
RESULTS_CSV = os.path.join(BASE_DIR, "final_eval_returns.csv")

# Output directory
OUTPUT_DIR = os.path.join(BASE, "additional_stats", "empirical_distributions")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TASKS = [
    "Hopper-v5",
    "Walker2d-v5",
    "HalfCheetah-v5",
    "Ant-v5",
    "Humanoid-v5",
]

ALGORITHMS = ["SAC", "TD3", "DDPG", "PPO"]

# Color map for algorithms
algo_colors = {
    'SAC': '#1f77b4',      # blue
    'TD3': '#ff7f0e',      # orange
    'DDPG': '#2ca02c',     # green
    'PPO': '#9467bd',      # purple
}

print("="*60)
print("Loading final evaluation returns...")
final_returns = pd.read_csv(RESULTS_CSV)
print(f"Loaded {len(final_returns)} entries")

# Check for duplicate (task, algorithm, seed) triples
duplicates = final_returns.groupby(['task', 'algorithm', 'seed']).filter(lambda x: len(x) > 1)
if len(duplicates) > 0:
    print(f"\nFound {len(duplicates)} rows with duplicate (task, algorithm, seed) triples")
    
    # Check if duplicates have same or different eval_return_mean
    different_returns = []
    same_returns_to_drop = []
    
    for (task, algo, seed), group in final_returns.groupby(['task', 'algorithm', 'seed']):
        if len(group) > 1:
            unique_returns = group['final_return_mean'].nunique()
            if unique_returns == 1:
                # Same return values - keep first, mark rest for dropping
                same_returns_to_drop.extend(group.index[1:].tolist())
            else:
                # Different return values - print them
                different_returns.append(group)
    
    if different_returns:
        print(f"\n*** WARNING: {len(different_returns)} (task, algo, seed) groups have DIFFERENT final_return_mean values: ***")
        diff_df = pd.concat(different_returns)
        display_cols = ['task', 'algorithm', 'seed', 'final_return_mean']
        if 'timestamp' in diff_df.columns:
            display_cols.append('timestamp')
        print(diff_df[display_cols].to_string())
    
    if same_returns_to_drop:
        print(f"\nDropping {len(same_returns_to_drop)} duplicate rows with same final_return_mean")
        final_returns = final_returns.drop(same_returns_to_drop).reset_index(drop=True)
        print(f"Final returns after deduplication: {len(final_returns)} entries")
else:
    print("No duplicate (task, algorithm, seed) triples found")

print(f"\nUnique tasks in final_returns: {sorted(final_returns.task.dropna().unique())}")
print(f"Unique algorithms in final_returns: {sorted(final_returns.algorithm.dropna().unique())}")

print("\n" + "="*60)
print("Generating empirical distribution plots")
print("="*60)

# Generate one plot per task
for task in TASKS:
    task_df = final_returns[final_returns['task'] == task]
    
    if len(task_df) == 0:
        print(f"\nSkipping {task}: no data available")
        continue
    
    available_algos = [a for a in ALGORITHMS if a in task_df['algorithm'].unique()]
    
    if len(available_algos) == 0:
        print(f"\nSkipping {task}: no algorithm data available")
        continue
    
    print(f"\nPlotting {task}...")
    print(f"  Available algorithms: {available_algos}")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Collect data for each algorithm
    algo_data = {}
    for algo in available_algos:
        algo_returns = task_df[task_df['algorithm'] == algo]['final_return_mean'].values
        if len(algo_returns) > 0:
            algo_data[algo] = algo_returns
            print(f"  {algo}: {len(algo_returns)} seeds, mean={np.mean(algo_returns):.2f}, std={np.std(algo_returns):.2f}")
    
    if len(algo_data) == 0:
        print(f"  No data to plot for {task}")
        plt.close()
        continue
    
    # Determine bin range from all data
    all_returns = np.concatenate(list(algo_data.values()))
    min_return = np.min(all_returns)
    max_return = np.max(all_returns)
    
    # Use more bins for finer-grained histograms (skinnier bars)
    # Calculate based on data range and standard deviation, but use more bins
    n_bins = min(100, max(40, int((max_return - min_return) / (np.std(all_returns) / 10))))
    
    # Create shared bins for all histograms to ensure consistent bar widths
    bins = np.linspace(min_return, max_return, n_bins + 1)
    
    # Plot histogram for each algorithm using the same bins
    for algo in available_algos:
        if algo not in algo_data:
            continue
        
        returns = algo_data[algo]
        color = algo_colors.get(algo, '#000000')
        
        # Calculate mean and std for this algorithm
        mean_val = np.mean(returns)
        std_val = np.std(returns)
        
        # Create label with mean and std
        label = f"{algo} (μ={mean_val:.2f}, σ={std_val:.2f})"
        
        # Plot histogram with transparency, using the shared bins
        ax.hist(returns, bins=bins, alpha=0.6, label=label, 
                color=color, edgecolor='black', linewidth=0.5)
    
    # Formatting
    ax.set_xlabel('Final Return (mean over evaluation episodes)', fontsize=14)
    ax.set_ylabel('Frequency (number of seeds)', fontsize=14)
    ax.set_title(f'Empirical Distribution of Returns: {task}', 
                fontsize=20, fontweight='bold')
    ax.legend(loc='best', frameon=True, fontsize=16)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    safe_task = task.replace('/', '_').replace('-', '_')
    filename = f"empirical_distribution_{safe_task}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"  Saved: {filepath}")

print("\n" + "="*60)
print(f"All plots saved to: {OUTPUT_DIR}")
print("="*60)

