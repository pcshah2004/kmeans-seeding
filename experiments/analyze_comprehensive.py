"""
Analyze comprehensive IMDB experiment results - all algorithms and index types.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Read results
results_path = Path(__file__).parent / "imdb_comprehensive_results.csv"
df = pd.read_csv(results_path)

# Filter successful runs
df = df[df['status'] == 'success'].copy()

print("="*80)
print("IMDB DATASET - COMPREHENSIVE ALGORITHM COMPARISON")
print("All RS-k-means++ Index Types + Fast-LSH + AFK-MC²")
print("="*80)

# Display full results
print("\nFull Results:")
print("="*80)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
print(df[['algorithm', 'k', 'time_seconds', 'initial_cost']].to_string(index=False))

# Average statistics
print("\n" + "="*80)
print("AVERAGE STATISTICS (across all k values)")
print("="*80)

summary_data = []
for algo in df['algorithm'].unique():
    algo_df = df[df['algorithm'] == algo]
    avg_time = algo_df['time_seconds'].mean()
    avg_cost = algo_df['initial_cost'].mean()

    summary_data.append({
        'Algorithm': algo,
        'Avg Time (s)': avg_time,
        'Avg Cost': avg_cost
    })

summary_df = pd.DataFrame(summary_data).sort_values('Avg Time (s)')
print(summary_df.to_string(index=False))

# Compute speedups and quality relative to k-means++
print("\n" + "="*80)
print("RELATIVE TO k-means++ (Averages)")
print("="*80)

kpp_avg_time = df[df['algorithm'] == 'k-means++']['time_seconds'].mean()
kpp_avg_cost = df[df['algorithm'] == 'k-means++']['initial_cost'].mean()

comparison_data = []
for algo in df['algorithm'].unique():
    if algo == 'k-means++':
        continue

    algo_df = df[df['algorithm'] == algo]
    avg_time = algo_df['time_seconds'].mean()
    avg_cost = algo_df['initial_cost'].mean()

    speedup = kpp_avg_time / avg_time
    cost_ratio = avg_cost / kpp_avg_cost

    comparison_data.append({
        'Algorithm': algo,
        'Speedup': speedup,
        'Cost Ratio': cost_ratio
    })

comp_df = pd.DataFrame(comparison_data).sort_values('Speedup', ascending=False)
print(comp_df.to_string(index=False))

# RS-k-means++ index comparison
print("\n" + "="*80)
print("RS-k-means++ INDEX TYPE COMPARISON")
print("="*80)

rs_algos = [a for a in df['algorithm'].unique() if 'RS-k-means++' in a]
rs_comparison = []

for algo in rs_algos:
    algo_df = df[df['algorithm'] == algo]
    avg_time = algo_df['time_seconds'].mean()
    avg_cost = algo_df['initial_cost'].mean()

    index_type = algo.split('(')[1].rstrip(')')

    rs_comparison.append({
        'Index Type': index_type,
        'Avg Time (s)': avg_time,
        'Avg Cost': avg_cost,
        'Speedup vs Flat': df[df['algorithm'] == 'RS-k-means++ (Flat)']['time_seconds'].mean() / avg_time
    })

rs_comp_df = pd.DataFrame(rs_comparison).sort_values('Avg Time (s)')
print(rs_comp_df.to_string(index=False))

# Key findings
print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

# Find fastest overall
fastest_algo = summary_df.iloc[0]['Algorithm']
fastest_time = summary_df.iloc[0]['Avg Time (s)']
fastest_speedup = kpp_avg_time / fastest_time

print(f"\n1. FASTEST OVERALL: {fastest_algo}")
print(f"   - {fastest_speedup:.2f}x faster than k-means++")
print(f"   - Average time: {fastest_time:.3f}s")

# Find best RS-k-means++ index
best_rs_idx = rs_comp_df.iloc[0]
print(f"\n2. BEST RS-k-means++ INDEX: {best_rs_idx['Index Type']}")
print(f"   - {best_rs_idx['Speedup vs Flat']:.2f}x vs Flat")
print(f"   - Average time: {best_rs_idx['Avg Time (s)']:.3f}s")

# Quality comparison
best_quality = comp_df.loc[comp_df['Cost Ratio'].idxmin()]
print(f"\n3. BEST QUALITY (closest to k-means++): {best_quality['Algorithm']}")
print(f"   - Cost ratio: {best_quality['Cost Ratio']:.4f}x")

# RS-k-means++ LSH vs Fast-LSH
rs_lsh_time = df[df['algorithm'] == 'RS-k-means++ (LSH)']['time_seconds'].mean()
fast_lsh_time = df[df['algorithm'] == 'Fast-LSH']['time_seconds'].mean()
rs_lsh_cost = df[df['algorithm'] == 'RS-k-means++ (LSH)']['initial_cost'].mean()
fast_lsh_cost = df[df['algorithm'] == 'Fast-LSH']['initial_cost'].mean()

print(f"\n4. LSH COMPARISON:")
print(f"   RS-k-means++ (LSH):  {rs_lsh_time:.3f}s, cost ratio: {rs_lsh_cost/kpp_avg_cost:.4f}x")
print(f"   Fast-LSH:            {fast_lsh_time:.3f}s, cost ratio: {fast_lsh_cost/kpp_avg_cost:.4f}x")
print(f"   Fast-LSH is {rs_lsh_time/fast_lsh_time:.2f}x faster!")

# Create visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Time comparison for all algorithms
algo_order = summary_df['Algorithm'].tolist()
colors = plt.cm.tab10(np.linspace(0, 1, len(algo_order)))

for i, algo in enumerate(algo_order):
    algo_df = df[df['algorithm'] == algo]
    ax1.plot(algo_df['k'], algo_df['time_seconds'],
             marker='o', label=algo, linewidth=2, markersize=6, color=colors[i])

ax1.set_xlabel('Number of clusters (k)', fontsize=11)
ax1.set_ylabel('Time (seconds)', fontsize=11)
ax1.set_title('Seeding Time Comparison', fontsize=13, fontweight='bold')
ax1.legend(fontsize=8, loc='best')
ax1.grid(True, alpha=0.3)

# Plot 2: Cost comparison for all algorithms
for i, algo in enumerate(algo_order):
    algo_df = df[df['algorithm'] == algo]
    ax2.plot(algo_df['k'], algo_df['initial_cost'],
             marker='o', label=algo, linewidth=2, markersize=6, color=colors[i])

ax2.set_xlabel('Number of clusters (k)', fontsize=11)
ax2.set_ylabel('Initial Cost', fontsize=11)
ax2.set_title('Initial Clustering Cost Comparison', fontsize=13, fontweight='bold')
ax2.legend(fontsize=8, loc='best')
ax2.grid(True, alpha=0.3)

# Plot 3: RS-k-means++ index comparison (time)
for algo in rs_algos:
    algo_df = df[df['algorithm'] == algo]
    index_type = algo.split('(')[1].rstrip(')')
    ax3.plot(algo_df['k'], algo_df['time_seconds'],
             marker='s', label=index_type, linewidth=2, markersize=7)

ax3.set_xlabel('Number of clusters (k)', fontsize=11)
ax3.set_ylabel('Time (seconds)', fontsize=11)
ax3.set_title('RS-k-means++: Index Type Comparison (Time)', fontsize=13, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: Speedup comparison (bar chart for k=300)
k_test = 300
speedups = []
labels = []

kpp_time_k300 = df[(df['algorithm'] == 'k-means++') & (df['k'] == k_test)]['time_seconds'].values[0]

for algo in algo_order:
    if algo == 'k-means++':
        continue
    algo_time = df[(df['algorithm'] == algo) & (df['k'] == k_test)]['time_seconds'].values[0]
    speedup = kpp_time_k300 / algo_time
    speedups.append(speedup)
    labels.append(algo.replace('RS-k-means++ ', '').replace('(', '').replace(')', ''))

# Sort by speedup
sorted_idx = np.argsort(speedups)[::-1]
speedups = [speedups[i] for i in sorted_idx]
labels = [labels[i] for i in sorted_idx]

bars = ax4.barh(labels, speedups, color=plt.cm.RdYlGn(np.array(speedups)/max(speedups)))
ax4.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.5, label='k-means++ baseline')
ax4.set_xlabel('Speedup vs k-means++', fontsize=11)
ax4.set_title(f'Speedup Comparison (k={k_test})', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

# Add value labels on bars
for i, (bar, speedup) in enumerate(zip(bars, speedups)):
    ax4.text(speedup + 0.1, i, f'{speedup:.2f}x',
             va='center', fontsize=9, fontweight='bold')

plt.tight_layout()

# Save plot
plot_path = Path(__file__).parent / "imdb_comprehensive_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\n{'='*80}")
print(f"Plot saved to: {plot_path}")
print(f"{'='*80}")

plt.close()

# Summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)

print(f"\n{'Algorithm':<30} {'Avg Time':>10} {'Speedup':>10} {'Cost Ratio':>12}")
print("-" * 80)
print(f"{'k-means++ (baseline)':<30} {kpp_avg_time:>9.3f}s {'1.00x':>10} {'1.0000x':>12}")

for _, row in comp_df.iterrows():
    algo = row['Algorithm']
    algo_time = df[df['algorithm'] == algo]['time_seconds'].mean()
    print(f"{algo:<30} {algo_time:>9.3f}s {row['Speedup']:>9.2f}x {row['Cost Ratio']:>11.4f}x")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
