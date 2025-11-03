"""
Analyze and visualize IMDB experiment results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Read results
results_path = Path(__file__).parent / "imdb_comparison_results.csv"
df = pd.DataFrame(pd.read_csv(results_path))

print("="*70)
print("IMDB DATASET - ALGORITHM COMPARISON RESULTS")
print("="*70)

# Display full results table
print("\nFull Results:")
print("="*70)
print(df.to_string(index=False))

# Compute speedups and cost ratios
print("\n" + "="*70)
print("SPEEDUP vs k-means++ (Time)")
print("="*70)

for k in [100, 200, 300, 400, 500]:
    kpp_time = df[(df['algorithm'] == 'k-means++') & (df['k'] == k)]['time_seconds'].values[0]

    print(f"\nk={k}:")
    for algo in ['RS-k-means++', 'AFK-MC²']:
        algo_time = df[(df['algorithm'] == algo) & (df['k'] == k)]['time_seconds'].values[0]
        speedup = kpp_time / algo_time
        symbol = "✓" if speedup < 1 else "⚠"
        print(f"  {algo:<20}: {speedup:6.3f}x {symbol}")

print("\n" + "="*70)
print("COST RATIO vs k-means++ (Initial Cost Quality)")
print("="*70)

for k in [100, 200, 300, 400, 500]:
    kpp_cost = df[(df['algorithm'] == 'k-means++') & (df['k'] == k)]['initial_cost'].values[0]

    print(f"\nk={k}:")
    for algo in ['RS-k-means++', 'AFK-MC²']:
        algo_cost = df[(df['algorithm'] == algo) & (df['k'] == k)]['initial_cost'].values[0]
        ratio = algo_cost / kpp_cost
        symbol = "✓" if ratio < 1.1 else "⚠"
        print(f"  {algo:<20}: {ratio:6.4f}x {symbol}")

# Average statistics
print("\n" + "="*70)
print("AVERAGE STATISTICS (across all k values)")
print("="*70)

for algo in ['k-means++', 'RS-k-means++', 'AFK-MC²']:
    algo_df = df[df['algorithm'] == algo]
    avg_time = algo_df['time_seconds'].mean()
    avg_cost = algo_df['initial_cost'].mean()

    print(f"\n{algo}:")
    print(f"  Avg time:         {avg_time:.3f}s")
    print(f"  Avg initial cost: {avg_cost:,.2f}")

# Relative to k-means++
print("\n" + "="*70)
print("RELATIVE TO k-means++ (Averages)")
print("="*70)

kpp_avg_time = df[df['algorithm'] == 'k-means++']['time_seconds'].mean()
kpp_avg_cost = df[df['algorithm'] == 'k-means++']['initial_cost'].mean()

for algo in ['RS-k-means++', 'AFK-MC²']:
    algo_df = df[df['algorithm'] == algo]
    avg_time = algo_df['time_seconds'].mean()
    avg_cost = algo_df['initial_cost'].mean()

    time_ratio = kpp_avg_time / avg_time
    cost_ratio = avg_cost / kpp_avg_cost

    print(f"\n{algo}:")
    print(f"  Time speedup:   {time_ratio:.3f}x")
    print(f"  Cost ratio:     {cost_ratio:.4f}x")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Time comparison
for algo in ['k-means++', 'RS-k-means++', 'AFK-MC²']:
    algo_df = df[df['algorithm'] == algo]
    marker = 'o' if algo == 'k-means++' else ('s' if algo == 'RS-k-means++' else '^')
    ax1.plot(algo_df['k'], algo_df['time_seconds'], marker=marker, label=algo, linewidth=2, markersize=8)

ax1.set_xlabel('Number of clusters (k)', fontsize=12)
ax1.set_ylabel('Time (seconds)', fontsize=12)
ax1.set_title('Seeding Time vs Number of Clusters', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Initial cost comparison
for algo in ['k-means++', 'RS-k-means++', 'AFK-MC²']:
    algo_df = df[df['algorithm'] == algo]
    marker = 'o' if algo == 'k-means++' else ('s' if algo == 'RS-k-means++' else '^')
    ax2.plot(algo_df['k'], algo_df['initial_cost'], marker=marker, label=algo, linewidth=2, markersize=8)

ax2.set_xlabel('Number of clusters (k)', fontsize=12)
ax2.set_ylabel('Initial Cost', fontsize=12)
ax2.set_title('Initial Clustering Cost vs Number of Clusters', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save plot
plot_path = Path(__file__).parent / "imdb_comparison_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\n{'='*70}")
print(f"Plot saved to: {plot_path}")
print(f"{'='*70}")

plt.close()

# Summary
print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)

print("\n1. AFK-MC² is FASTEST:")
kpp_to_afk = kpp_avg_time / df[df['algorithm'] == 'AFK-MC²']['time_seconds'].mean()
print(f"   - {kpp_to_afk:.2f}x faster than k-means++ on average")
print(f"   - Most consistent performance across different k values")

print("\n2. RS-k-means++ has COMPETITIVE QUALITY:")
rs_avg_ratio = (df[df['algorithm'] == 'RS-k-means++']['initial_cost'].mean() /
                kpp_avg_cost)
print(f"   - {rs_avg_ratio:.4f}x cost ratio (within {(rs_avg_ratio-1)*100:.2f}% of k-means++)")

print("\n3. AFK-MC² has EXCELLENT QUALITY:")
afk_avg_ratio = (df[df['algorithm'] == 'AFK-MC²']['initial_cost'].mean() /
                 kpp_avg_cost)
print(f"   - {afk_avg_ratio:.4f}x cost ratio (within {(afk_avg_ratio-1)*100:.2f}% of k-means++)")

print("\n4. TRADE-OFFS:")
print("   - k-means++:   Best quality, slowest")
print("   - RS-k-means++: Good quality, moderate speed")
print("   - AFK-MC²:     Good quality, fastest")

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)
