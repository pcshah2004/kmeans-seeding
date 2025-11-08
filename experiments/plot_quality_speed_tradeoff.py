#!/usr/bin/env python3
"""
Plot Quality-Speed Tradeoff: K-means Cost vs Runtime
Single plot showing all algorithms with lines connecting k values
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Set publication-quality defaults
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 10

# Read data
df = pd.read_csv('benchmark_results_prone_boosted.csv')

# Color scheme for algorithms (solid colors, colorblind-friendly)
COLORS = {
    'RS-k-means++ (FastLSH)': '#E69F00',      # Orange/Gold
    'RejectionSamplingLSH': '#56B4E9',        # Sky blue
    'AFK-MC²': '#CC79A7',                      # Pink/Purple
    'PRONE (boosted α=0.001)': '#009E73',     # Green
    'PRONE (boosted α=0.01)': '#F0E442',      # Yellow
    'PRONE (boosted α=0.1)': '#0072B2',       # Dark blue
    'Lightweight': '#D55E00',                 # Vermillion/Orange-Red
    'PRONE (Standard)': '#009E73',            # Green
    'PRONE (Variance)': '#F0E442',            # Yellow
    'PRONE (Covariance)': '#999999',          # Gray
}

# Marker styles
MARKERS = {
    'RS-k-means++ (FastLSH)': 'o',            # Circle
    'RejectionSamplingLSH': 's',              # Square
    'AFK-MC²': 'v',                            # Triangle down
    'PRONE (boosted α=0.001)': '^',           # Triangle up
    'PRONE (boosted α=0.01)': 'D',            # Diamond
    'PRONE (boosted α=0.1)': 'p',             # Pentagon
    'Lightweight': '*',                        # Star
    'PRONE (Standard)': '^',
    'PRONE (Variance)': 'D',
    'PRONE (Covariance)': 'p',
}

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

# Plot each algorithm
for algo in sorted(df['algorithm'].unique()):
    algo_data = df[df['algorithm'] == algo].sort_values('k')

    # Plot line connecting points
    ax.plot(algo_data['runtime_seconds'], algo_data['cost'],
            label=algo,
            color=COLORS.get(algo, 'gray'),
            marker=MARKERS.get(algo, 'o'),
            linewidth=2.5,
            markersize=12,
            linestyle='--')

# Styling
ax.set_xlabel('Runtime (seconds)', fontsize=14, fontweight='bold')
ax.set_ylabel('K-means Cost', fontsize=14, fontweight='bold')
ax.set_title('Quality-Speed Tradeoff: K-means Cost vs Runtime',
             fontsize=18, fontweight='bold', pad=20)

# Log scale for both axes
ax.set_xscale('log')
ax.set_yscale('log')

# Grid
ax.grid(True, alpha=0.3, linestyle='--', which='both')

# Legend
ax.legend(loc='upper right', fontsize=11, framealpha=0.95)

# Add "Better" annotation
ax.text(0.05, 0.05, 'Better\n(faster, lower cost)',
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))

# Tight layout
plt.tight_layout()

# Save
plt.savefig('quality_speed_tradeoff.png', dpi=300, bbox_inches='tight')
plt.savefig('quality_speed_tradeoff.pdf', bbox_inches='tight')

print("✓ Saved: quality_speed_tradeoff.png")
print("✓ Saved: quality_speed_tradeoff.pdf")
print(f"\nPlotted {len(df['algorithm'].unique())} algorithms with {len(df['k'].unique())} k values each")
