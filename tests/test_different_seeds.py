"""
Test RS-k-means++ with different random seeds on blobs dataset.
"""

import numpy as np
from sklearn.datasets import make_blobs
from kmeans_seeding import kmeanspp, rejection_sampling


def compute_initial_cost(X, centers):
    """Compute k-means cost for given centers (no iteration)."""
    n_samples = X.shape[0]
    min_distances_sq = np.zeros(n_samples)

    for i in range(n_samples):
        distances_sq = np.sum((centers - X[i]) ** 2, axis=1)
        min_distances_sq[i] = np.min(distances_sq)

    return np.sum(min_distances_sq)


# Create the blobs dataset
print("Creating well-separated blobs dataset...")
X, _ = make_blobs(n_samples=500, n_features=20, centers=5,
                  cluster_std=1.0, random_state=42)
X = X.astype(np.float64)

print(f"Dataset shape: {X.shape}")
print(f"k = 5")
print()

# Test multiple random seeds
seeds = [1, 10, 42, 100, 123, 456, 999, 1234, 5678, 9999]

print("="*80)
print("Testing different random seeds")
print("="*80)
print(f"{'Seed':>6} | {'k-means++':>12} | {'RS-k-means++':>12} | {'Ratio':>8} | {'Result':>10}")
print("-"*80)

kmeanspp_costs = []
rs_costs = []
ratios = []

for seed in seeds:
    # k-means++
    centers_kpp = kmeanspp(X, n_clusters=5, random_state=seed)
    cost_kpp = compute_initial_cost(X, centers_kpp)

    # RS-k-means++
    centers_rs = rejection_sampling(X, n_clusters=5, random_state=seed)
    cost_rs = compute_initial_cost(X, centers_rs)

    ratio = cost_rs / cost_kpp

    kmeanspp_costs.append(cost_kpp)
    rs_costs.append(cost_rs)
    ratios.append(ratio)

    # Determine result
    if ratio < 1.5:
        result = "✓ Good"
    elif ratio < 3.0:
        result = "~ OK"
    else:
        result = "✗ Bad"

    print(f"{seed:6d} | {cost_kpp:12.2f} | {cost_rs:12.2f} | {ratio:8.2f}x | {result:>10}")

print("-"*80)
print()

# Summary statistics
print("SUMMARY STATISTICS")
print("="*80)

print("\nk-means++:")
print(f"  Mean cost:   {np.mean(kmeanspp_costs):12.2f}")
print(f"  Std dev:     {np.std(kmeanspp_costs):12.2f}")
print(f"  Min cost:    {np.min(kmeanspp_costs):12.2f}")
print(f"  Max cost:    {np.max(kmeanspp_costs):12.2f}")

print("\nRS-k-means++:")
print(f"  Mean cost:   {np.mean(rs_costs):12.2f}")
print(f"  Std dev:     {np.std(rs_costs):12.2f}")
print(f"  Min cost:    {np.min(rs_costs):12.2f}")
print(f"  Max cost:    {np.max(rs_costs):12.2f}")

print("\nRatio (RS / k-means++):")
print(f"  Mean ratio:  {np.mean(ratios):8.2f}x")
print(f"  Std dev:     {np.std(ratios):8.2f}x")
print(f"  Min ratio:   {np.min(ratios):8.2f}x")
print(f"  Max ratio:   {np.max(ratios):8.2f}x")

# Count how many are good
good_count = sum(1 for r in ratios if r < 1.5)
ok_count = sum(1 for r in ratios if 1.5 <= r < 3.0)
bad_count = sum(1 for r in ratios if r >= 3.0)

print(f"\nResults breakdown:")
print(f"  Good (ratio < 1.5x): {good_count}/{len(seeds)} ({100*good_count/len(seeds):.0f}%)")
print(f"  OK (1.5x-3x):        {ok_count}/{len(seeds)} ({100*ok_count/len(seeds):.0f}%)")
print(f"  Bad (ratio >= 3x):   {bad_count}/{len(seeds)} ({100*bad_count/len(seeds):.0f}%)")

print()
print("="*80)

# Find best and worst seeds
best_idx = np.argmin(ratios)
worst_idx = np.argmax(ratios)

print(f"\nBest seed:  {seeds[best_idx]} (ratio = {ratios[best_idx]:.2f}x)")
print(f"Worst seed: {seeds[worst_idx]} (ratio = {ratios[worst_idx]:.2f}x)")
