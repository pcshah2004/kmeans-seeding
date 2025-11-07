"""
Test seeding quality across different datasets.
Compares the INITIAL cost (before k-means iterations) of different seeding methods.
"""

import numpy as np
from sklearn.datasets import make_blobs, make_classification
from kmeans_seeding import (
    kmeanspp,
    rejection_sampling,
    afkmc2,
    fast_lsh,
    rejection_sampling_lsh_2020,
)


def compute_initial_cost(X, centers):
    """Compute k-means cost for given centers (no iteration)."""
    # For each point, find distance to nearest center
    n_samples = X.shape[0]
    min_distances_sq = np.zeros(n_samples)

    for i in range(n_samples):
        distances_sq = np.sum((centers - X[i]) ** 2, axis=1)
        min_distances_sq[i] = np.min(distances_sq)

    return np.sum(min_distances_sq)


def test_on_dataset(X, dataset_name, n_clusters=5, random_state=42):
    """Test all seeding methods on a dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Shape: {X.shape}, k={n_clusters}, random_state={random_state}")
    print(f"{'='*60}")

    methods = {
        'k-means++': kmeanspp,
        'RS-k-means++': rejection_sampling,
        'AFK-MC²': afkmc2,
        'Fast-LSH': fast_lsh,
        'RS-LSH-2020': rejection_sampling_lsh_2020,
    }

    costs = {}
    for name, method in methods.items():
        # Get initial centers
        centers = method(X, n_clusters=n_clusters, random_state=random_state)

        # Compute initial cost (before k-means iterations)
        cost = compute_initial_cost(X, centers)
        costs[name] = cost

        print(f"{name:15s}: {cost:12.2f}")

    # Compare quality
    cost_values = list(costs.values())
    max_cost = max(cost_values)
    min_cost = min(cost_values)
    ratio = max_cost / min_cost

    print(f"\nQuality ratio (max/min): {ratio:.2f}x")

    # Find best and worst
    best = min(costs.items(), key=lambda x: x[1])
    worst = max(costs.items(), key=lambda x: x[1])

    print(f"Best:  {best[0]} ({best[1]:.2f})")
    print(f"Worst: {worst[0]} ({worst[1]:.2f})")

    return costs, ratio


if __name__ == "__main__":
    print("Testing seeding quality on different datasets")
    print("(Testing INITIAL cost after seeding, not after k-means)")

    # Dataset 1: Well-separated blobs
    print("\n" + "="*60)
    print("DATASET 1: Well-separated blobs")
    X1, _ = make_blobs(n_samples=500, n_features=20, centers=5,
                       cluster_std=1.0, random_state=42)
    X1 = X1.astype(np.float64)
    test_on_dataset(X1, "Blobs (well-separated)", n_clusters=5, random_state=42)

    # Dataset 2: Overlapping blobs
    print("\n" + "="*60)
    print("DATASET 2: Overlapping blobs")
    X2, _ = make_blobs(n_samples=1000, n_features=50, centers=10,
                       cluster_std=5.0, random_state=123)
    X2 = X2.astype(np.float64)
    test_on_dataset(X2, "Blobs (overlapping)", n_clusters=10, random_state=42)

    # Dataset 3: Classification dataset
    print("\n" + "="*60)
    print("DATASET 3: Classification dataset")
    X3, _ = make_classification(n_samples=800, n_features=30, n_informative=20,
                                n_redundant=5, n_clusters_per_class=2,
                                random_state=456)
    X3 = X3.astype(np.float64)
    test_on_dataset(X3, "Classification", n_clusters=8, random_state=42)

    # Dataset 4: High-dimensional uniform
    print("\n" + "="*60)
    print("DATASET 4: High-dimensional uniform random")
    X4 = np.random.randn(2000, 100).astype(np.float64)
    test_on_dataset(X4, "Uniform random", n_clusters=20, random_state=42)

    # Dataset 5: Different random seed on blobs
    print("\n" + "="*60)
    print("DATASET 5: Blobs with different random_state")
    test_on_dataset(X1, "Blobs (random_state=999)", n_clusters=5, random_state=999)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("All tests compare INITIAL seeding quality (cost before k-means iterations)")
    print("A good seeding method should have consistently low initial cost.")
