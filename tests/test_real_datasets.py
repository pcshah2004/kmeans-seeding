"""
Test seeding quality on real UCI ML datasets.
"""

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler
from kmeans_seeding import (
    kmeanspp,
    rejection_sampling,
    afkmc2,
)


def compute_initial_cost(X, centers):
    """Compute k-means cost for given centers (no iteration)."""
    n_samples = X.shape[0]
    min_distances_sq = np.zeros(n_samples)

    for i in range(n_samples):
        distances_sq = np.sum((centers - X[i]) ** 2, axis=1)
        min_distances_sq[i] = np.min(distances_sq)

    return np.sum(min_distances_sq)


def test_dataset(X, name, n_clusters, n_trials=5):
    """Test all methods on a dataset with multiple random seeds."""
    print(f"\n{'='*70}")
    print(f"Dataset: {name}")
    print(f"Shape: {X.shape}, k={n_clusters}")
    print(f"{'='*70}")

    methods = {
        'k-means++': kmeanspp,
        'RS-k-means++': rejection_sampling,
        'AFK-MC²': afkmc2,
    }

    # Test with multiple seeds
    seeds = [42, 123, 456, 789, 999][:n_trials]

    all_results = {name: [] for name in methods.keys()}

    for seed in seeds:
        for method_name, method in methods.items():
            centers = method(X, n_clusters=n_clusters, random_state=seed)
            cost = compute_initial_cost(X, centers)
            all_results[method_name].append(cost)

    # Print results
    print(f"\n{'Method':<15} | {'Mean Cost':>12} | {'Std Dev':>10} | {'Min':>12} | {'Max':>12}")
    print("-" * 70)

    for method_name in methods.keys():
        costs = all_results[method_name]
        mean_cost = np.mean(costs)
        std_cost = np.std(costs)
        min_cost = np.min(costs)
        max_cost = np.max(costs)

        print(f"{method_name:<15} | {mean_cost:12.2f} | {std_cost:10.2f} | {min_cost:12.2f} | {max_cost:12.2f}")

    # Compare methods
    print(f"\n{'Comparison:'}")
    kpp_mean = np.mean(all_results['k-means++'])
    rs_mean = np.mean(all_results['RS-k-means++'])
    afk_mean = np.mean(all_results['AFK-MC²'])

    print(f"  RS-k-means++ vs k-means++: {rs_mean/kpp_mean:.2f}x")
    print(f"  AFK-MC² vs k-means++:      {afk_mean/kpp_mean:.2f}x")

    # Find best method
    best_method = min(methods.keys(), key=lambda m: np.mean(all_results[m]))
    print(f"  Best (lowest mean): {best_method}")


if __name__ == "__main__":
    print("Testing seeding quality on REAL UCI ML datasets")
    print("(Testing INITIAL cost after seeding, before k-means iterations)")

    # Dataset 1: Iris (classic, small)
    iris = load_iris()
    X_iris = StandardScaler().fit_transform(iris.data).astype(np.float64)
    test_dataset(X_iris, "Iris", n_clusters=3, n_trials=5)

    # Dataset 2: Wine (medium)
    wine = load_wine()
    X_wine = StandardScaler().fit_transform(wine.data).astype(np.float64)
    test_dataset(X_wine, "Wine", n_clusters=3, n_trials=5)

    # Dataset 3: Breast Cancer (larger, binary)
    cancer = load_breast_cancer()
    X_cancer = StandardScaler().fit_transform(cancer.data).astype(np.float64)
    test_dataset(X_cancer, "Breast Cancer", n_clusters=2, n_trials=5)

    # Dataset 4: Digits (high-dimensional, many clusters)
    digits = load_digits()
    X_digits = StandardScaler().fit_transform(digits.data).astype(np.float64)
    test_dataset(X_digits, "Digits (64-dim)", n_clusters=10, n_trials=5)

    # Dataset 5: Digits with more clusters
    test_dataset(X_digits, "Digits (64-dim, k=20)", n_clusters=20, n_trials=5)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("All tests use standardized real UCI ML datasets.")
    print("Results show mean±std over 5 different random seeds.")
