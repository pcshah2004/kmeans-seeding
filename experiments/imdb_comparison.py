"""
Comprehensive comparison of seeding algorithms on IMDB dataset.

Compares:
- k-means++ (standard)
- RS-k-means++ (our method with rejection sampling + FAISS)
- AFK-MC² (MCMC-based sampling)

Tests k = 100, 200, 300, 400, 500
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from kmeans_seeding import kmeanspp, rskmeans, afkmc2


def load_imdb_embeddings():
    """Load IMDB embeddings from file."""
    embeddings_path = Path(__file__).parent.parent / "embeddings" / "text" / "imdb_embeddings.npy"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"IMDB embeddings not found at {embeddings_path}.\n"
            "Please run the embedding generation script first."
        )

    print(f"Loading IMDB embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    print(f"Loaded embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")

    return embeddings


def compute_initial_cost(X, centers):
    """Compute k-means cost for given centers (before iteration)."""
    n_samples = X.shape[0]
    min_distances_sq = np.zeros(n_samples)

    for i in range(n_samples):
        distances_sq = np.sum((centers - X[i]) ** 2, axis=1)
        min_distances_sq[i] = np.min(distances_sq)

    return np.sum(min_distances_sq)


def run_experiment(X, algorithm_name, algorithm_func, k_values, random_state=42, **kwargs):
    """Run algorithm for multiple k values and record results."""
    print(f"\n{'='*70}")
    print(f"Running {algorithm_name}")
    print(f"{'='*70}")

    results = []

    for k in k_values:
        print(f"\n{algorithm_name} with k={k}...")

        # Run algorithm and measure time
        start_time = time.time()
        centers = algorithm_func(X, n_clusters=k, random_state=random_state, **kwargs)
        end_time = time.time()

        elapsed_time = end_time - start_time

        # Compute initial cost
        initial_cost = compute_initial_cost(X, centers)

        print(f"  Time: {elapsed_time:.3f}s")
        print(f"  Initial cost: {initial_cost:,.2f}")

        results.append({
            'algorithm': algorithm_name,
            'k': k,
            'time_seconds': elapsed_time,
            'initial_cost': initial_cost,
            'random_state': random_state
        })

    return results


def main():
    """Run comprehensive comparison experiment."""
    print("="*70)
    print("IMDB Dataset - Seeding Algorithm Comparison")
    print("="*70)

    # Load IMDB embeddings
    try:
        X = load_imdb_embeddings()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo generate embeddings, run:")
        print("  cd quantization_analysis")
        print("  python generate_all_embeddings.py")
        return

    # Convert to float64 for consistency
    X = X.astype(np.float64)

    print(f"\nDataset info:")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Memory: {X.nbytes / (1024**2):.2f} MB")

    # K values to test
    k_values = [100, 200, 300, 400, 500]
    print(f"\nTesting k values: {k_values}")

    # Random state for reproducibility
    random_state = 42
    print(f"Random state: {random_state}")

    # Run experiments
    all_results = []

    # 1. k-means++ (baseline)
    results_kpp = run_experiment(
        X,
        "k-means++",
        kmeanspp,
        k_values,
        random_state=random_state
    )
    all_results.extend(results_kpp)

    # 2. RS-k-means++ (our method)
    results_rs = run_experiment(
        X,
        "RS-k-means++",
        rskmeans,
        k_values,
        random_state=random_state,
        max_iter=50,
        index_type='LSH'
    )
    all_results.extend(results_rs)

    # 3. AFK-MC²
    results_afk = run_experiment(
        X,
        "AFK-MC²",
        afkmc2,
        k_values,
        random_state=random_state,
        chain_length=200
    )
    all_results.extend(results_afk)

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Save results
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "imdb_comparison_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*70}")
    print(f"Results saved to: {csv_path}")
    print(f"{'='*70}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n{'Algorithm':<20} {'k':<10} {'Time (s)':<12} {'Initial Cost':<15}")
    print("-"*70)
    for _, row in df.iterrows():
        print(f"{row['algorithm']:<20} {row['k']:<10} {row['time_seconds']:<12.3f} {row['initial_cost']:<15,.2f}")

    # Compute speedups relative to k-means++
    print("\n" + "="*70)
    print("SPEEDUP vs k-means++ (time)")
    print("="*70)

    for k in k_values:
        kpp_time = df[(df['algorithm'] == 'k-means++') & (df['k'] == k)]['time_seconds'].values[0]

        print(f"\nk={k}:")
        for algo in ['RS-k-means++', 'AFK-MC²']:
            algo_time = df[(df['algorithm'] == algo) & (df['k'] == k)]['time_seconds'].values[0]
            speedup = kpp_time / algo_time
            print(f"  {algo:<20}: {speedup:6.2f}x")

    # Compute cost ratios relative to k-means++
    print("\n" + "="*70)
    print("COST RATIO vs k-means++ (initial cost)")
    print("="*70)

    for k in k_values:
        kpp_cost = df[(df['algorithm'] == 'k-means++') & (df['k'] == k)]['initial_cost'].values[0]

        print(f"\nk={k}:")
        for algo in ['RS-k-means++', 'AFK-MC²']:
            algo_cost = df[(df['algorithm'] == algo) & (df['k'] == k)]['initial_cost'].values[0]
            ratio = algo_cost / kpp_cost
            print(f"  {algo:<20}: {ratio:6.3f}x")

    print("\n" + "="*70)
    print("Experiment complete!")
    print("="*70)


if __name__ == "__main__":
    main()
