"""
Comprehensive comparison of RS-k-means++ index types on IMDB dataset.

Compares:
- k-means++ (standard baseline)
- RS-k-means++ with Flat (exact search)
- RS-k-means++ with LSH (FAISS LSH)
- RS-k-means++ with IVFFlat
- RS-k-means++ with HNSW
- RS-k-means++ with FastLSH (DHHash)
- MultiTree-LSH (Google 2020 with tree embedding)

Tests k = 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
Note: AFK-MC² skipped (too slow for large k)
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from kmeans_seeding import kmeanspp, rskmeans, afkmc2, multitree_lsh


def load_imdb_embeddings():
    """Load IMDB embeddings from file."""
    embeddings_path = Path(__file__).parent.parent / "embeddings" / "text" / "imdb_embeddings.npy"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"IMDB embeddings not found at {embeddings_path}.\n"
            "Please ensure embeddings exist."
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
        print(f"\n{algorithm_name} with k={k}...", flush=True)

        # Run algorithm and measure time
        start_time = time.time()
        try:
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
                'random_state': random_state,
                'status': 'success'
            })

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            results.append({
                'algorithm': algorithm_name,
                'k': k,
                'time_seconds': -1,
                'initial_cost': -1,
                'random_state': random_state,
                'status': f'error: {str(e)}'
            })

    return results


def main():
    """Run comprehensive comparison experiment."""
    print("="*70)
    print("IMDB Dataset - COMPREHENSIVE Algorithm Comparison")
    print("="*70)

    # Load IMDB embeddings
    try:
        X = load_imdb_embeddings()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return

    # Convert to float64 for consistency
    X = X.astype(np.float64)

    print(f"\nDataset info:")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Memory: {X.nbytes / (1024**2):.2f} MB")

    # K values to test
    k_values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
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

    # 2. RS-k-means++ with Flat (exact search)
    results_rs_flat = run_experiment(
        X,
        "RS-k-means++ (Flat)",
        rskmeans,
        k_values,
        random_state=random_state,
        max_iter=50,
        index_type='Flat'
    )
    all_results.extend(results_rs_flat)

    # 3. RS-k-means++ with LSH (FAISS)
    results_rs_lsh = run_experiment(
        X,
        "RS-k-means++ (LSH)",
        rskmeans,
        k_values,
        random_state=random_state,
        max_iter=50,
        index_type='LSH'
    )
    all_results.extend(results_rs_lsh)

    # 4. RS-k-means++ with IVFFlat
    results_rs_ivf = run_experiment(
        X,
        "RS-k-means++ (IVFFlat)",
        rskmeans,
        k_values,
        random_state=random_state,
        max_iter=50,
        index_type='IVFFlat'
    )
    all_results.extend(results_rs_ivf)

    # 5. RS-k-means++ with HNSW
    results_rs_hnsw = run_experiment(
        X,
        "RS-k-means++ (HNSW)",
        rskmeans,
        k_values,
        random_state=random_state,
        max_iter=50,
        index_type='HNSW'
    )
    all_results.extend(results_rs_hnsw)

    # 6. RS-k-means++ with FastLSH (DHHash)
    results_rs_fastlsh = run_experiment(
        X,
        "RS-k-means++ (FastLSH)",
        rskmeans,
        k_values,
        random_state=random_state,
        max_iter=50,
        index_type='FastLSH'
    )
    all_results.extend(results_rs_fastlsh)

    # 7. AFK-MC² (SKIPPED - too slow for k up to 1000)
    # results_afk = run_experiment(
    #     X,
    #     "AFK-MC²",
    #     afkmc2,
    #     k_values,
    #     random_state=random_state,
    #     chain_length=200
    # )
    # all_results.extend(results_afk)

    # 8. MultiTree-LSH (Google 2020 with tree embedding)
    results_multitree_lsh = run_experiment(
        X,
        "MultiTree-LSH",
        multitree_lsh,
        k_values,
        random_state=random_state,
        n_trees=4
    )
    all_results.extend(results_multitree_lsh)

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Save results
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "imdb_comprehensive_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*70}")
    print(f"Results saved to: {csv_path}")
    print(f"{'='*70}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY - ALL ALGORITHMS")
    print("="*70)

    # Filter out errors
    df_success = df[df['status'] == 'success'].copy()

    print(f"\n{'Algorithm':<25} {'k':<10} {'Time (s)':<12} {'Initial Cost':<15}")
    print("-"*70)
    for _, row in df_success.iterrows():
        print(f"{row['algorithm']:<25} {row['k']:<10} {row['time_seconds']:<12.3f} {row['initial_cost']:<15,.2f}")

    # Compute speedups relative to k-means++
    print("\n" + "="*70)
    print("SPEEDUP vs k-means++ (time)")
    print("="*70)

    for k in k_values:
        kpp_time = df_success[(df_success['algorithm'] == 'k-means++') & (df_success['k'] == k)]['time_seconds'].values
        if len(kpp_time) == 0:
            continue
        kpp_time = kpp_time[0]

        print(f"\nk={k}:")
        for algo in df_success['algorithm'].unique():
            if algo == 'k-means++':
                continue
            algo_data = df_success[(df_success['algorithm'] == algo) & (df_success['k'] == k)]
            if len(algo_data) > 0:
                algo_time = algo_data['time_seconds'].values[0]
                speedup = kpp_time / algo_time
                symbol = "✓" if speedup > 1 else "⚠"
                print(f"  {algo:<30}: {speedup:6.2f}x {symbol}")

    # Compute cost ratios relative to k-means++
    print("\n" + "="*70)
    print("COST RATIO vs k-means++ (initial cost)")
    print("="*70)

    for k in k_values:
        kpp_cost = df_success[(df_success['algorithm'] == 'k-means++') & (df_success['k'] == k)]['initial_cost'].values
        if len(kpp_cost) == 0:
            continue
        kpp_cost = kpp_cost[0]

        print(f"\nk={k}:")
        for algo in df_success['algorithm'].unique():
            if algo == 'k-means++':
                continue
            algo_data = df_success[(df_success['algorithm'] == algo) & (df_success['k'] == k)]
            if len(algo_data) > 0:
                algo_cost = algo_data['initial_cost'].values[0]
                ratio = algo_cost / kpp_cost
                symbol = "✓" if ratio < 1.1 else "⚠"
                print(f"  {algo:<30}: {ratio:6.4f}x {symbol}")

    # Average statistics
    print("\n" + "="*70)
    print("AVERAGE STATISTICS (across all k)")
    print("="*70)

    for algo in df_success['algorithm'].unique():
        algo_df = df_success[df_success['algorithm'] == algo]
        avg_time = algo_df['time_seconds'].mean()
        avg_cost = algo_df['initial_cost'].mean()

        print(f"\n{algo}:")
        print(f"  Avg time:         {avg_time:7.3f}s")
        print(f"  Avg initial cost: {avg_cost:,.2f}")

    print("\n" + "="*70)
    print("Comprehensive experiment complete!")
    print("="*70)


if __name__ == "__main__":
    main()
