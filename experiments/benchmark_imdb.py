"""
Benchmark all k-means++ initialization algorithms on IMDB dataset.

This script compares:
- Standard k-means++
- RS-k-means++ (with FastLSH)
- AFK-MC²
- Fast-LSH k-means++

For k = 10, 50, 100, 200, 300, ..., 1000
Metrics: Runtime and final k-means cost
"""

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from kmeans_seeding import rskmeans, afkmc2, kmeanspp
    HAVE_CEXT = True
except ImportError:
    print("Warning: C++ extensions not available. Using sklearn k-means++ only.")
    HAVE_CEXT = False


def load_imdb_dataset(max_samples=10000, max_features=5000):
    """
    Load IMDB-like dataset (using 20newsgroups as proxy with text data).

    Parameters
    ----------
    max_samples : int
        Maximum number of samples to use
    max_features : int
        Maximum number of TF-IDF features

    Returns
    -------
    X : ndarray
        Feature matrix (n_samples, n_features)
    """
    print("Loading 20newsgroups dataset (text data similar to IMDB)...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    print(f"Vectorizing text with TF-IDF (max_features={max_features})...")
    vectorizer = TfidfVectorizer(max_features=max_features, max_df=0.5, min_df=2)
    X = vectorizer.fit_transform(newsgroups.data[:max_samples])
    X = X.toarray()  # Convert to dense

    print(f"Dataset shape: {X.shape}")
    return X


def compute_kmeans_cost(X, centers):
    """Compute k-means cost given centers."""
    n = X.shape[0]
    # Compute distances to nearest center
    min_dists = np.zeros(n)
    for i in range(n):
        dists = np.sum((X[i] - centers)**2, axis=1)
        min_dists[i] = np.min(dists)
    return np.sum(min_dists)


def run_sklearn_kmeanspp(X, k, random_state=42):
    """Run standard k-means++ (sklearn implementation)."""
    start_time = time.time()
    # Initialize with k-means++ but don't run Lloyd's algorithm
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=1, random_state=random_state)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    runtime = time.time() - start_time

    # Compute final cost
    cost = compute_kmeans_cost(X, centers)

    return runtime, cost


def run_kmeanspp(X, k, random_state=42):
    """Run standard k-means++ (our C++ implementation)."""
    start_time = time.time()
    centers = kmeanspp(X, n_clusters=k, random_state=random_state)
    runtime = time.time() - start_time

    cost = compute_kmeans_cost(X, centers)
    return runtime, cost


def run_rskmeans(X, k, index_type='FastLSH', random_state=42):
    """Run RS-k-means++ with specified index."""
    start_time = time.time()
    centers = rskmeans(X, n_clusters=k, index_type=index_type, random_state=random_state)
    runtime = time.time() - start_time

    cost = compute_kmeans_cost(X, centers)
    return runtime, cost


def run_afkmc2(X, k, m=None, random_state=42):
    """Run AFK-MC² algorithm."""
    if m is None:
        m = min(k * 10, X.shape[0] // 2)  # Default: 10x oversampling

    start_time = time.time()
    centers = afkmc2(X, n_clusters=k, m=m, random_state=random_state)
    runtime = time.time() - start_time

    cost = compute_kmeans_cost(X, centers)
    return runtime, cost


def benchmark_algorithm(X, k_values, algorithm_name, algorithm_func, **kwargs):
    """
    Benchmark a single algorithm across different k values.

    Parameters
    ----------
    X : ndarray
        Data matrix
    k_values : list
        List of k values to test
    algorithm_name : str
        Name of the algorithm
    algorithm_func : callable
        Function to run the algorithm
    **kwargs : dict
        Additional arguments to pass to algorithm_func

    Returns
    -------
    results : list of dict
        Results for each k value
    """
    results = []

    print(f"\n{'='*60}")
    print(f"Benchmarking: {algorithm_name}")
    print(f"{'='*60}")

    for k in k_values:
        print(f"  k={k:4d}...", end=' ', flush=True)

        try:
            runtime, cost = algorithm_func(X, k, **kwargs)
            results.append({
                'algorithm': algorithm_name,
                'k': k,
                'runtime': runtime,
                'cost': cost
            })
            print(f"✓ Runtime: {runtime:.3f}s, Cost: {cost:.2e}")

        except Exception as e:
            print(f"✗ Error: {e}")
            results.append({
                'algorithm': algorithm_name,
                'k': k,
                'runtime': np.nan,
                'cost': np.nan
            })

    return results


def plot_results(df, output_path='experiments/benchmark_imdb_results.png'):
    """
    Create a comprehensive plot showing cost vs runtime for all algorithms.

    Parameters
    ----------
    df : DataFrame
        Results dataframe with columns: algorithm, k, runtime, cost
    output_path : str
        Path to save the plot
    """
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Define colors and markers for each algorithm
    styles = {
        'sklearn k-means++': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},
        'k-means++ (C++)': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '-'},
        'RS-k-means++ (FastLSH)': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-'},
        'AFK-MC²': {'color': '#d62728', 'marker': 'D', 'linestyle': '-'},
    }

    # Plot 1: Runtime vs k
    ax1 = axes[0]
    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo].sort_values('k')
        if algo in styles:
            ax1.plot(algo_df['k'], algo_df['runtime'],
                    label=algo,
                    color=styles[algo]['color'],
                    marker=styles[algo]['marker'],
                    markersize=8,
                    linewidth=2,
                    linestyle=styles[algo]['linestyle'],
                    alpha=0.8)

    ax1.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Initialization Runtime vs k', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Cost vs Runtime (scatter plot)
    ax2 = axes[1]
    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo].sort_values('k')
        if algo in styles:
            # Add arrows to show direction of increasing k
            for i in range(len(algo_df) - 1):
                x1, x2 = algo_df.iloc[i]['runtime'], algo_df.iloc[i+1]['runtime']
                y1, y2 = algo_df.iloc[i]['cost'], algo_df.iloc[i+1]['cost']
                ax2.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->',
                                         color=styles[algo]['color'],
                                         alpha=0.3, lw=1.5))

            ax2.scatter(algo_df['runtime'], algo_df['cost'],
                       label=algo,
                       color=styles[algo]['color'],
                       marker=styles[algo]['marker'],
                       s=100,
                       alpha=0.8,
                       edgecolors='black',
                       linewidth=0.5)

            # Annotate k values
            for _, row in algo_df.iterrows():
                if row['k'] in [10, 100, 300, 500, 1000]:  # Annotate select k values
                    ax2.annotate(f"k={row['k']}",
                               xy=(row['runtime'], row['cost']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.7)

    ax2.set_xlabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('K-means Cost', fontsize=12, fontweight='bold')
    ax2.set_title('Quality-Speed Tradeoff', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def plot_cost_comparison(df, output_path='experiments/benchmark_imdb_cost.png'):
    """Plot cost comparison across algorithms."""
    fig, ax = plt.subplots(figsize=(12, 6))

    styles = {
        'sklearn k-means++': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},
        'k-means++ (C++)': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '-'},
        'RS-k-means++ (FastLSH)': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-'},
        'AFK-MC²': {'color': '#d62728', 'marker': 'D', 'linestyle': '-'},
    }

    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo].sort_values('k')
        if algo in styles:
            ax.plot(algo_df['k'], algo_df['cost'],
                   label=algo,
                   color=styles[algo]['color'],
                   marker=styles[algo]['marker'],
                   markersize=8,
                   linewidth=2,
                   linestyle=styles[algo]['linestyle'],
                   alpha=0.8)

    ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('K-means Cost', fontsize=12, fontweight='bold')
    ax.set_title('Clustering Cost vs k', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Cost plot saved to: {output_path}")
    plt.close()


def main():
    """Main benchmark function."""
    # Configuration
    K_VALUES = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    MAX_SAMPLES = 10000
    MAX_FEATURES = 5000
    RANDOM_STATE = 42

    print("="*80)
    print("K-MEANS++ INITIALIZATION ALGORITHMS BENCHMARK")
    print("Dataset: 20newsgroups (text data)")
    print("="*80)

    # Load dataset
    X = load_imdb_dataset(max_samples=MAX_SAMPLES, max_features=MAX_FEATURES)

    # Run benchmarks
    all_results = []

    # 1. sklearn k-means++
    results = benchmark_algorithm(
        X, K_VALUES,
        'sklearn k-means++',
        run_sklearn_kmeanspp,
        random_state=RANDOM_STATE
    )
    all_results.extend(results)

    if HAVE_CEXT:
        # 2. Our k-means++ (C++)
        results = benchmark_algorithm(
            X, K_VALUES,
            'k-means++ (C++)',
            run_kmeanspp,
            random_state=RANDOM_STATE
        )
        all_results.extend(results)

        # 3. RS-k-means++ with FastLSH
        results = benchmark_algorithm(
            X, K_VALUES,
            'RS-k-means++ (FastLSH)',
            run_rskmeans,
            index_type='FastLSH',
            random_state=RANDOM_STATE
        )
        all_results.extend(results)

        # 4. AFK-MC²
        results = benchmark_algorithm(
            X, K_VALUES,
            'AFK-MC²',
            run_afkmc2,
            m=None,  # Auto: 10x oversampling
            random_state=RANDOM_STATE
        )
        all_results.extend(results)

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Save results
    output_csv = 'experiments/benchmark_imdb_results.csv'
    df.to_csv(output_csv, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*80}\n")

    # Display summary
    print("\nSUMMARY STATISTICS:")
    print("="*80)
    summary = df.groupby('algorithm').agg({
        'runtime': ['mean', 'std', 'min', 'max'],
        'cost': ['mean', 'std', 'min', 'max']
    })
    print(summary)

    # Create plots
    plot_results(df)
    plot_cost_comparison(df)

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
