# Experiments

This folder contains benchmark experiments comparing different k-means++ initialization algorithms.

## Benchmark: IMDB Dataset

The `benchmark_imdb.py` script compares all available k-means++ initialization algorithms on text data.

### Algorithms Tested

1. **sklearn k-means++** - Standard implementation from scikit-learn
2. **k-means++ (C++)** - Our C++ implementation
3. **RS-k-means++ (FastLSH)** - Rejection sampling with FastLSH index
4. **AFK-MC²** - Approximate k-Means via Markov Chains

### Running the Benchmark

```bash
# Install dependencies
pip install -r experiments/requirements.txt

# Run benchmark
python experiments/benchmark_imdb.py
```

### Benchmark Configuration

- **Dataset**: 20newsgroups (text data, similar characteristics to IMDB)
- **Samples**: 10,000 documents
- **Features**: 5,000 TF-IDF features
- **k values**: 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
- **Metrics**:
  - Runtime (seconds)
  - K-means cost (sum of squared distances to nearest center)

### Output Files

- `benchmark_imdb_results.csv` - Raw results for all algorithms and k values
- `benchmark_imdb_results.png` - Combined plot showing:
  - Runtime vs k (log scale)
  - Quality-Speed tradeoff (cost vs runtime)
- `benchmark_imdb_cost.png` - Cost comparison across algorithms

### Expected Results

**Speed Ranking** (fastest to slowest):
1. RS-k-means++ (FastLSH) - O(nkd) with low constants
2. AFK-MC² - O(m*k*d) where m << n
3. k-means++ - O(nkd)

**Cost/Quality Ranking** (best to worst):
1. k-means++ (optimal)
2. RS-k-means++ (FastLSH) - Near-optimal with approximation
3. AFK-MC² - Good approximation with probabilistic guarantees

**Tradeoff**: RS-k-means++ and AFK-MC² trade slight quality loss for significant speedup on large datasets.
