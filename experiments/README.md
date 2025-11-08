# Benchmarks

This directory contains benchmark scripts for evaluating k-means++ initialization algorithms.

## Available Benchmarks

### 1. `benchmark.cpp` - Three-Method Benchmark

Compares the three main algorithms:
- **RS-k-means++ (FastLSH)**: Our optimized DHHash-based implementation
- **RejectionSamplingLSH (Google 2020)**: Multi-tree LSH implementation
- **AFK-MC²**: Markov chain Monte Carlo sampling (requires FAISS)

**Usage:**
```bash
# Build
cd build
cmake .. && make benchmark

# Run (uses default IMDB dataset)
./benchmark

# Run with custom dataset
./benchmark /path/to/embeddings.npy

# Plot results
python3 ../experiments/plot_benchmark.py
```

**Output:**
- `benchmark_results.csv`: Raw benchmark data
- `benchmark_runtime_vs_k.pdf/png`: Runtime comparison
- `benchmark_cost_vs_runtime.pdf/png`: Quality-speed tradeoff
- `benchmark_combined.pdf/png`: Combined multi-plot view

### 2. `benchmark_imdb.cpp` - IMDB Dataset Benchmark

Full benchmark on IMDB text embeddings dataset.

**Usage:**
```bash
cd build
./benchmark_imdb
python3 ../experiments/plot_results.py
```

## Dataset Format

Benchmarks expect NumPy `.npy` files with float32 data:
- Shape: `(n, d)` where n = number of points, d = dimensions
- Dtype: `float32`
- Order: C-contiguous (row-major)

**Example:**
```python
import numpy as np

# Create random data
data = np.random.randn(10000, 128).astype(np.float32)

# Save
np.save('my_dataset.npy', data)

# Run benchmark
./build/benchmark my_dataset.npy
```

## Benchmark Parameters

### RS-k-means++ (FastLSH)
- **m**: `-1` (auto-select based on data)
- **index_type**: `"FastLSH"`
- **random_seed**: `42`

### RejectionSamplingLSH (Google 2020)
- **number_of_trees**: `10`
- **scaling_factor**: `4.0`
- **number_greedy_rounds**: `0`
- **boosting_prob_factor**: `0.0`

### AFK-MC²
- **m** (Markov chain length): `max(200, k * 2)`
- **index_type**: `"Flat"` (exact nearest neighbors)
- **random_seed**: `42`

## K Values Tested

Default: `{10, 50, 100, 200, 300, 400, 500}`

Edit the `k_values` vector in the benchmark source to change.

## Creating Custom Benchmarks

**Template:**
```cpp
#include "kmeans_seeding/rs_kmeans.hpp"
#include "kmeans_seeding/rejection_sampling_lsh.h"

int main() {
    // Load data
    std::vector<float> data = load_your_data();
    int n = ..., d = ..., k = 50;

    // RS-k-means++ (FastLSH)
    rs_kmeans::RSkMeans rskm;
    rskm.preprocess(data, n, d);
    auto start = std::chrono::high_resolution_clock::now();
    auto result = rskm.cluster(k, -1, "FastLSH", "", 42);
    auto end = std::chrono::high_resolution_clock::now();
    double runtime = std::chrono::duration<double>(end - start).count();

    // Compute cost and save results
    ...
}
```

Add to `CMakeLists.txt`:
```cmake
add_executable(my_benchmark experiments/my_benchmark.cpp)
target_link_libraries(my_benchmark PRIVATE kmeans_seeding_core)
```

## Plotting

All benchmark plotting scripts use matplotlib with publication-quality defaults:
- **Formats**: PDF (vector) + PNG (raster)
- **DPI**: 300
- **Colors**: Colorblind-friendly palette
- **Font**: Serif, publication-ready

**Dependencies:**
```bash
pip install pandas matplotlib numpy
```
