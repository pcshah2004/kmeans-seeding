# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

```bash
# Build the library
mkdir build && cd build
cmake ..
make -j8

# Run example
./examples/basic_usage

# Run benchmarks
./benchmark_imdb

# Clean build
rm -rf build && mkdir build && cd build && cmake .. && make -j8
```

## Project Overview

This is a **pure C++ library** for fast k-means++ initialization algorithms with theoretical guarantees. It was converted from a Python package to pure C++ in November 2025.

### Key Algorithms

1. **Standard k-means++** (Google 2020 implementation)
   - Classic D² sampling with O(nkd) complexity
   - Implementation: `kmeanspp_seeding.cc`

2. **RS-k-means++ with FastLSH** ⭐ **Primary contribution**
   - Rejection sampling with locality-sensitive hashing
   - 95× faster than baseline for small k (after Nov 2025 optimizations)
   - **No FAISS dependency required**
   - Implementation: `fast_lsh.cpp` + `rs_kmeans.cpp`
   - **Major optimizations (Nov 2025)**: SIMD vectorization, parallel FHT, compiler optimizations, prefetching

3. **PRONE** (Projected ONE-dimensional k-means++)
   - Random 1D projection reduces complexity from O(nkd) to O(nnz(X) + n log n)
   - Three projection variants: standard, variance-weighted, covariance-weighted
   - Implementation: `prone.cpp` + `prone.hpp`
   - Fastest algorithm for small-to-medium k

4. **RejectionSamplingLSH** (Google 2020 multi-tree implementation)
   - Tree embedding with LSH for fast sampling
   - Implementation: `rejection_sampling_lsh.cc` + `lsh.cc` + `tree_embedding.cc`

5. **AFK-MC²** (MCMC-based sampling)
   - Requires FAISS
   - Implementation: `afkmc2.cpp`

## Repository Structure

```
.
├── cpp/                              # C++ library source
│   ├── CMakeLists.txt               # C++ library build config
│   ├── include/kmeans_seeding/      # Public headers
│   │   ├── rs_kmeans.hpp            # RS-k-means++ (FastLSH + FAISS)
│   │   ├── afkmc2.hpp               # AFK-MC² algorithm
│   │   ├── fast_lsh.h               # FastLSH data structure
│   │   ├── prone.hpp                # PRONE algorithm
│   │   ├── simd_utils.hpp           # SIMD vectorization utilities
│   │   ├── kmeanspp_seeding.h       # Standard k-means++
│   │   ├── rejection_sampling_lsh.h # Google 2020 rejection sampling
│   │   ├── lsh.h, tree_embedding.h  # LSH infrastructure
│   │   └── [other headers]          # Utilities
│   └── src/                         # Implementation files
│       ├── rs_kmeans.cpp            # RS-k-means++ (requires FAISS)
│       ├── afkmc2.cpp               # AFK-MC² (requires FAISS)
│       ├── fast_lsh.cpp             # FastLSH (no FAISS needed)
│       ├── prone.cpp                # PRONE implementation
│       ├── simd_utils.cpp           # SIMD distance computations
│       ├── kmeanspp_seeding.cc      # Standard k-means++
│       ├── rejection_sampling_lsh.cc # Google 2020 implementation
│       └── [other core files]
│
├── examples/
│   ├── basic_usage.cpp              # Usage demonstration
│   └── CMakeLists.txt
│
├── experiments/
│   ├── benchmark.cpp                # Three-method benchmark
│   ├── benchmark_imdb.cpp           # IMDB benchmark (C++)
│   ├── benchmark_prone_boosted.cpp  # PRONE boosted benchmark
│   ├── test_prone.cpp               # PRONE unit tests
│   ├── plot_results.py              # Python plotting script
│   └── README.md                    # Benchmark documentation
│
├── archive/                         # Legacy code (DO NOT MODIFY)
│   ├── rs_kmeans/                  # Old Python package
│   └── fast_k_means_2020/          # Original standalone C++ tool
│
├── CMakeLists.txt                  # Top-level build (delegates to cpp/)
└── README.md                       # User documentation
```

## Build System Architecture

### CMake Structure

1. **Top-level `CMakeLists.txt`**:
   - Finds dependencies (OpenMP, FAISS)
   - Delegates to `cpp/` subdirectory
   - Builds examples (if `BUILD_EXAMPLES=ON`)
   - Builds experiments (if `BUILD_EXPERIMENTS=ON`)

2. **`cpp/CMakeLists.txt`**:
   - Builds `kmeans_seeding_core` static library
   - Conditionally compiles FAISS-dependent code
   - Sets up install rules

### Key Build Options

- **Optional dependencies**:
  - FAISS: Enables `rs_kmeans.cpp` and `afkmc2.cpp`
  - OpenMP: Parallel distance computations

- **Conditional compilation**:
  - `#ifdef HAS_FAISS` guards FAISS-specific code
  - `rs_kmeans.cpp` and `afkmc2.cpp` only built if FAISS found

### Platform-Specific Setup

**macOS**:
```bash
brew install cmake libomp
conda install -c pytorch faiss-cpu  # Optional
```

**Linux**:
```bash
sudo apt-get install cmake libomp-dev
conda install -c pytorch faiss-cpu  # Optional
```

## C++ API Usage

### Data Format

All algorithms work with **flat arrays** (`std::vector<float>`):
- Data: `n * d` floats (row-major order)
- Point i, dimension j: `data[i * d + j]`

Some algorithms (Google's k-means++, RejectionSamplingLSH) require `std::vector<std::vector<double>>` format. Use the conversion pattern from `basic_usage.cpp`.

### RS-k-means++ (FastLSH)

```cpp
#include "kmeans_seeding/rs_kmeans.hpp"

std::vector<float> data = /* n * d floats */;
int n = 10000, d = 100, k = 50;

rs_kmeans::RSkMeans rskm;
rskm.preprocess(data, n, d);
auto result = rskm.cluster(k, -1, "FastLSH", "", 42);
// result.first: centers (k * d floats)
// result.second: labels (n ints)
```

Index types: `"FastLSH"` (no FAISS), `"Flat"`, `"LSH"`, `"IVFFlat"`, `"HNSW"` (require FAISS)

### RejectionSamplingLSH (Google 2020)

```cpp
#include "kmeans_seeding/rejection_sampling_lsh.h"

std::vector<std::vector<double>> data = /* ... */;
int k = 50;

fast_k_means::RejectionSamplingLSH rslsh;
rslsh.RunAlgorithm(data, k,
    10,    // number_of_trees
    4.0,   // scaling_factor
    0,     // number_greedy_rounds
    0.0    // boosting_prob_factor
);
// rslsh.centers contains selected point indices
```

### Standard k-means++

```cpp
#include "kmeans_seeding/kmeanspp_seeding.h"

std::vector<std::vector<double>> data = /* ... */;
int k = 50;

fast_k_means::KMeansPPSeeding kmpp;
kmpp.RunAlgorithm(data, k, 0);  // 0 = no greedy rounds
// kmpp.centers_ contains selected point indices
```

## Development Workflow

### Making Changes to Algorithms

1. Edit files in `cpp/src/` or `cpp/include/kmeans_seeding/`
2. Rebuild: `cd build && make -j8`
3. Test: Run `./examples/basic_usage` or `./benchmark_imdb`

### Adding a New Example

1. Create `examples/my_example.cpp`
2. Add to `examples/CMakeLists.txt`:
   ```cmake
   add_executable(my_example my_example.cpp)
   target_link_libraries(my_example PRIVATE kmeans_seeding_core)
   ```
3. Rebuild and run: `./examples/my_example`

### Adding a New Benchmark

1. Create `experiments/my_benchmark.cpp`
2. Add to top-level `CMakeLists.txt` in the `BUILD_EXPERIMENTS` section:
   ```cmake
   add_executable(my_benchmark experiments/my_benchmark.cpp)
   target_link_libraries(my_benchmark PRIVATE kmeans_seeding_core)
   if(OpenMP_CXX_FOUND)
       target_link_libraries(my_benchmark PRIVATE OpenMP::OpenMP_CXX)
   endif()
   ```
3. Rebuild and run: `./my_benchmark`

## Important Implementation Details

### FAISS Integration

- **Conditional compilation**: `rs_kmeans.cpp` and `afkmc2.cpp` only compiled if FAISS found
- **Runtime detection**: Code checks `HAS_FAISS` preprocessor flag
- **Graceful degradation**: FastLSH works without FAISS
- **Installation**: `conda install -c pytorch faiss-cpu`

### FastLSH Data Structure

**Location**: `cpp/src/fast_lsh.cpp`, `cpp/include/kmeans_seeding/fast_lsh.h`

**Recent optimizations (Nov 2025)** - **95× speedup achieved**:
- SIMD vectorization (AVX2/NEON) for distance computations: 3-4× speedup
- Parallel Hadamard transform with OpenMP: 2-3× speedup for high dimensions
- Aggressive compiler optimizations (-O3, -march=native, -flto): 15-25% speedup
- Hash table optimization with FNV-1a hash: 20-30% faster lookups
- Software prefetching for cache optimization: 10-15% reduction in cache misses
- Thread-local memory pools eliminate allocations

**Key methods**:
- `insert(vector)`: Add point to index
- `query_knn(point, k)`: Find k approximate nearest neighbors
- `query_radius(point, r)`: Find points within radius r

**Performance** (IMDB-62 dataset, 25k points × 384 dims):
- k=10: 0.019s (95× faster than pre-optimization)
- k=50: 0.054s (33× faster than pre-optimization)

### SIMD Vectorization

**Location**: `cpp/src/simd_utils.cpp`, `cpp/include/kmeans_seeding/simd_utils.hpp`

**Platform support**:
- **x86-64**: AVX2 with FMA instructions (8 floats/operation)
- **ARM (Apple Silicon)**: NEON instructions (4 floats/operation)
- **Fallback**: Scalar implementation for unsupported platforms

**Usage in code**:
```cpp
#include "kmeans_seeding/simd_utils.hpp"

float dist_sq = simd::squared_distance_simd(point_a, point_b, d);
```

**Performance impact**: 3-4× speedup on distance computations (hot path)

### OpenMP Parallelization

- Automatically parallelizes distance computations, preprocessing, and FHT
- Platform-specific setup (see README.md)
- Falls back gracefully if not available
- Control threads: `export OMP_NUM_THREADS=8`
- Uses static scheduling for better cache locality
- Conditional parallelization: only activates for large data to avoid overhead

### PRONE Algorithm

**Location**: `cpp/src/prone.cpp`, `cpp/include/kmeans_seeding/prone.hpp`

**Algorithm**: PRojected ONE-dimensional k-means++ seeding
- Reduces k-means++ from O(nkd) to O(nnz(X) + n log n) via random 1D projection
- Uses dynamic binary tree for efficient D² sampling in 1D

**Projection types**:
1. `STANDARD`: Standard Gaussian projection (default)
2. `VARIANCE_WEIGHTED`: Variance-weighted projection
3. `COVARIANCE`: Covariance-weighted projection

**Best for**: Small to medium k values where asymptotic speedup matters

### Memory Layout

- **Flat arrays**: C-contiguous, row-major (`data[i * d + j]`)
- **No copying**: Algorithms work in-place where possible
- **Large datasets**: Use `std::vector::reserve()` to avoid reallocations

## Benchmarking

### IMDB Benchmark

**Datasets**: Store embeddings in `embeddings/text/`:
- `imdb_embeddings.npy`: NumPy array (n × d, float32)

**Running**:
```bash
cd build
./benchmark_imdb
```

**Output**:
- `experiments/imdb_benchmark_results.csv`: Raw data
- `experiments/imdb_benchmark_log.txt`: Full log

**Plotting**:
```bash
python3 experiments/plot_results.py
```

**Benchmark structure**:
1. Loads `.npy` file using custom numpy reader
2. Tests all algorithms at different k values
3. Measures runtime and k-means cost
4. Writes CSV for analysis

### Creating New Benchmarks

**Pattern** (from `benchmark_imdb.cpp`):
1. Load data (custom format or `.npy`)
2. Convert between flat and Google formats as needed
3. Time each algorithm with `std::chrono`
4. Compute k-means cost for quality comparison
5. Write results to CSV
6. Create Python plotting script

## Common Issues & Solutions

### "FAISS not found" during build

- **Solution**: `conda install -c pytorch faiss-cpu`
- **Alternative**: Build without FAISS (FastLSH still works)
- **Check**: `which conda` and ensure CMake can find FAISS

### "undefined reference to `omp_*`"

- **Cause**: OpenMP not found or incorrectly linked
- **macOS**: `brew install libomp`
- **Linux**: `sudo apt-get install libomp-dev`
- **Fallback**: Code still works without OpenMP (slower)

### Segmentation fault in FastLSH

- **Common cause**: Data dimensionality mismatch
- **Check**: Ensure `n * d` matches actual data size
- **Debug**: Add bounds checking in debug build

### Poor clustering quality

- **Increase LSH parameters**: More tables/hashes improve accuracy
- **Try different index**: FastLSH → LSH → IVFFlat → Flat
- **Baseline**: Compare against standard k-means++ cost

## Code Style & Conventions

### C++ Code

- **Standard**: C++17
- **Compiler flags**: `-std=c++17 -O3 -march=native`
- **Naming**:
  - Classes: `PascalCase` (e.g., `RSkMeans`, `FastLSH`)
  - Functions: `snake_case` (e.g., `compute_cost`, `query_knn`)
  - Member variables: `snake_case_` with trailing underscore
  - Constants: `UPPER_CASE`

### Headers

- Use include guards: `#ifndef KMEANS_SEEDING_FOO_H`
- Public API in `include/kmeans_seeding/`
- Implementation details in `.cpp` files

### Comments

- Algorithm references: Cite papers in header comments
- Complex logic: Explain "why" not "what"
- TODOs: Use `// TODO(name): description` format

## Legacy Code (Do Not Modify)

### `archive/rs_kmeans/`

- Old Python package with pybind11 bindings
- Contains benchmark scripts and experiments
- **Status**: Archived (Nov 2025)
- **Use**: Historical reference only

### `archive/fast_k_means_2020/`

- Original standalone C++ implementation from NeurIPS 2020
- Command-line tool (no library interface)
- **Compilation**: `g++ -std=c++11 -O3 -o fast_kmeans *.cc`
- **Use**: Reproducibility of original paper results

## Publications & References

1. **k-means++: The Advantages of Careful Seeding**
   Arthur & Vassilvitskii, SODA 2007

2. **Fast and Accurate k-means++ via Rejection Sampling**
   Bachem et al., NeurIPS 2016

3. **Approximate k-Means++ in Sublinear Time**
   Bachem et al., AAAI 2016

4. **Scalable k-means++ Clustering via Locality-Sensitive Hashing** (Google 2020)
   Implementation in `rejection_sampling_lsh.cc`

5. **Fast k-means++ with Optimized DHHash-based LSH** (This work, 2025)
   FastLSH implementation with November 2025 optimizations
