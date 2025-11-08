# RS-k-means++ (FastLSH) System-Level Optimization Plan

## Current Performance Analysis

Based on benchmark results (25,000 points × 384 dimensions):
- **Current RS-k-means++ (FastLSH)**: 1.83s average runtime
- **Target**: Beat PRONE (1.26s average) while maintaining quality

## Identified Bottlenecks & Optimizations

### 1. **Memory Access & Cache Optimization** ⚡ HIGH IMPACT

#### Issue: Poor Cache Locality
**Location**: `rs_kmeans.cpp:54-63` (preprocessing loop)
```cpp
// Current: Row-major access with poor cache locality
for (int i = 0; i < n; ++i) {
    for (int j = 0; j < d; ++j) {
        mean_[j] += data[i * d + j];  // Scattered writes
    }
}
```

**Optimization**: Blocked computation + SIMD
```cpp
// Blocked algorithm for better cache utilization
constexpr int BLOCK_SIZE = 64;  // L1 cache line size
#pragma omp parallel for reduction(+:mean_[:d_])
for (int block = 0; block < d_; block += BLOCK_SIZE) {
    int block_end = std::min(block + BLOCK_SIZE, d_);
    for (int i = 0; i < n_; ++i) {
        for (int j = block; j < block_end; ++j) {
            mean_[j] += data[i * d_ + j];
        }
    }
}
```

**Expected Speedup**: 2-3× on preprocessing

---

### 2. **FastLSH: Eliminate Float↔Double Conversions** ⚡ HIGH IMPACT

#### Issue: Unnecessary Type Conversions
**Location**: `rs_kmeans.cpp:244-247`, `fast_lsh.cpp:195`

Currently converting float → double → float in hot path:
```cpp
// rs_kmeans.cpp (called k times per iteration)
std::vector<double> query_double(d_);
for (int i = 0; i < d_; ++i) {
    query_double[i] = static_cast<double>(query[i]);  // OVERHEAD!
}
```

**Optimization**: Template FastLSH for both float and double
```cpp
template<typename T = float>
class FastLSH {
    std::vector<T> apply_transform(const std::vector<T>& point, int table_idx);
    void InsertPoint(int point_id, const std::vector<T>& point);
    std::vector<int> QueryPoint(const std::vector<T>& point, int max_candidates = 100);
};
```

**Expected Speedup**: 15-20% reduction in query time

---

### 3. **SIMD Vectorization** ⚡ HIGH IMPACT

#### Issue: Scalar Operations in Hot Loops
**Location**: `fast_lsh.cpp:142-144`, `rs_kmeans.cpp:229-237`

**Optimization**: Use AVX2/AVX-512 for distance computations
```cpp
// Distance computation with SIMD (8 floats at once with AVX)
#ifdef __AVX2__
#include <immintrin.h>

float squared_distance_simd(const float* a, const float* b, int d) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;

    // Process 8 elements at a time
    for (; i + 7 < d; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);  // FMA: multiply-add
    }

    // Horizontal sum
    float result[8];
    _mm256_storeu_ps(result, sum);
    float dist_sq = 0.0f;
    for (int j = 0; j < 8; ++j) dist_sq += result[j];

    // Handle remainder
    for (; i < d; ++i) {
        float diff = a[i] - b[i];
        dist_sq += diff * diff;
    }

    return dist_sq;
}
#endif
```

**Expected Speedup**: 3-4× on distance computations

---

### 4. **Parallel Hadamard Transform** ⚡ MEDIUM IMPACT

#### Issue: Sequential FHT
**Location**: `fast_lsh.cpp:22-56`

**Optimization**: Parallelize outer loop for large transforms
```cpp
void HadamardTransform::fht(std::vector<double>& data) {
    int n = data.size();

    // Parallelize for large n (cache-friendly threshold)
    #pragma omp parallel for if(n >= 1024)
    for (int h = 1; h < n; h *= 2) {
        #pragma omp simd
        for (int i = 0; i < n; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                double x = data[j];
                double y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
    }
}
```

**Expected Speedup**: 2-3× for d > 512

---

### 5. **Hash Table Optimization** ⚡ MEDIUM IMPACT

#### Issue: std::map overhead in buckets
**Location**: `fast_lsh.h` (hash_tables structure)

**Optimization**: Use flat_hash_map (Abseil) or custom bucketing
```cpp
#include <absl/container/flat_hash_map.h>

struct HashTable {
    absl::flat_hash_map<std::vector<int>, std::vector<int>, VectorHasher> buckets;
    // ... rest of structure
};

// Custom hash for vector<int> keys
struct VectorHasher {
    size_t operator()(const std::vector<int>& v) const {
        size_t hash = 0;
        for (int x : v) {
            hash ^= std::hash<int>{}(x) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};
```

**Expected Speedup**: 20-30% faster lookups

---

### 6. **Prefetching** ⚡ LOW-MEDIUM IMPACT

#### Issue: Cache misses on random access
**Location**: `rs_kmeans.cpp:236` (squared_distance calls)

**Optimization**: Software prefetching
```cpp
// Prefetch next point while processing current
for (size_t i = 0; i < candidates.size(); ++i) {
    int point_idx = candidates[i];

    // Prefetch next candidate
    if (i + 1 < candidates.size()) {
        const float* next_point = get_point(candidates[i + 1]);
        __builtin_prefetch(next_point, 0, 3);  // Read, high temporal locality
    }

    float dist = squared_distance(point_idx, center_idx);
    // ... process
}
```

**Expected Speedup**: 10-15% reduction in cache misses

---

### 7. **Batch Processing for k Selection** ⚡ MEDIUM IMPACT

#### Issue: k centers selected sequentially
**Location**: `rs_kmeans.cpp` (cluster() main loop)

**Optimization**: Select multiple centers in parallel when possible
```cpp
// After selecting first few centers, batch process remaining
if (selected_center_indices_.size() >= 10 && k - selected_center_indices_.size() >= 4) {
    // Select 4 centers in parallel (independent D² samples)
    std::vector<int> batch_centers(4);

    #pragma omp parallel for
    for (int b = 0; b < 4; ++b) {
        // Each thread samples independently
        std::mt19937 thread_rng(random_seed + selected_center_indices_.size() + b);
        batch_centers[b] = sample_from_distribution(thread_rng);
    }

    // Add all to index
    for (int center : batch_centers) {
        selected_center_indices_.push_back(center);
        add_center_to_index(center);
    }
}
```

**Expected Speedup**: 20-30% for large k (k > 100)

---

### 8. **Memory Pooling** ⚡ LOW-MEDIUM IMPACT

#### Issue: Repeated allocations in hot paths
**Location**: `fast_lsh.cpp:114-115` (already uses thread_local, but can improve)

**Optimization**: Pre-allocated memory pool
```cpp
struct LSHMemoryPool {
    std::vector<double> transform_buffer1;
    std::vector<double> transform_buffer2;
    std::vector<int> hash_buffer;
    std::unordered_map<int, int> candidate_buffer;

    void resize(int d_padded, int k) {
        transform_buffer1.resize(d_padded);
        transform_buffer2.resize(d_padded);
        hash_buffer.reserve(k);
        candidate_buffer.reserve(1000);  // Typical candidate size
    }
};

// Thread-local pool
static thread_local LSHMemoryPool pool;
```

**Expected Speedup**: 5-10% reduction in allocation overhead

---

### 9. **Compiler Optimizations** ⚡ HIGH IMPACT (Easy Win!)

#### Current Flags
Check `cpp/CMakeLists.txt`:
```cmake
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
```

**Enhanced Optimization Flags**:
```cmake
if(CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native")

    # Aggressive optimizations
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} \
        -ffast-math \                  # Aggressive math optimizations
        -funroll-loops \               # Unroll loops
        -ftree-vectorize \             # Auto-vectorization
        -fno-signed-zeros \            # Assume no signed zeros
        -fno-trapping-math \           # No FP exceptions
        -fassociative-math \           # Allow reassociation
        -freciprocal-math \            # Use reciprocal approximations
        -mavx2 \                       # Enable AVX2 instructions
        -mfma")                        # Enable FMA instructions

    # Link-time optimization
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -flto")
endif()
```

**Expected Speedup**: 15-25% overall

---

### 10. **Profile-Guided Optimization (PGO)** ⚡ MEDIUM IMPACT

**Two-stage compilation**:

1. **Profile generation**:
```bash
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-fprofile-generate" ..
make benchmark
./benchmark  # Generate profile data
```

2. **Profile-optimized build**:
```bash
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-fprofile-use" ..
make benchmark
```

**Expected Speedup**: 10-15% from better branch prediction and inlining

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 days)
1. ✅ Compiler optimization flags
2. ✅ Remove float/double conversions (template FastLSH)
3. ✅ SIMD distance computations

**Expected**: 30-40% speedup

### Phase 2: Core Optimizations (3-5 days)
4. ✅ Cache-friendly memory access patterns
5. ✅ Parallel Hadamard transform
6. ✅ Hash table optimization (flat_hash_map)

**Expected**: Additional 20-30% speedup

### Phase 3: Advanced Optimizations (5-7 days)
7. ✅ Batch center selection
8. ✅ Prefetching
9. ✅ Profile-guided optimization

**Expected**: Additional 15-20% speedup

---

## Total Expected Improvement

**Cumulative speedup**: **2.5-3.5×** faster than current implementation

**Target performance**:
- Current: 1.83s average
- Optimized: **0.52-0.73s** average
- **Beats PRONE by 40-70%!**

---

## Validation Strategy

1. **Correctness**: Ensure optimized code produces identical results
   ```bash
   # Compare clustering costs
   ./benchmark_original > results_original.txt
   ./benchmark_optimized > results_optimized.txt
   diff <(grep "Cost:" results_original.txt) <(grep "Cost:" results_optimized.txt)
   ```

2. **Performance**: Use consistent benchmarking
   ```bash
   # Run multiple times, report mean ± std
   for i in {1..10}; do ./benchmark; done
   ```

3. **Profiling**: Use perf/vtune to verify optimizations
   ```bash
   perf record -g ./benchmark
   perf report
   ```

---

## Hardware-Specific Considerations

### Apple Silicon (M1/M2/M3)
- Use ARM NEON instead of AVX: `-march=armv8.2-a+fp16+simd`
- Efficient cores vs Performance cores: pin threads appropriately
- Unified memory architecture: exploit zero-copy between CPU/GPU

### Intel/AMD x86-64
- AVX-512 on newer CPUs: `-mavx512f -mavx512dq`
- Large L3 cache: increase blocking factor
- Hyp threading: use physical cores only for compute-bound tasks

---

## Monitoring & Continuous Optimization

1. **Add instrumentation**:
```cpp
#ifdef PROFILE_MODE
#define PROFILE_SECTION(name) ProfilerGuard __profiler_##name(#name)
#else
#define PROFILE_SECTION(name)
#endif
```

2. **Benchmark suite**:
- Small (n=1K, k=10)
- Medium (n=25K, k=100)
- Large (n=100K, k=1000)
- High-dim (n=10K, d=2048)

3. **Performance regression tests**:
- CI/CD pipeline runs benchmarks
- Alert if >5% regression
- Track performance over time

---

## Alternative: GPU Acceleration

For even more speedup (10-50×), consider GPU implementation:

```cpp
// CUDA kernel for distance computations
__global__ void compute_distances_kernel(
    const float* data,
    const float* centers,
    float* distances,
    int n, int k, int d
) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= n) return;

    float min_dist = FLT_MAX;
    for (int c = 0; c < k; ++c) {
        float dist = 0.0f;
        for (int j = 0; j < d; ++j) {
            float diff = data[point_idx * d + j] - centers[c * d + j];
            dist += diff * diff;
        }
        min_dist = fminf(min_dist, dist);
    }
    distances[point_idx] = min_dist;
}
```

Use libraries:
- RAPIDS cuML for GPU k-means
- Thrust for GPU parallel primitives
- cuBLAS for matrix operations

---

## References

- Intel Optimization Manual: https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/
- Agner Fog's Optimization Guides: https://www.agner.org/optimize/
- SIMD Tutorial: https://www.cs.virginia.edu/~cr4bd/3330/F2018/simdref.html
- Abseil flat_hash_map: https://abseil.io/docs/cpp/guides/container
