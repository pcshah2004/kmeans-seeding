# FAISS Optional Dependency Implementation

**Date**: November 7, 2025
**Status**: ✅ Complete

## Summary

Successfully implemented conditional compilation to make FAISS a truly optional dependency. The package now builds and works correctly both with and without FAISS installed.

## Problem Statement

### Before

- C++ code had **unconditional** `#include <faiss/...>` statements
- Code would **not compile** without FAISS headers
- Despite being marked "optional" in `pyproject.toml`, FAISS was actually **required at build time**
- Users installing from PyPI wheels needed FAISS installed at runtime (dynamic linking)
- This limited installation to conda environments

### Impact

- Users couldn't install on pip-only systems
- Package claimed to work without FAISS but actually didn't
- Confusion about when FAISS was needed

## Solution Implemented

### 1. Conditional Compilation in C++ (`cpp/src/rs_kmeans.cpp`)

**Before**:
```cpp
#include "rs_kmeans.hpp"
#include <faiss/IndexFlat.h>
#include <faiss/index_factory.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexLSH.h>
```

**After**:
```cpp
#include "rs_kmeans.hpp"

// FAISS is optional - only include if available
#ifdef HAS_FAISS
#include <faiss/IndexFlat.h>
#include <faiss/index_factory.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexLSH.h>
#endif
```

### 2. Conditional Index Initialization

Added `#ifdef HAS_FAISS` guards around FAISS-specific index types:

```cpp
void RSkMeans::initialize_index(const std::string& index_type, const std::string& params) {
    // FastLSH and GoogleLSH always work (no FAISS needed)
    if (index_type == "FastLSH") { /* ... */ return; }
    if (index_type == "GoogleLSH") { /* ... */ return; }

#ifdef HAS_FAISS
    // FAISS indices
    if (index_type == "Flat") { /* ... */ }
    else if (index_type == "LSH") { /* ... */ }
    else if (index_type == "IVFFlat") { /* ... */ }
    else if (index_type == "HNSW") { /* ... */ }
#else
    // FAISS not available - throw informative error
    throw std::runtime_error(
        "FAISS index type '" + index_type + "' requested but FAISS library is not available.\n"
        "RS-k-means++ requires FAISS for rejection sampling with approximate nearest neighbors.\n\n"
        "To use RS-k-means++ with FAISS indices (Flat, LSH, IVFFlat, HNSW):\n"
        "  1. Install FAISS: conda install -c pytorch faiss-cpu\n"
        "  2. Rebuild the package: pip install --force-reinstall --no-cache-dir kmeans-seeding\n\n"
        "Alternatively, you can use:\n"
        "  - FastLSH index (index_type='FastLSH') - works without FAISS\n"
        "  - GoogleLSH index (index_type='GoogleLSH') - works without FAISS\n"
        "  - Other algorithms: kmeanspp(), afkmc2(), multitree_lsh() - do not require FAISS"
    );
#endif
}
```

### 3. Header File Updates (`cpp/include/kmeans_seeding/rs_kmeans.hpp`)

Wrapped FAISS forward declarations and member variables:

```cpp
// Forward declare FAISS types (only if FAISS is available)
#ifdef HAS_FAISS
namespace faiss {
    struct Index;
}
#endif

// In class definition:
private:
    // Index for selected centers (incremental)
#ifdef HAS_FAISS
    std::unique_ptr<faiss::Index> center_index_;  // FAISS index (only if available)
#endif
    std::unique_ptr<fast_k_means::LSHDataStructure> google_lsh_index_;  // Always available
    std::unique_ptr<fast_k_means::FastLSH> fast_lsh_index_;  // Always available
```

### 4. Fallback Label Assignment

Added brute-force label assignment when FAISS not available:

```cpp
std::vector<int> RSkMeans::assign_labels() {
    std::vector<int> labels(n_);

#ifdef HAS_FAISS
    // Use FAISS for fast nearest neighbor search
    // [FAISS-based implementation]
#else
    // Brute-force label assignment when FAISS is not available
    #pragma omp parallel for
    for (int i = 0; i < n_; ++i) {
        float min_dist_sq = std::numeric_limits<float>::max();
        int best_label = 0;
        for (size_t j = 0; j < selected_center_indices_.size(); ++j) {
            float dist_sq = /* compute distance */;
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                best_label = static_cast<int>(j);
            }
        }
        labels[i] = best_label;
    }
#endif
    return labels;
}
```

### 5. Python API Documentation Updates

Updated docstring in `python/kmeans_seeding/initializers.py`:

```python
def rskmeans(..., index_type: str = 'LSH', ...):
    """
    ...
    index_type : {'Flat', 'LSH', 'IVFFlat', 'HNSW', 'FastLSH', 'GoogleLSH'}, default='LSH'
        Approximate nearest neighbor index type:
        - 'Flat': Exact search (slowest, most accurate) [requires FAISS]
        - 'LSH': FAISS LSH (fast, ~90-95% accuracy) [requires FAISS]
        - 'IVFFlat': Inverted file index (fast, ~99% accuracy) [requires FAISS]
        - 'HNSW': Hierarchical NSW (very fast, ~95-99% accuracy) [requires FAISS]
        - 'FastLSH': DHHash-based Fast LSH (very fast, ~90-95% accuracy) [works without FAISS]
        - 'GoogleLSH': Google's LSH implementation (fast, ~85-90% accuracy) [works without FAISS]

        Note: FAISS indices (Flat, LSH, IVFFlat, HNSW) are only available if FAISS is installed.
        Install with: conda install -c pytorch faiss-cpu
    ...
    """
```

## What Works Now

### ✅ Without FAISS

**Algorithms that work**:
- ✅ `kmeanspp()` - Standard k-means++ (always worked)
- ✅ `afkmc2()` - AFK-MC² MCMC sampling (always worked)
- ✅ `multitree_lsh()` (alias: `fast_lsh()`) - Google 2020 algorithm (always worked)
- ✅ `rskmeans()` with `index_type='FastLSH'` - NEW! Works without FAISS
- ✅ `rskmeans()` with `index_type='GoogleLSH'` - NEW! Works without FAISS

**What happens if user requests FAISS index**:
```python
from kmeans_seeding import rskmeans
import numpy as np

X = np.random.randn(1000, 50)

# This will raise a clear RuntimeError with installation instructions
centers = rskmeans(X, n_clusters=10, index_type='LSH')  # ← Requires FAISS
```

**Error message**:
```
RuntimeError: FAISS index type 'LSH' requested but FAISS library is not available.
RS-k-means++ requires FAISS for rejection sampling with approximate nearest neighbors.

To use RS-k-means++ with FAISS indices (Flat, LSH, IVFFlat, HNSW):
  1. Install FAISS: conda install -c pytorch faiss-cpu
  2. Rebuild the package: pip install --force-reinstall --no-cache-dir kmeans-seeding

Alternatively, you can use:
  - FastLSH index (index_type='FastLSH') - works without FAISS
  - GoogleLSH index (index_type='GoogleLSH') - works without FAISS
  - Other algorithms: kmeanspp(), afkmc2(), multitree_lsh() - do not require FAISS
```

### ✅ With FAISS

All features work as before:
- ✅ All algorithms work
- ✅ All index types available (Flat, LSH, IVFFlat, HNSW, FastLSH, GoogleLSH)
- ✅ Faster label assignment using FAISS
- ✅ Better performance for large datasets

## Build System Behavior

### CMake Detection

```cmake
# cpp/CMakeLists.txt
find_package(faiss QUIET)
if(faiss_FOUND)
    message(STATUS "FAISS found: ${faiss_VERSION}")
    add_compile_definitions(HAS_FAISS)  # ← Sets preprocessor flag
    set(FAISS_LIBRARIES faiss)
else()
    message(WARNING "FAISS not found. Some features will be disabled.")
    set(FAISS_LIBRARIES "")
endif()
```

### Compilation Modes

**Mode 1: With FAISS**
- CMake finds FAISS → sets `HAS_FAISS=1`
- Compiles all FAISS code
- Links against FAISS library
- All index types available at runtime

**Mode 2: Without FAISS**
- CMake doesn't find FAISS → `HAS_FAISS` not defined
- Skips FAISS `#include` statements
- Skips FAISS-dependent code sections
- Compiles successfully
- FAISS indices throw helpful error at runtime
- FastLSH/GoogleLSH indices work normally

## Testing

### Test 1: Build Without FAISS

```bash
# Create clean environment without FAISS
python3 -m venv test_no_faiss
source test_no_faiss/bin/activate
pip install numpy pybind11

# Build package
pip install .

# Result: ✅ Builds successfully
```

### Test 2: Runtime Without FAISS

```python
import numpy as np
from kmeans_seeding import kmeanspp, afkmc2, rskmeans

X = np.random.randn(100, 10)

# Works
centers = kmeanspp(X, n_clusters=5)  # ✅
centers = afkmc2(X, n_clusters=5)  # ✅
centers = rskmeans(X, n_clusters=5, index_type='FastLSH')  # ✅

# Fails with helpful error
centers = rskmeans(X, n_clusters=5, index_type='LSH')  # ❌ + clear message
```

## Benefits

### For Users

1. **Broader compatibility**: Works on pip-only systems
2. **Smaller dependency footprint**: Don't need FAISS if not using it
3. **Clear error messages**: Know exactly what to install and why
4. **Gradual adoption**: Start without FAISS, add it later if needed

### For Developers

1. **True optional dependency**: Matches what we claim in docs
2. **Cleaner architecture**: Clear separation of FAISS/non-FAISS code
3. **Better testing**: Can test both configurations
4. **Future-proof**: Easy to add more non-FAISS index types

### For Maintainers

1. **Easier PyPI publishing**: Works for more users out of the box
2. **Better user experience**: Fewer "why doesn't this work?" issues
3. **More flexible deployment**: Can build wheels without FAISS dependency

## Migration Guide

### For Existing Users

**No changes needed** if:
- You already have FAISS installed
- You're using default settings
- Everything works for you

**Optional improvements**:
- Try `index_type='FastLSH'` for FAISS-free operation
- Remove FAISS if you don't need it

### For New Users

**Lightweight install** (no FAISS):
```bash
pip install kmeans-seeding

# Use these algorithms:
from kmeans_seeding import kmeanspp, afkmc2, multitree_lsh
# Or rskmeans with FastLSH/GoogleLSH
from kmeans_seeding import rskmeans
centers = rskmeans(X, n_clusters=k, index_type='FastLSH')
```

**Full install** (with FAISS):
```bash
conda install -c pytorch faiss-cpu
pip install kmeans-seeding

# All features available, including FAISS indices
```

## Files Modified

1. ✅ `cpp/src/rs_kmeans.cpp` - Added `#ifdef HAS_FAISS` guards
2. ✅ `cpp/include/kmeans_seeding/rs_kmeans.hpp` - Conditional forward declarations
3. ✅ `python/kmeans_seeding/initializers.py` - Updated docstrings
4. ✅ `CLAUDE.md` - Updated documentation
5. ✅ Created `FAISS_OPTIONAL_IMPLEMENTATION.md` - This file

## Conclusion

FAISS is now a truly optional dependency. The package:
- ✅ Builds without FAISS
- ✅ Works without FAISS (with fallbacks)
- ✅ Provides helpful errors when FAISS features are requested but unavailable
- ✅ Maintains full functionality when FAISS is installed

This improves accessibility while maintaining all existing functionality for users who have FAISS.
