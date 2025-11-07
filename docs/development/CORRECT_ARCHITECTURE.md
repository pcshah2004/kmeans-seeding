# kmeans-seeding Library Architecture

## Core Principle
**C++ First**: All algorithms implemented in C++, Python bindings added on top.

## Revised Structure

```
kmeans-seeding/
├── cpp/                                    # Core C++ implementations
│   ├── include/
│   │   └── kmeans_seeding/
│   │       ├── common.hpp                  # Common utilities
│   │       ├── kmeanspp.hpp                # Standard k-means++
│   │       ├── rs_kmeans.hpp               # RS-k-means++ (rejection sampling)
│   │       ├── afkmc2.hpp                  # AFK-MC²
│   │       ├── fast_lsh.hpp                # Fast-LSH (Google 2020)
│   │       └── lsh.hpp                     # LSH data structure
│   ├── src/
│   │   ├── kmeanspp.cpp
│   │   ├── rs_kmeans.cpp
│   │   ├── afkmc2.cpp
│   │   ├── fast_lsh.cpp
│   │   ├── lsh.cpp
│   │   └── python_bindings.cpp             # pybind11 bindings
│   ├── CMakeLists.txt                      # Main build config
│   └── tests/
│       └── test_cpp.cpp                    # C++ unit tests
│
├── python/                                 # Python wrapper layer
│   └── kmeans_seeding/
│       ├── __init__.py                     # Main API exports
│       ├── _core.pyi                       # Type stubs for C++ extension
│       ├── initializers.py                 # User-facing seeding functions
│       ├── utils.py                        # Python utilities
│       └── datasets.py                     # Example datasets
│
├── tests/                                  # Python tests
│   ├── test_kmeanspp.py
│   ├── test_rs_kmeans.py
│   ├── test_afkmc2.py
│   ├── test_fast_lsh.py
│   └── test_sklearn_compatibility.py
│
├── benchmarks/                             # Benchmark suite
│   ├── compare_methods.py
│   ├── scaling_analysis.py
│   └── datasets/                           # Small example datasets
│
├── examples/                               # Usage examples
│   ├── basic_usage.py
│   ├── comparison.py
│   └── advanced.py
│
├── docs/                                   # Sphinx documentation
│   ├── source/
│   │   ├── conf.py
│   │   ├── index.rst
│   │   ├── installation.rst
│   │   ├── api.rst
│   │   ├── algorithms.rst                  # Algorithm explanations
│   │   └── benchmarks.rst
│   └── requirements.txt
│
├── research/                               # Research code (kept separate)
│   └── quantization_analysis/
│
├── paper/                                  # LaTeX paper
│   ├── main.tex
│   ├── prefix.sty
│   └── refs.bib
│
├── .github/
│   └── workflows/
│       ├── build-wheels.yml                # Multi-platform wheel building
│       ├── tests.yml                       # CI tests
│       ├── docs.yml                        # Doc building
│       └── publish.yml                     # PyPI publishing
│
├── pyproject.toml                          # Modern Python packaging
├── setup.py                                # Build configuration
├── CMakeLists.txt                          # Top-level CMake
├── README.md
├── LICENSE                                 # MIT License
├── CITATION.cff                            # Academic citation
└── MANIFEST.in                             # Package data inclusion

```

## Data Flow

1. **C++ Core**: All algorithms implemented in `cpp/src/*.cpp`
2. **Python Bindings**: `cpp/src/python_bindings.cpp` exposes C++ to Python as `_core` module
3. **Python Wrapper**: `python/kmeans_seeding/initializers.py` provides clean API
4. **User**: Imports and uses functions from `kmeans_seeding`

## Example API Design

### C++ Layer (cpp/src/rs_kmeans.cpp)
```cpp
namespace kmeans_seeding {
    std::vector<int> rs_kmeans_init(
        const float* data, int n, int d, int k, 
        int max_iter, const std::string& index_type
    );
}
```

### Python Bindings (cpp/src/python_bindings.cpp)
```cpp
PYBIND11_MODULE(_core, m) {
    m.def("rs_kmeans_init", &rs_kmeans_init,
          "RS-k-means++ initialization");
}
```

### Python Wrapper (python/kmeans_seeding/initializers.py)
```python
from . import _core
import numpy as np

def rejection_sampling(X, n_clusters, max_iter=50, index_type='LSH', random_state=None):
    """
    RS-k-means++ initialization (rejection sampling).
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data
    n_clusters : int
        Number of clusters
    ...
    
    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        Initial cluster centers
    """
    X = np.asarray(X, dtype=np.float32, order='C')
    center_indices = _core.rs_kmeans_init(X, n_clusters, max_iter, index_type)
    return X[center_indices]
```

### User Code
```python
from kmeans_seeding import rejection_sampling
from sklearn.cluster import KMeans

# Get initial centers using RS-k-means++
centers = rejection_sampling(X, n_clusters=10, index_type='LSH')

# Use with sklearn
kmeans = KMeans(n_clusters=10, init=centers, n_init=1)
kmeans.fit(X)
```

## Build Process

1. **CMake** builds C++ core + Python bindings → `_core.so` (extension module)
2. **setuptools** packages Python wrapper + C++ extension
3. **cibuildwheel** creates wheels for Linux/macOS/Windows
4. **twine** uploads to PyPI

