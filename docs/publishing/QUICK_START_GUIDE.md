# 🚀 Quick Start Guide - kmeans-seeding Library

## What We Built

A professional, pip-installable Python library for fast k-means++ initialization with:
- **C++ core** (fast, optimized algorithms)
- **Python bindings** (easy to use, sklearn-compatible)
- **Multi-platform wheels** (Linux, macOS, Windows)
- **CI/CD pipelines** (automated testing & publishing)
- **Read the Docs** integration (professional documentation)

## Current Status

**✅ 90% Complete** - Core functionality ready, needs docs & tests

## Test It Right Now

```bash
# Navigate to project
cd "/Users/poojanshah/Desktop/Fast k means++"

# Install dependencies
pip install numpy pybind11 cmake

# Install in development mode
pip install -e .

# Test import
python -c "import kmeans_seeding; print(f'Version: {kmeans_seeding.__version__}')"

# Run a quick test
python << 'PYTHON'
import numpy as np
from kmeans_seeding import rejection_sampling

# Generate data
X = np.random.randn(1000, 20).astype(np.float32)

# Get centers
centers = rejection_sampling(X, n_clusters=10, index_type='Flat')

print(f"Success! Centers shape: {centers.shape}")
print(f"Expected: (10, 20)")
assert centers.shape == (10, 20), "Shape mismatch!"
print("✅ All tests passed!")
PYTHON
```

## File Structure Created

```
kmeans-seeding/
├── 📦 Core Package
│   ├── cpp/                      # C++ implementation
│   │   ├── include/kmeans_seeding/
│   │   ├── src/
│   │   └── CMakeLists.txt
│   └── python/kmeans_seeding/    # Python wrapper
│       ├── __init__.py
│       └── initializers.py
│
├── 🔧 Build Configuration
│   ├── pyproject.toml            # Modern packaging
│   ├── setup.py                  # CMake integration
│   └── CMakeLists.txt            # Top-level build
│
├── 📚 Documentation
│   ├── README.md                 # Main docs
│   ├── LICENSE                   # MIT
│   ├── CITATION.cff              # Academic citation
│   ├── SETUP_SUMMARY.md          # Setup guide
│   └── PROJECT_STATUS.md         # Progress tracker
│
├── 🤖 CI/CD
│   └── .github/workflows/
│       ├── build-wheels.yml      # Multi-platform builds
│       ├── tests.yml             # Testing
│       └── docs.yml              # Documentation
│
├── 📁 Directories (to populate)
│   ├── tests/                    # Unit tests
│   ├── examples/                 # Usage examples
│   ├── benchmarks/               # Performance tests
│   └── docs/                     # Sphinx docs
│
└── 🗂️ Preserved
    ├── rs_kmeans/                # Original 2025 implementation
    ├── fast_k_means_2020/        # Legacy 2020 implementation
    ├── quantization_analysis/    # Research code
    └── paper/                    # LaTeX paper
```

## Available API

```python
from kmeans_seeding import (
    kmeanspp,                     # Standard k-means++
    rejection_sampling,           # RS-k-means++ (main contribution)
    afkmc2,                       # AFK-MC² (MCMC-based)
    fast_lsh,                     # Fast-LSH (Google 2020)
    rejection_sampling_lsh_2020,  # Rejection + LSH (Google 2020)
)

# All functions have the same interface:
centers = rejection_sampling(
    X,                    # numpy array (n_samples, n_features)
    n_clusters=10,        # number of clusters
    random_state=42       # for reproducibility
)

# Use with sklearn
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10, init=centers, n_init=1)
kmeans.fit(X)
```

## Next Steps (Priority Order)

### 1. **Test the Build** (5 minutes)
```bash
# Clean install test
pip install -e .
python -c "from kmeans_seeding import rejection_sampling; print('✅ Success!')"
```

### 2. **Write Basic Tests** (2-3 hours)
```python
# tests/test_basic.py
import numpy as np
import pytest
from kmeans_seeding import rejection_sampling, afkmc2

def test_rejection_sampling():
    X = np.random.randn(100, 10).astype(np.float32)
    centers = rejection_sampling(X, n_clusters=5)

    assert centers.shape == (5, 10)
    assert centers.dtype == np.float32
    assert not np.any(np.isnan(centers))

def test_reproducibility():
    X = np.random.randn(100, 10).astype(np.float32)

    centers1 = rejection_sampling(X, n_clusters=5, random_state=42)
    centers2 = rejection_sampling(X, n_clusters=5, random_state=42)

    np.testing.assert_array_equal(centers1, centers2)

# Run: pytest tests/test_basic.py -v
```

### 3. **Set Up Sphinx Docs** (1-2 hours)
```bash
cd docs
pip install sphinx sphinx-rtd-theme

# Quick start
sphinx-quickstart \
    --project="kmeans-seeding" \
    --author="Poojan Shah" \
    --release="0.1.0" \
    --language="en"

# Build docs
make html
open build/html/index.html
```

### 4. **Add Examples** (1 hour)
```python
# examples/basic_usage.py
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from kmeans_seeding import rejection_sampling
import matplotlib.pyplot as plt

# Generate data
X, y_true = make_blobs(n_samples=1000, centers=5, random_state=42)

# Get initial centers
centers = rejection_sampling(X, n_clusters=5)

# Cluster
kmeans = KMeans(n_clusters=5, init=centers, n_init=1)
y_pred = kmeans.fit_predict(X)

# Visualize
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X')
plt.title('RS-k-means++ Initialization')
plt.savefig('results.png')
print('✅ Saved to results.png')
```

### 5. **Run Benchmarks** (2-3 hours)
```python
# benchmarks/compare_methods.py
import numpy as np
import time
from sklearn.cluster import KMeans
from kmeans_seeding import (
    kmeanspp, rejection_sampling, afkmc2
)

X = np.random.randn(10000, 50).astype(np.float32)

methods = {
    'k-means++': kmeanspp,
    'RS-k-means++': rejection_sampling,
    'AFK-MC²': afkmc2,
}

for name, method in methods.items():
    start = time.time()
    centers = method(X, n_clusters=100)
    duration = time.time() - start

    # Measure final cost
    kmeans = KMeans(n_clusters=100, init=centers, n_init=1, max_iter=10)
    kmeans.fit(X)
    cost = kmeans.inertia_

    print(f"{name:15s} | Time: {duration:6.3f}s | Cost: {cost:.2e}")
```

## Common Issues & Solutions

### Issue 1: CMake not found
```bash
# macOS
brew install cmake

# Ubuntu/Debian
sudo apt-get install cmake

# Conda
conda install cmake
```

### Issue 2: pybind11 not found
```bash
pip install pybind11
```

### Issue 3: FAISS not found (optional)
```bash
# CPU version
conda install -c pytorch faiss-cpu

# Library works without FAISS but slower
```

### Issue 4: Compilation errors
```bash
# Check compiler version
g++ --version  # Need 7+ for C++17

# Try clean build
pip install -e . --force-reinstall --no-cache-dir
```

## Publishing Checklist

Before publishing to PyPI:

- [ ] Update version numbers
- [ ] Update GitHub URLs (replace 'yourusername')
- [ ] Write CHANGELOG.md
- [ ] Complete documentation
- [ ] Write comprehensive tests (>80% coverage)
- [ ] Test on all platforms locally
- [ ] Upload to Test PyPI first
- [ ] Create GitHub release
- [ ] Monitor CI/CD pipelines

## Resources

- **Setup Summary**: `SETUP_SUMMARY.md`
- **Project Status**: `PROJECT_STATUS.md`
- **Architecture**: `CORRECT_ARCHITECTURE.md`
- **Python Packaging**: https://packaging.python.org/
- **Sphinx Docs**: https://www.sphinx-doc.org/
- **Read the Docs**: https://readthedocs.org/

## Questions?

1. Check `SETUP_SUMMARY.md` for detailed instructions
2. Review `PROJECT_STATUS.md` for current progress
3. See GitHub Issues for known problems
4. Contact: cs1221594@cse.iitd.ac.in

---

**Ready to go!** The foundation is solid. Just need tests, docs, and examples. 🎉
