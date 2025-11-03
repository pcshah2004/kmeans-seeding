# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

**Common Commands:**

```bash
# LaTeX paper compilation
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

# Python environment setup
cd quantization_analysis
source venv/bin/activate  # Activate virtual environment
pip install -r requirements.txt

# Run quantization experiments
python3 run_local_experiments.py      # Local datasets
python3 run_experiments.py            # UCI ML datasets
python3 huggingface_analysis.py       # Text datasets
python3 image_analysis.py             # Image datasets

# Build and install rs_kmeans (C++ implementation)
cd rs_kmeans
conda install -c pytorch faiss-cpu    # Install FAISS dependency
pip install .                          # Install Python package

# Run rs_kmeans benchmarks
cd rs_kmeans
python3 benchmark_comparison.py       # Compare with standard k-means++
python3 benchmark_multi_dataset.py    # Test on multiple datasets

# Compile legacy C++ implementation (2020)
cd fast_k_means_2020
g++ -std=c++11 -O3 -o fast_kmeans *.cc
```

## Project Overview

This repository contains research on **Fast k-means++ with Locality-Sensitive Hashing (LSH)** and **Rejection Sampling k-means++**. The project consists of:

1. **Theoretical analysis** (LaTeX paper in `main.tex`)
   - (ε,δ)-k-means++ approximation guarantees
   - Main result: Expected cost bound with LSH approximation and perturbed sampling

2. **Empirical quantization analysis** (Python code in `quantization_analysis/`)
   - Measures how k-means cost scales with number of clusters
   - Supports UCI ML, HuggingFace text/image datasets, and local files

3. **C++ Implementations**:
   - **rs_kmeans/** (2025): Modern FAISS-based rejection sampling with Python bindings
   - **fast_k_means_2020/** (2020): Legacy implementation of fast k-means++

The main theoretical contribution is an approximation guarantee for k-means++ when using:
- An LSH data structure for approximate nearest neighbor queries (with parameter ε)
- A perturbed D² sampling distribution (with parameter δ)

## Repository Structure

```
.
├── main.tex                          # Main LaTeX paper with algorithm analysis
├── prefix.sty                        # LaTeX style/package definitions
├── refs.bib                          # Bibliography
├── rs_kmeans/                        # C++ implementation with Python bindings (NEW, 2025)
├── fast_k_means_2020/                # Legacy C++ implementation (2020 paper)
└── quantization_analysis/            # Python experimental code
    ├── Core Analysis Modules
    │   ├── kmeans_analysis.py        # Quantization dimension computation (core algorithm)
    │   └── visualization.py          # Plotting and visualization
    │
    ├── Data Source Modules
    │   ├── dataset_fetcher.py        # UCI ML Repository datasets
    │   ├── local_dataset_loader.py   # Local CSV/Excel/Parquet files
    │   ├── huggingface_analysis.py   # HuggingFace text datasets (via sentence-transformers)
    │   └── image_analysis.py         # HuggingFace image datasets (via CLIP embeddings)
    │
    ├── Experiment Runners
    │   ├── run_experiments.py        # Run UCI ML experiments
    │   ├── run_experiments_custom.py # Run specific UCI datasets by ID
    │   └── run_local_experiments.py  # Run experiments on local datasets
    │
    ├── Utility Scripts
    │   ├── generate_all_embeddings.py      # Generate embeddings for all datasets
    │   ├── generate_image_embeddings.py    # Generate only image embeddings (CLIP)
    │   └── plot_eps_q_histogram.py         # Create histogram of quantization dimensions
    │
    ├── Test Scripts
    │   ├── test_huggingface.py       # Test HuggingFace dataset loading
    │   └── test_image_analysis.py    # Test image embedding generation
    │
    ├── Documentation
    │   ├── USAGE.md                  # Comprehensive usage guide
    │   └── LOCAL_DATASETS_GUIDE.md   # Guide for analyzing local datasets
    │
    ├── Output Directories
    │   ├── plots/                    # UCI ML plots
    │   ├── plots_huggingface/        # HuggingFace text plots
    │   ├── plots_images/             # Image analysis plots
    │   ├── results/                  # UCI ML results CSVs
    │   ├── results_huggingface/      # HuggingFace results CSVs
    │   └── results_images/           # Image results CSVs
    │
    └── datasets/                     # Place local datasets here (CSV, Excel, Parquet)
```

## Working with the LaTeX Paper

### Building the Paper

```bash
# Compile the LaTeX document
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# View the output
open main.pdf  # macOS
```

### Key Sections in main.tex

- **Theorem 1 (Main Result)**: Approximation guarantee for (ε,δ)-k-means++
- **Section 2**: LSH data structure properties (Lemma 1-2)
- **Section 3**: Analysis of the algorithm
  - Algorithm 1: The (ε,δ)-k-means++ procedure
  - Lemmas on covered/uncovered clusters and potential function analysis
- **Mathematical notation** defined in Section 1

### LaTeX Dependencies

The paper uses `prefix.sty` which imports:
- `algorithm2e` for pseudocode
- `amsthm`, `thmtools` for theorem environments
- `cleveref` for cross-references
- `tikz` for diagrams
- `natbib` for bibliography

## Working with the Python Code

### Setup

```bash
cd quantization_analysis

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments

The codebase supports **four different data sources**. See `USAGE.md` for detailed instructions.

**1. Local Datasets (Easiest - Recommended for custom data)**
```bash
# Place CSV/Excel/Parquet files in datasets/ folder, then:
python3 run_local_experiments.py
```

**2. UCI ML Repository Datasets**
```bash
# Run on default curated list:
python3 run_experiments.py

# Or specify dataset IDs in run_experiments_custom.py:
python3 run_experiments_custom.py
```

**3. HuggingFace Text Datasets**
```bash
# Analyzes text datasets using sentence-transformers embeddings:
python3 huggingface_analysis.py
```

**4. HuggingFace Image Datasets**
```bash
# Analyzes image datasets using CLIP embeddings:
python3 image_analysis.py
```

### Python Module Architecture

**Core Analysis Modules:**

1. **`kmeans_analysis.py`**: Quantization dimension computation (core algorithm)
   - `compute_kmeans_cost(X, k)`: Run k-means++ and return inertia
   - `estimate_quantization_dimension(X, k_min, k_max, n_init)`: Main analysis function
   - Fits log-log regression: log(cost_k / cost_1) = a * log(k) + b
   - Quantization dimension = 2/a

2. **`visualization.py`**: Creates plots
   - `plot_loglog_analysis(...)`: Individual dataset log-log plot with residuals
   - `create_summary_plot(results_df)`: Aggregate analysis across datasets

**Data Source Modules:**

3. **`dataset_fetcher.py`**: UCI ML Repository datasets
   - `fetch_dataset(dataset_id)`: Get single dataset by ID
   - `get_all_numerical_datasets(max_datasets)`: Fetch multiple datasets
   - Automatically filters for numerical features and handles data cleaning

4. **`local_dataset_loader.py`**: Local file datasets
   - `load_local_dataset(file_path, name)`: Load CSV/Excel/Parquet/Feather
   - `load_all_local_datasets(dir_path)`: Load all files from directory
   - Automatically removes categorical columns and keeps only numerical features

5. **`huggingface_analysis.py`**: HuggingFace text datasets
   - Loads text datasets and converts to embeddings using sentence-transformers
   - Supports 100+ pre-configured text datasets
   - Uses `all-MiniLM-L6-v2` model by default

6. **`image_analysis.py`**: HuggingFace image datasets
   - Loads image datasets and converts to embeddings using CLIP
   - Supports datasets like ImageNette, CIFAR, Fashion-MNIST, etc.
   - Uses `openai/clip-vit-base-patch32` model

**Utility Scripts:**

7. **`generate_all_embeddings.py`**: Pre-generate embeddings for batch processing
   - Generates both text and image embeddings
   - Saves embeddings to `embeddings/text/` and `embeddings/image/` directories
   - Each embedding saved with metadata JSON file
   - Automatically skips already-processed datasets
   - **Note**: Image embedding generation is CPU-intensive (~1.5-2 hours for all datasets)

8. **`generate_image_embeddings.py`**: Generate only image embeddings
   - Standalone script for CLIP image embeddings
   - Processes 6 image datasets: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, Beans, Cats vs Dogs
   - Uses batch processing (32 images per batch)
   - Checkpointing: saves each dataset after completion

9. **`plot_eps_q_histogram.py`**: Visualization of quantization dimension distribution
   - Creates histogram of ε_q (quantization dimension) values
   - Computes and displays mean, median, and standard deviation
   - Useful for aggregate analysis across multiple datasets

### Key Algorithms

**Quantization Dimension Estimation:**
- For k ∈ {1, 2, ..., k_max}, compute k-means++ cost
- Fit linear regression on log-log scale: log(cost_k/cost_1) vs log(k)
- Slope 'a' relates to intrinsic dimension; quantization dimension = 2/a
- This measures how the clustering cost scales with the number of clusters

## Key Research Concepts

### The (ε,δ)-k-means++ Algorithm

The algorithm modifies standard k-means++ in two ways:

1. **LSH approximation (ε)**: Instead of computing exact nearest centers, uses LSH data structure that returns (1/√ε)-approximate nearest neighbors. This speeds up the D² sampling.

2. **Perturbed sampling (δ)**: The sampling distribution is:
   ```
   π_{ε,δ}(x|S) = (1-δ) · π_L(x|S) + δ · (1/n)
   ```
   where π_L is the D² distribution computed via LSH queries, and δ adds uniform randomness.

### Main Theoretical Result

The expected k-means cost satisfies:
```
E[Δ(X,S)] ≤ O(ε^{-3} log k) · Δ_k(X) + (δ/(1-δ)) · O(k + ε^{-3} log k) · Δ_1(X)
```

This shows the approximation degrades gracefully with ε (LSH approximation) and δ (sampling noise).

### Analysis Techniques

- **Potential function**: Ψ_t = (W_t / |U_t|) · Δ_L^t(U_t), tracks cost of uncovered clusters
- **Covered vs uncovered clusters**: Clusters are "covered" once they contain a sampled center
- **Telescoping sum**: Final cost = E[Δ^k(H_k)] + Σ E[Ψ_{t+1} - Ψ_t]

### Usage Examples

**Quick Start - Analyze Your Own Data:**
```python
from local_dataset_loader import load_local_dataset
from kmeans_analysis import estimate_quantization_dimension
from visualization import plot_loglog_analysis

# Load your dataset (CSV, Excel, Parquet, or Feather)
X, metadata = load_local_dataset("datasets/my_data.csv")

# Run quantization analysis
results = estimate_quantization_dimension(X, k_min=1, k_max=50, n_init=10)

# Display results
print(f"Quantization Dimension: {results['quantization_dimension']:.3f}")
print(f"R² value: {results['r_squared']:.4f}")

# Create visualization
plot_loglog_analysis(
    results['k_values'], results['costs'],
    results['a'], results['b'], results['r_squared'],
    results['quantization_dimension'],
    metadata['name'], metadata['n_samples'], metadata['n_features'],
    save_path=f"plots/{metadata['name']}.png", show=True
)
```

**Programmatic UCI ML Analysis:**
```python
from dataset_fetcher import fetch_dataset
from kmeans_analysis import estimate_quantization_dimension

# Fetch Iris dataset (ID 53)
X, metadata = fetch_dataset(dataset_id=53)
results = estimate_quantization_dimension(X, k_max=50)
print(f"Dataset: {metadata['name']}")
print(f"Q-dim: {results['quantization_dimension']:.3f}")
```

### Output Structure

All experiment runners save results in the following format:

**Plots:** Dual-panel figures with:
- Left: Log-log plot of cost_1/cost_k vs k with fitted power law
- Right: Residual plot showing model fit quality

**Results CSV:** Contains columns:
- `dataset`: Dataset name
- `n`: Number of samples
- `d`: Number of features (dimensionality)
- `quantization_dimension`: Computed quantization dimension (2/a)
- `a`: Slope of log-log regression
- `b`: Intercept of log-log regression
- `r_squared`: R² goodness of fit

## Working with the C++ Implementations

This repository contains **two C++ implementations** of fast k-means algorithms.

### RS-k-means++ (rs_kmeans/) - RECOMMENDED

**Modern FAISS-based rejection sampling implementation** (2025 paper).

**Features:**
- FAISS-powered approximate nearest neighbors (Flat, LSH, IVF, HNSW indices)
- Python bindings via pybind11 (NumPy-compatible)
- Scikit-learn compatible API
- Fast: O(nnz(X)) preprocessing + O(mk²d) clustering time

**Build and Install:**
```bash
cd rs_kmeans

# Prerequisites (install FAISS first)
conda install -c pytorch faiss-cpu  # or faiss-gpu

# Build using CMake
mkdir build && cd build
cmake ..
make

# Or install Python package
cd ..
pip install .  # or pip install -e . for development
```

**Python Usage:**
```python
import numpy as np
from rs_kmeans import RSkMeans, RSkMeansEstimator

# Low-level API
X = np.random.randn(10000, 50).astype(np.float32)
model = RSkMeans()
model.preprocess(X)
centers, labels = model.cluster(k=100, m=50, index_type="LSH")

# Scikit-learn API
kmeans = RSkMeansEstimator(n_clusters=100, index_type="LSH")
labels = kmeans.fit_predict(X)
```

**Benchmarking:**
```bash
cd rs_kmeans

# Compare different FAISS index types
python3 benchmark_index_comparison.py

# Compare with standard k-means++
python3 benchmark_comparison.py

# Test on multiple datasets
python3 benchmark_multi_dataset.py

# Analyze rejection efficiency
python3 benchmark_rejection_efficiency.py
```

**Index Types:**
- `Flat`: Exact search, slow but accurate (k < 1000)
- `LSH`: Fast, ~90-95% accuracy, matches theory
- `IVFFlat`: Fast, ~99% accuracy (1000 < k < 100K)
- `HNSW`: Very fast, ~95-99% accuracy (k > 10K)

**Key Files:**
- `src/rs_kmeans.cpp`: Core C++ implementation
- `src/bindings.cpp`: Python bindings
- `include/rs_kmeans/rs_kmeans.hpp`: C++ header
- `test_rs_kmeans.py`: Unit tests
- `CMakeLists.txt`: Build configuration

### Fast k-means 2020 (fast_k_means_2020/) - LEGACY

**Original C++ implementation** from "Fast and Accurate k-means++ via Rejection Sampling" (2020).

**Compilation:**
```bash
cd fast_k_means_2020

# Compile all files together (no external dependencies)
g++ -std=c++11 -O3 -o fast_kmeans \
    fast_k_means_main.cc \
    compute_cost.cc \
    fast_k_means_algo.cc \
    kmeanspp_seeding.cc \
    lsh.cc \
    multi_tree_clustering.cc \
    preprocess_input_points.cc \
    random_handler.cc \
    rejection_sampling_lsh.cc \
    single_tree_clustering.cc \
    tree_embedding.cc

# Run
./fast_kmeans < input_data.txt
```

**Input Format:**
```
<n_points> <n_dimensions>
<x1_1> <x1_2> ... <x1_d>
<x2_1> <x2_2> ... <x2_d>
...
```

**Example:**
```
3 2
1.00 2.50
3.30 4.12
0.0 -10.0
```

**Key Files:**
- `fast_k_means_main.cc`: Main entry point
- `kmeanspp_seeding.{cc,h}`: Standard k-means++ seeding
- `rejection_sampling_lsh.{cc,h}`: Fast k-means++ via rejection sampling
- `lsh.{cc,h}`: LSH data structure implementation
- `tree_embedding.{cc,h}`: Tree embedding for clustering

**Note**: This is a legacy implementation. For new projects, use `rs_kmeans/` instead.

## Development Notes

### Architecture Overview

The codebase follows a modular design:
1. **Core algorithms** (`kmeans_analysis.py`, `visualization.py`) are data-source agnostic
2. **Data loaders** (`dataset_fetcher.py`, `local_dataset_loader.py`, etc.) provide a consistent interface: all return `(X, metadata)` tuples where `X` is a numpy array or pandas DataFrame with only numerical features
3. **Experiment runners** orchestrate data loading, analysis, and output generation

### Important Implementation Details

- **Automatic preprocessing**: All data loaders automatically:
  - Remove categorical/non-numerical columns
  - Drop rows with missing values
  - Return only clean numerical feature matrices

- **Virtual environment**: `quantization_analysis/venv/` exists but may need dependency reinstallation if packages are updated

- **Memory considerations**:
  - Image and text embeddings can be memory-intensive
  - Adjust `max_samples` parameters in dataset configs for large datasets
  - Consider using `n_init=5` instead of default `10` for faster k-means

- **Three implementations exist**:
  1. **Python (quantization_analysis/)**: Empirical analysis of quantization dimensions
  2. **C++ (rs_kmeans/)**: Modern FAISS-based rejection sampling implementation (2025)
  3. **C++ (fast_k_means_2020/)**: Legacy implementation from 2020 paper

- **LaTeX paper**: The theoretical paper in `main.tex` is the primary research artifact; Python code provides empirical validation

- **Notation consistency**: When editing theorems/lemmas, maintain consistency with mathematical notation defined in Section 1 of main.tex

### Common Workflows

**Adding a new data source:**
1. Create a new loader module (e.g., `new_source_loader.py`)
2. Implement functions that return `(X, metadata)` tuples
3. Ensure X contains only numerical features
4. Create a corresponding experiment runner script

**Modifying analysis parameters:**
- Edit `estimate_quantization_dimension()` call parameters in experiment runners
- Key parameters: `k_min`, `k_max`, `n_init`
- Lower `n_init` for speed, raise for accuracy

**Debugging dataset loading:**
- Check `metadata` dict for preprocessing info (rows/columns removed)
- Use `get_dataset_info()` functions to inspect before full loading
- Enable logging: `logging.basicConfig(level=logging.INFO)`

**Working directory context:**
- All Python experiment scripts should be run from `quantization_analysis/` directory
- Virtual environment must be activated before running scripts
- LaTeX compilation should be run from root directory

**Performance optimization:**
- For faster k-means: Install `faiss-cpu` or `faiss-gpu` (already in requirements.txt)
- For large datasets: Use `method='minibatch'` in `compute_kmeans_cost()`
- For HuggingFace datasets: Pre-generate embeddings using utility scripts to avoid re-computation

**Working with rs_kmeans C++ code:**
- All C++ source files are in `rs_kmeans/src/` and headers in `rs_kmeans/include/`
- After modifying C++, rebuild with: `cd rs_kmeans && pip install -e .`
- Run unit tests: `python3 rs_kmeans/test_rs_kmeans.py`
- Benchmarking scripts are in `rs_kmeans/` directory
- CMake build files are generated in `rs_kmeans/build/`

**Testing different FAISS indices in rs_kmeans:**
```python
from rs_kmeans import RSkMeansEstimator

# Try different index types to optimize speed/accuracy tradeoff
for index_type in ["Flat", "LSH", "IVFFlat", "HNSW"]:
    kmeans = RSkMeansEstimator(n_clusters=k, index_type=index_type)
    labels = kmeans.fit_predict(X)
```
