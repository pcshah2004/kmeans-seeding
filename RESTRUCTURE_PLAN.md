# Restructuring Plan for kmeans-seeding Library

## Current Structure
```
Fast k means++/
├── rs_kmeans/                  # Modern C++ implementation
├── fast_k_means_2020/          # Legacy 2020 implementation
├── quantization_analysis/      # Research code
└── main.tex                    # Theory paper
```

## Target Structure
```
kmeans-seeding/
├── src/kmeans_seeding/         # Main Python package
├── cpp/                        # C++ implementations
├── docs/                       # Sphinx documentation
├── tests/                      # Test suite
├── examples/                   # Example scripts & datasets
├── benchmarks/                 # Benchmark scripts
├── research/                   # Research code (quantization analysis)
└── paper/                      # LaTeX paper
```

## Migration Steps
1. Create new src/kmeans_seeding/ package structure
2. Move C++ code to cpp/ directory
3. Create Python wrappers for seeding functions
4. Set up documentation
5. Reorganize tests
6. Update build configuration

## Files to Keep
- All C++ implementations (rs_kmeans, fast_k_means_2020)
- Python analysis code (quantization_analysis)
- LaTeX paper (main.tex)
- All benchmark scripts

## Files to Create
- Modern pyproject.toml
- GitHub Actions workflows
- Sphinx documentation
- Comprehensive tests
- Examples with small datasets
