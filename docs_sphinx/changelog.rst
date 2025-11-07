Changelog
=========

Version 0.2.2 (November 2025)
------------------------------

**FastLSH Optimizations**

- Fixed critical hash collision bug when k > d_padded
- 20-40% faster query performance
- Optimized memory allocations with thread-local buffers
- Up to 13× faster top-k candidate selection
- Added comprehensive stress tests

**Bug Fixes**

- Fixed systematic sampling bug in fast_lsh.cpp (lines 147-151)
- Proper handling of edge case where k > d_padded

**Documentation**

- Created comprehensive Read the Docs documentation
- Added algorithm comparison guide
- Detailed parameter tuning guides

Version 0.2.1 (2025)
--------------------

**Improvements**

- Added backwards compatibility alias for rejection_sampling
- Updated package metadata

Version 0.2.0 (2025)
--------------------

**New Features**

- Renamed algorithms for clarity:
  - rejection_sampling → rskmeans
  - fast_lsh → multitree_lsh
- Maintained backwards compatibility with old names
- Comprehensive experiments and benchmarks

**Build System**

- Improved CMake configuration
- Better FAISS detection
- Enhanced wheel building

Version 0.1.0 (2024)
--------------------

**Initial Release**

- Standard k-means++ (kmeanspp)
- RS-k-means++ (rejection_sampling)
- AFK-MC² (afkmc2)
- Fast-LSH k-means++ (fast_lsh)
- Python bindings via pybind11
- FAISS integration (optional)
- OpenMP support for parallelization

Future Plans
------------

**Planned for 0.3.0**

- GPU support via CUDA/FAISS-GPU
- Additional index types
- Better parameter auto-tuning
- Distributed clustering support
