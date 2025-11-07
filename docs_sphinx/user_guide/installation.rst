Installation
============

Basic Installation
------------------

The simplest way to install kmeans-seeding is via pip:

.. code-block:: bash

   pip install kmeans-seeding

This will install the package with pre-compiled wheels for most platforms (Linux, macOS, Windows).

**Supported Python versions**: 3.8, 3.9, 3.10, 3.11, 3.12, 3.13

Requirements
------------

**Mandatory:**

- Python ≥ 3.8
- NumPy ≥ 1.20.0

**Optional:**

- FAISS ≥ 1.7.0 (for RS-k-means++ with FAISS indices)
- scikit-learn (for integration examples)
- pytest (for running tests)

Installing with FAISS Support
------------------------------

For full functionality including FAISS-based approximate nearest neighbor search, install FAISS first:

**Via Conda (Recommended):**

.. code-block:: bash

   conda install -c pytorch faiss-cpu
   pip install kmeans-seeding

**Via Conda with GPU support:**

.. code-block:: bash

   conda install -c pytorch faiss-gpu
   pip install kmeans-seeding

.. note::
   FAISS is **optional**. The package works without it using FastLSH and GoogleLSH indices, which are built-in and highly optimized (Nov 2025: 20-40% faster than before).

Installing from Source
----------------------

For development or the latest features:

.. code-block:: bash

   git clone https://github.com/poojanshah/kmeans-seeding.git
   cd kmeans-seeding
   pip install -e .

**Build requirements:**

- CMake ≥ 3.15
- C++17 compatible compiler (GCC ≥ 7, Clang ≥ 5, MSVC ≥ 2017)
- pybind11 ≥ 2.6

Platform-Specific Notes
-----------------------

macOS
~~~~~

For OpenMP support (faster parallel processing):

.. code-block:: bash

   brew install libomp
   pip install kmeans-seeding

Without OpenMP, the package still works but uses single-threaded distance computations.

Linux
~~~~~

OpenMP is usually available by default. If needed:

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt-get install libomp-dev

   # Fedora/RHEL
   sudo yum install libomp-devel

Windows
~~~~~~~

Install Visual Studio 2017 or later for the C++ compiler. OpenMP is included with MSVC.

Verifying Installation
----------------------

Test that the package is correctly installed:

.. code-block:: python

   import kmeans_seeding
   import numpy as np

   # Quick test
   X = np.random.randn(100, 10)
   centers = kmeans_seeding.kmeanspp(X, n_clusters=5)
   print(f"Initialized {len(centers)} centers")

Check available algorithms:

.. code-block:: python

   from kmeans_seeding import kmeanspp, rskmeans, afkmc2, multitree_lsh
   print("All algorithms imported successfully!")

Troubleshooting
---------------

C++ Extension Not Found
~~~~~~~~~~~~~~~~~~~~~~~

**Error**: ``ImportError: No module named '_core'``

**Solution**: Rebuild the C++ extension:

.. code-block:: bash

   pip install --force-reinstall --no-cache-dir kmeans-seeding

FAISS Not Found
~~~~~~~~~~~~~~~

**Error**: ``RuntimeError: FAISS index type 'LSH' requested but FAISS library is not available``

**Solution**: Either:

1. Install FAISS: ``conda install -c pytorch faiss-cpu``
2. Use FAISS-free indices: ``index_type='FastLSH'`` or ``index_type='GoogleLSH'``

.. code-block:: python

   # Works without FAISS
   centers = rskmeans(X, n_clusters=10, index_type='FastLSH')

OpenMP Warnings
~~~~~~~~~~~~~~~

**Warning**: ``OMP: Warning: ... libomp has already been initialized``

**Solution**: This is harmless but can be suppressed:

.. code-block:: python

   import os
   os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

Upgrading
---------

To upgrade to the latest version:

.. code-block:: bash

   pip install --upgrade kmeans-seeding

To upgrade and rebuild from source:

.. code-block:: bash

   pip install --upgrade --force-reinstall --no-cache-dir kmeans-seeding

Uninstalling
------------

.. code-block:: bash

   pip uninstall kmeans-seeding
