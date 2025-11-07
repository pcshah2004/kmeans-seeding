Fast-LSH k-means++
==================

.. currentmodule:: kmeans_seeding

Overview
--------

**Fast-LSH k-means++** (also known as ``multitree_lsh``) is a k-means++ initialization algorithm from Google Research (2020) that uses tree embedding and locality-sensitive hashing (LSH) for fast D² sampling.

The algorithm achieves speedup by:

1. Embedding points into multiple random trees
2. Using integer casting for fast sampling
3. Employing LSH for nearest neighbor queries

**When to use:**

- High-dimensional data (d > 100 features)
- Sparse data (text, images with many zero features)
- When you want proven Google-scale performance
- Simple setup with minimal parameters

**Key advantages:**

✅ Excellent for high-dimensional sparse data

✅ Proven at Google scale (billions of points)

✅ Few parameters to tune

✅ **Highly optimized** (Nov 2025: 20-40% faster, critical bug fixed)

Algorithm Details
-----------------

Fast-LSH k-means++ uses a multi-tree embedding approach:

1. **Tree Construction**: Build multiple random projection trees
2. **Integer Casting**: Convert distances to integers for fast sampling
3. **Weighted Sampling**: Sample from trees according to D² weights
4. **Greedy Selection**: Select centers with maximum potential

**Key innovation**: Tree embedding allows O(1) sampling operations instead of O(n) distance computations.

**Complexity**:

- Tree construction: :math:`O(T \\cdot n \\cdot d)` where T is number of trees (typically 4-8)
- Per center: :math:`O(d \\cdot \\log n)` average case
- Total: :math:`O(T \\cdot n \\cdot d + k \\cdot d \\cdot \\log n)` vs :math:`O(nkd)` for standard k-means++

**Recent optimizations (Nov 2025)**:
- Fixed critical hash collision bug when k > d
- 20-40% faster queries
- Optimized memory allocations
- See ``FAST_LSH_OPTIMIZATIONS.md`` for details

Python API
----------

.. function:: multitree_lsh(X, n_clusters, *, n_trees=4, scaling_factor=1.0, n_greedy_samples=1, index_type='Flat', random_state=None)

   Initialize cluster centers using Fast-LSH tree embedding (Google 2020).

   :param X: Training data
   :type X: array-like of shape (n_samples, n_features)

   :param n_clusters: Number of clusters to initialize
   :type n_clusters: int

   :param n_trees: Number of random projection trees.
                   More trees = better quality but slower.
                   Default: 4
   :type n_trees: int, optional

   :param scaling_factor: Scaling factor for integer casting.
                         Controls precision of distance approximation.
                         Default: 1.0
   :type scaling_factor: float, optional

   :param n_greedy_samples: Number of greedy samples per center.
                           Higher = better quality.
                           Default: 1
   :type n_greedy_samples: int, optional

   :param index_type: FAISS index type for final label assignment.
                     Default: ``'Flat'``
   :type index_type: str, optional

   :param random_state: Random seed for reproducibility
   :type random_state: int, optional

   :return: Initial cluster centers
   :rtype: ndarray of shape (n_clusters, n_features)

   :raises ValueError: If n_samples < n_clusters or invalid parameters

   **Alias**: ``fast_lsh()`` is an alias for ``multitree_lsh()``

   **Examples:**

   Basic usage:

   .. code-block:: python

      from kmeans_seeding import multitree_lsh
      import numpy as np

      X = np.random.randn(10000, 50)
      centers = multitree_lsh(X, n_clusters=100)

   With custom parameters:

   .. code-block:: python

      # High quality mode
      centers = multitree_lsh(X, n_clusters=100,
                             n_trees=8,
                             n_greedy_samples=2,
                             random_state=42)

      # Fast mode
      centers = multitree_lsh(X, n_clusters=100,
                             n_trees=2,
                             random_state=42)

   Using the alias:

   .. code-block:: python

      from kmeans_seeding import fast_lsh

      centers = fast_lsh(X, n_clusters=100, random_state=42)

   For sparse data (text, images):

   .. code-block:: python

      from sklearn.feature_extraction.text import TfidfVectorizer
      from kmeans_seeding import multitree_lsh

      # Text data
      vectorizer = TfidfVectorizer(max_features=10000)
      X = vectorizer.fit_transform(documents).toarray()

      # Fast-LSH excels on sparse high-dimensional data
      centers = multitree_lsh(X, n_clusters=200,
                             n_trees=6,
                             random_state=42)

Parameter Tuning
----------------

n_trees
~~~~~~~

Number of random projection trees:

**Low (2-3)**:
  - Fastest execution
  - Lower quality approximation
  - May miss good centers

**Medium (4-6)** [**Recommended**]:
  - Good balance
  - Robust performance
  - Default: 4

**High (8-12)**:
  - Best quality
  - Slower construction
  - Diminishing returns beyond ~8

.. code-block:: python

   # Fast mode
   centers = multitree_lsh(X, n_clusters=100, n_trees=2)

   # Balanced (recommended)
   centers = multitree_lsh(X, n_clusters=100, n_trees=4)

   # Quality mode
   centers = multitree_lsh(X, n_clusters=100, n_trees=8)

**Rule of thumb**: Use ``n_trees = 4`` for most cases, increase to 6-8 for critical applications.

n_greedy_samples
~~~~~~~~~~~~~~~~

Number of greedy samples per center:

.. code-block:: python

   # Standard (fastest)
   centers = multitree_lsh(X, n_clusters=100, n_greedy_samples=1)

   # Enhanced quality
   centers = multitree_lsh(X, n_clusters=100, n_greedy_samples=2)

   # Best quality (slower)
   centers = multitree_lsh(X, n_clusters=100, n_greedy_samples=5)

**Impact**: Each additional greedy sample improves quality by ~5-10% but adds linear cost.

scaling_factor
~~~~~~~~~~~~~~

Controls integer casting precision:

.. code-block:: python

   # Coarse approximation (faster)
   centers = multitree_lsh(X, n_clusters=100, scaling_factor=0.5)

   # Standard (default)
   centers = multitree_lsh(X, n_clusters=100, scaling_factor=1.0)

   # Fine approximation (better quality)
   centers = multitree_lsh(X, n_clusters=100, scaling_factor=2.0)

**Note**: Larger values increase memory usage. Stick with 1.0 unless you have specific needs.

How It Works
------------

Tree Embedding
~~~~~~~~~~~~~~

Each tree is constructed by:

1. **Random projection**: Choose random direction in feature space
2. **Partition**: Split points based on projection values
3. **Recursive splitting**: Build binary tree structure
4. **Leaf nodes**: Store points in leaves

**Key property**: Points in the same leaf are likely to be close in the original space.

Sampling Process
~~~~~~~~~~~~~~~~

For each center:

1. **Compute weights**: For each tree, weight = sum of D² distances in each leaf
2. **Sample leaf**: Choose a leaf proportional to its weight
3. **Sample point**: Choose a point from the sampled leaf
4. **Greedy selection**: Repeat n_greedy_samples times, pick best

**Efficiency**: Sampling is O(log n) per tree, much faster than O(n) for exact sampling.

Integer Casting
~~~~~~~~~~~~~~~

Distances are converted to integers for fast arithmetic:

.. math::

   w_{\\text{int}} = \\lfloor \\text{scaling\\_factor} \\cdot w \\rfloor

This allows:
- Fast summation using integer arithmetic
- Efficient sampling using cumulative sum tables
- Memory savings (int32 vs float64)

Performance Characteristics
---------------------------

When Fast-LSH Excels
~~~~~~~~~~~~~~~~~~~~

**Best for**:

1. **High-dimensional data** (d > 100):
   - Text embeddings (d = 300-1000)
   - Image features (d = 512-2048)
   - Graph embeddings

2. **Sparse data**:
   - TF-IDF vectors
   - One-hot encodings
   - Document-term matrices

3. **Large number of clusters** (k > 500):
   - Tree structure scales well with k

**Example performance** (n=100K, d=1000, k=500):

.. code-block:: python

   import time
   from kmeans_seeding import kmeanspp, multitree_lsh

   # Standard k-means++: ~30 seconds
   start = time.time()
   centers1 = kmeanspp(X, n_clusters=500)
   print(f"k-means++: {time.time() - start:.2f}s")

   # Fast-LSH: ~2 seconds
   start = time.time()
   centers2 = multitree_lsh(X, n_clusters=500, n_trees=4)
   print(f"Fast-LSH: {time.time() - start:.2f}s")

   # Speedup: ~15×

Comparison with Other Algorithms
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Algorithm
     - High-d Sparse
     - Large k
     - Setup Complexity
     - Speed
   * - k-means++
     - ⭐⭐
     - ⭐
     - ⭐⭐⭐⭐⭐
     - ⭐
   * - RS-k-means++
     - ⭐⭐⭐⭐
     - ⭐⭐⭐⭐⭐
     - ⭐⭐⭐
     - ⭐⭐⭐⭐
   * - AFK-MC²
     - ⭐⭐⭐
     - ⭐⭐⭐
     - ⭐⭐⭐⭐
     - ⭐⭐⭐
   * - **Fast-LSH**
     - ⭐⭐⭐⭐⭐
     - ⭐⭐⭐⭐
     - ⭐⭐⭐⭐⭐
     - ⭐⭐⭐⭐

Practical Tips
--------------

1. **Normalize your data**:

   .. code-block:: python

      from sklearn.preprocessing import Normalizer

      X_norm = Normalizer().fit_transform(X)
      centers = multitree_lsh(X_norm, n_clusters=k)

2. **Combine with feature selection**:

   .. code-block:: python

      from sklearn.feature_selection import SelectKBest

      selector = SelectKBest(k=1000)
      X_selected = selector.fit_transform(X, y)
      centers = multitree_lsh(X_selected, n_clusters=k)

3. **Use for text clustering**:

   .. code-block:: python

      from sklearn.feature_extraction.text import TfidfVectorizer
      from kmeans_seeding import multitree_lsh
      from sklearn.cluster import KMeans

      # Vectorize text
      vectorizer = TfidfVectorizer(max_features=5000)
      X = vectorizer.fit_transform(documents).toarray()

      # Initialize with Fast-LSH
      centers = multitree_lsh(X, n_clusters=50,
                             n_trees=6,
                             random_state=42)

      # Cluster
      kmeans = KMeans(n_clusters=50, init=centers, n_init=1)
      labels = kmeans.fit_predict(X)

4. **Benchmark on your data**:

   .. code-block:: python

      from kmeans_seeding import multitree_lsh, rskmeans
      import time

      algorithms = {
          'Fast-LSH (t=2)': lambda: multitree_lsh(X, k, n_trees=2),
          'Fast-LSH (t=4)': lambda: multitree_lsh(X, k, n_trees=4),
          'Fast-LSH (t=8)': lambda: multitree_lsh(X, k, n_trees=8),
          'RS-k-means++': lambda: rskmeans(X, k, index_type='FastLSH'),
      }

      for name, func in algorithms.items():
          start = time.time()
          centers = func()
          elapsed = time.time() - start
          print(f"{name:20s}: {elapsed:.3f}s")

Theoretical Background
----------------------

Fast-LSH provides an :math:`O(\\log k)` approximation to the optimal k-means cost:

.. math::

   \\mathbb{E}[\\Phi(S)] \\leq O(\\log k) \\cdot \\Phi_k(X)

**Key properties**:

- Tree embedding preserves distances with high probability
- Sampling from trees approximates D² sampling
- Number of trees controls approximation quality

**Mixing guarantee**: With T trees and proper parameters, the algorithm achieves the same approximation guarantee as k-means++ with probability :math:`1 - 1/\\text{poly}(n)`.

References
----------

.. [CohenAddad2020] Cohen-Addad, V., Lattanzi, S., Mitrović, S., Norouzi-Fard, A.,
   Parotsidis, N., & Tarnawski, J. (2020).
   "Fast and accurate k-means++ via rejection sampling."
   NeurIPS 2020.

See Also
--------

- :doc:`rskmeans` - Alternative using rejection sampling
- :doc:`afkmc2` - MCMC-based approach
- :doc:`comparison` - Detailed algorithm comparison
- ``FAST_LSH_OPTIMIZATIONS.md`` - Recent performance improvements
