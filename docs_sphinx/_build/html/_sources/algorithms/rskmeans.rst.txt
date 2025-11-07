RS-k-means++: Rejection Sampling
=================================

.. currentmodule:: kmeans_seeding

Overview
--------

**RS-k-means++** (Rejection Sampling k-means++) is a fast k-means++ initialization algorithm that uses rejection sampling with approximate nearest neighbor (ANN) queries to efficiently sample from the D² distribution.

The algorithm achieves speedup over standard k-means++ by:

1. Using approximate nearest neighbor search instead of exact distances
2. Employing rejection sampling to avoid computing all distances
3. Leveraging FAISS or custom LSH data structures for fast ANN queries

**When to use:**

- Large datasets (n > 10,000 samples)
- High-dimensional data (d > 50 features)
- Many clusters (k > 100)
- When quality matters but exact k-means++ is too slow

**Key advantages:**

✅ Near-optimal clustering quality (comparable to k-means++)

✅ Much faster than standard k-means++ on large datasets

✅ Theoretical guarantees on approximation quality

✅ Multiple index types for speed/accuracy tradeoff

Algorithm Details
-----------------

RS-k-means++ samples centers using rejection sampling from the D² distribution:

.. math::

   P(x | S) \\propto \\Delta(x, S)

where :math:`\\Delta(x, S) = \\min_{c \\in S} \\|x - c\\|^2` is the squared distance to the nearest selected center.

**Key innovation**: Instead of computing exact nearest center distances, we use approximate nearest neighbor queries, which are much faster for high-dimensional data.

**Complexity**:

- Preprocessing: :math:`O(\\text{nnz}(X))` where nnz(X) is the number of non-zeros
- Per center: :math:`O(m \\cdot d)` where m is the number of rejection sampling iterations
- Total: :math:`O(\\text{nnz}(X) + k \\cdot m \\cdot d)` vs :math:`O(nkd)` for standard k-means++

Python API
----------

.. function:: rskmeans(X, n_clusters, *, max_iter=50, index_type='LSH', random_state=None)

   Initialize cluster centers using RS-k-means++ rejection sampling.

   :param X: Training data
   :type X: array-like of shape (n_samples, n_features)

   :param n_clusters: Number of clusters to initialize
   :type n_clusters: int

   :param max_iter: Maximum number of rejection sampling iterations per center.
                    Higher values improve quality but take longer.
                    Default: 50
   :type max_iter: int, optional

   :param index_type: Type of approximate nearest neighbor index to use.
                      Options:

                      - ``'Flat'``: Exact search (slowest, most accurate) [requires FAISS]
                      - ``'LSH'``: FAISS LSH (fast, ~90-95% accuracy) [requires FAISS]
                      - ``'IVFFlat'``: Inverted file index (fast, ~99% accuracy) [requires FAISS]
                      - ``'HNSW'``: Hierarchical NSW (very fast, ~95-99% accuracy) [requires FAISS]
                      - ``'FastLSH'``: DHHash-based Fast LSH (very fast, ~90-95% accuracy) [**no FAISS needed**]
                      - ``'GoogleLSH'``: Google's LSH implementation (fast, ~85-90% accuracy) [**no FAISS needed**]

                      Default: ``'LSH'``
   :type index_type: str, optional

   :param random_state: Random seed for reproducibility
   :type random_state: int, optional

   :return: Initial cluster centers
   :rtype: ndarray of shape (n_clusters, n_features)

   :raises RuntimeError: If FAISS index type is requested but FAISS is not available
   :raises ValueError: If n_samples < n_clusters or invalid parameters

   **Examples:**

   Basic usage:

   .. code-block:: python

      from kmeans_seeding import rskmeans
      import numpy as np

      X = np.random.randn(10000, 50)
      centers = rskmeans(X, n_clusters=100)

   With specific index type (no FAISS needed):

   .. code-block:: python

      # FastLSH: highly optimized, works without FAISS
      centers = rskmeans(X, n_clusters=100,
                        index_type='FastLSH',
                        random_state=42)

   With FAISS for maximum speed:

   .. code-block:: python

      # Requires: conda install -c pytorch faiss-cpu
      centers = rskmeans(X, n_clusters=100,
                        index_type='IVFFlat',
                        max_iter=50,
                        random_state=42)

   Controlling quality vs speed tradeoff:

   .. code-block:: python

      # High quality (slower)
      centers = rskmeans(X, n_clusters=100,
                        index_type='Flat',
                        max_iter=100)

      # Balanced (recommended)
      centers = rskmeans(X, n_clusters=100,
                        index_type='FastLSH',
                        max_iter=50)

      # Fast (lower quality)
      centers = rskmeans(X, n_clusters=100,
                        index_type='GoogleLSH',
                        max_iter=20)

Index Type Comparison
---------------------

Choosing the right index type depends on your dataset size, dimensionality, and quality requirements:

.. list-table:: Index Type Characteristics
   :header-rows: 1
   :widths: 15 15 15 20 15 20

   * - Index Type
     - Speed
     - Accuracy
     - Best For
     - FAISS Needed
     - Notes
   * - ``Flat``
     - ⭐
     - ⭐⭐⭐⭐⭐
     - k < 100, d < 50
     - ✅ Yes
     - Exact search, baseline
   * - ``LSH``
     - ⭐⭐⭐⭐
     - ⭐⭐⭐⭐
     - General purpose
     - ✅ Yes
     - Good balance
   * - ``IVFFlat``
     - ⭐⭐⭐⭐⭐
     - ⭐⭐⭐⭐⭐
     - k > 1000
     - ✅ Yes
     - Best for large k
   * - ``HNSW``
     - ⭐⭐⭐⭐⭐
     - ⭐⭐⭐⭐
     - k > 10,000
     - ✅ Yes
     - Very fast queries
   * - ``FastLSH``
     - ⭐⭐⭐⭐
     - ⭐⭐⭐⭐
     - **Recommended**
     - ❌ No
     - Optimized Nov 2025
   * - ``GoogleLSH``
     - ⭐⭐⭐
     - ⭐⭐⭐
     - Simple setup
     - ❌ No
     - Easy to use

**Recommendation**: Start with ``index_type='FastLSH'`` (no FAISS needed, highly optimized).

Parameter Tuning
----------------

max_iter
~~~~~~~~

Controls the number of rejection sampling iterations:

- **Low (10-20)**: Faster, slightly lower quality
- **Medium (50)**: Good balance (default)
- **High (100-200)**: Best quality, slower

.. code-block:: python

   # Fast mode
   centers = rskmeans(X, n_clusters=100, max_iter=20)

   # Quality mode
   centers = rskmeans(X, n_clusters=100, max_iter=100)

**Rule of thumb**: Use ``max_iter ≈ sqrt(n/k)``

index_type
~~~~~~~~~~

For different data characteristics:

**High-dimensional sparse data** (text, images):

.. code-block:: python

   centers = rskmeans(X, n_clusters=100, index_type='LSH')

**Dense data with many clusters**:

.. code-block:: python

   centers = rskmeans(X, n_clusters=1000, index_type='IVFFlat')

**No FAISS installation**:

.. code-block:: python

   centers = rskmeans(X, n_clusters=100, index_type='FastLSH')

Performance Tips
----------------

1. **Use FastLSH for most cases**: It's highly optimized (Nov 2025) and doesn't require FAISS

2. **Increase max_iter for better quality**: If clustering quality is poor, try doubling max_iter

3. **Use IVFFlat for large k**: When k > 1000, IVFFlat is much faster than other indices

4. **Normalize your data**: RS-k-means++ works better on normalized data

   .. code-block:: python

      from sklearn.preprocessing import StandardScaler

      scaler = StandardScaler()
      X_normalized = scaler.fit_transform(X)
      centers = rskmeans(X_normalized, n_clusters=100)

5. **Consider dimensionality reduction**: For very high dimensions (d > 500), try PCA first

Theoretical Background
----------------------

RS-k-means++ provides an approximation to the standard k-means++ objective:

.. math::

   \\mathbb{E}[\\Phi(S)] \\leq O(\\epsilon^{-3} \\log k) \\cdot \\Phi_k(X)

where:

- :math:`\\Phi(S)` is the k-means cost for centers :math:`S`
- :math:`\\Phi_k(X)` is the optimal k-means cost
- :math:`\\epsilon` is the approximation parameter (depends on index type)

**Key properties**:

- Quality degrades gracefully with approximation error
- Faster than exact k-means++ by orders of magnitude
- Maintains near-optimal clustering cost

References
----------

.. [Shah2025] Shah, P., Agrawal, S., & Jaiswal, R. (2025).
   "A New Rejection Sampling Approach to k-means++ With Improved Trade-Offs."
   arXiv preprint arXiv:2502.02085.

.. [Arthur2007] Arthur, D., & Vassilvitskii, S. (2007).
   "k-means++: The advantages of careful seeding."
   SODA 2007.

See Also
--------

- :doc:`afkmc2` - Alternative MCMC-based sampling
- :doc:`fast_lsh` - Simpler LSH-based algorithm
- :doc:`comparison` - Detailed algorithm comparison
