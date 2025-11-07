Standard k-means++
==================

.. currentmodule:: kmeans_seeding

Overview
--------

**Standard k-means++** is the classic k-means initialization algorithm that carefully seeds cluster centers using D² sampling to achieve an :math:`O(\\log k)` approximation to the optimal k-means clustering.

**When to use:**

- Small to medium datasets (n < 10,000)
- When you need the baseline/reference implementation
- When theoretical guarantees are critical
- For comparison with faster approximations

**Key advantages:**

✅ Proven :math:`O(\\log k)` approximation guarantee

✅ Simple and well-understood

✅ No hyperparameters to tune

✅ Baseline for comparing other algorithms

Python API
----------

.. function:: kmeanspp(X, n_clusters, *, random_state=None)

   Standard k-means++ initialization using D² sampling.

   :param X: Training data
   :type X: array-like of shape (n_samples, n_features)

   :param n_clusters: Number of clusters
   :type n_clusters: int

   :param random_state: Random seed for reproducibility
   :type random_state: int, optional

   :return: Initial cluster centers
   :rtype: ndarray of shape (n_clusters, n_features)

   **Examples:**

   .. code-block:: python

      from kmeans_seeding import kmeanspp
      import numpy as np

      X = np.random.randn(1000, 20)
      centers = kmeanspp(X, n_clusters=10, random_state=42)

   With scikit-learn:

   .. code-block:: python

      from kmeans_seeding import kmeanspp
      from sklearn.cluster import KMeans

      centers = kmeanspp(X, n_clusters=10)
      kmeans = KMeans(n_clusters=10, init=centers, n_init=1)
      labels = kmeans.fit_predict(X)

Algorithm
---------

D² Sampling
~~~~~~~~~~~

The algorithm selects k centers sequentially:

1. Choose first center uniformly at random
2. For each subsequent center:

   a. Compute D²(x) = squared distance to nearest selected center
   b. Sample new center with probability ∝ D²(x)
   c. Add to selected centers

**Time complexity**: :math:`O(nkd)` for n points, k centers, d dimensions

**Space complexity**: :math:`O(nd)` for storing data

**Approximation**: :math:`\\mathbb{E}[\\Phi] \\leq O(\\log k) \\cdot \\Phi_k(X)`

Advantages
----------

1. **Strong theoretical guarantees**: Proven :math:`O(\\log k)` approximation
2. **No hyperparameters**: Works out of the box
3. **Deterministic quality**: Given seed, always same quality
4. **Simple implementation**: Easy to understand and verify

Limitations
-----------

1. **Slow for large data**: :math:`O(nkd)` becomes prohibitive
2. **Sequential**: Can't easily parallelize center selection
3. **No approximation**: Must compute all distances exactly

When to Use Alternatives
------------------------

Use faster alternatives when:

- **n > 10,000**: RS-k-means++ or AFK-MC² are much faster
- **k > 100**: Fast-LSH scales better with many clusters
- **d > 100**: Fast-LSH optimized for high dimensions
- **Need speed**: Any fast algorithm is 10-100× faster

**Recommendation**: Use standard k-means++ for:
- Small datasets
- Establishing baselines
- When simplicity matters
- Academic/research comparisons

See Also
--------

- :doc:`rskmeans` - Fast approximation using rejection sampling
- :doc:`afkmc2` - Fast approximation using MCMC
- :doc:`fast_lsh` - Fast approximation using tree embedding
- :doc:`comparison` - Detailed algorithm comparison
