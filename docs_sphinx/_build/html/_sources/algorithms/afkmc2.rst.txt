AFK-MC²: Adaptive Fast k-MC²
=============================

.. currentmodule:: kmeans_seeding

Overview
--------

**AFK-MC²** (Adaptive Fast k-Markov Chain Monte Carlo squared) is a k-means++ initialization algorithm that uses Markov Chain Monte Carlo (MCMC) sampling to approximate the D² distribution without computing all pairwise distances.

The algorithm achieves speedup by:

1. Using MCMC to sample from the D² distribution
2. Computing only distances needed for the Markov chain
3. Adapting chain length based on data characteristics

**When to use:**

- Medium to large datasets (1,000 < n < 100,000)
- When you want a simpler alternative to RS-k-means++
- Good balance between speed and implementation simplicity
- When FAISS is not available or desired

**Key advantages:**

✅ Sublinear time complexity per center

✅ No need for auxiliary data structures (unlike RS-k-means++)

✅ Provable approximation guarantees

✅ Simple to understand and implement

Algorithm Details
-----------------

AFK-MC² uses a Markov chain to sample from the D² distribution:

1. Start at a random point
2. Run a Markov chain for ``m`` steps
3. Accept/reject based on D² probabilities
4. Return the final state as the sampled center

**Key insight**: A Markov chain with proper transition probabilities converges to the D² distribution. We don't need to compute all distances, only those visited by the chain.

**Complexity**:

- Per center: :math:`O(m \\cdot d)` where m is the chain length
- Total: :math:`O(k \\cdot m \\cdot d)` vs :math:`O(nkd)` for standard k-means++
- Typically :math:`m \\ll n`, giving significant speedup

Python API
----------

.. function:: afkmc2(X, n_clusters, *, chain_length=200, index_type='Flat', random_state=None)

   Initialize cluster centers using AFK-MC² MCMC sampling.

   :param X: Training data
   :type X: array-like of shape (n_samples, n_features)

   :param n_clusters: Number of clusters to initialize
   :type n_clusters: int

   :param chain_length: Length of the Markov chain per center.
                        Longer chains give better quality but take more time.
                        Default: 200
   :type chain_length: int, optional

   :param index_type: FAISS index type for final label assignment (not used in sampling).
                      Options: ``'Flat'``, ``'LSH'``, ``'HNSW'``
                      Default: ``'Flat'``
   :type index_type: str, optional

   :param random_state: Random seed for reproducibility
   :type random_state: int, optional

   :return: Initial cluster centers
   :rtype: ndarray of shape (n_clusters, n_features)

   :raises ValueError: If n_samples < n_clusters or invalid parameters

   **Examples:**

   Basic usage:

   .. code-block:: python

      from kmeans_seeding import afkmc2
      import numpy as np

      X = np.random.randn(10000, 50)
      centers = afkmc2(X, n_clusters=100)

   With custom chain length:

   .. code-block:: python

      # Faster, slightly lower quality
      centers = afkmc2(X, n_clusters=100,
                      chain_length=100,
                      random_state=42)

      # Higher quality, slower
      centers = afkmc2(X, n_clusters=100,
                      chain_length=500,
                      random_state=42)

   Complete clustering pipeline:

   .. code-block:: python

      from kmeans_seeding import afkmc2
      from sklearn.cluster import KMeans

      # Initialize with AFK-MC²
      centers = afkmc2(X, n_clusters=100,
                      chain_length=200,
                      random_state=42)

      # Run k-means
      kmeans = KMeans(n_clusters=100, init=centers, n_init=1)
      labels = kmeans.fit_predict(X)

How It Works
------------

Markov Chain Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~

AFK-MC² constructs a Markov chain with the following properties:

1. **States**: All points in the dataset :math:`X`
2. **Transition probability**: From point :math:`x` to :math:`y`:

   .. math::

      P(x \\to y) = \\frac{1}{2n} + \\frac{1}{2} \\cdot \\frac{\\Delta(y, S)}{\\sum_{z} \\Delta(z, S)}

3. **Stationary distribution**: The D² distribution (after convergence)

**Intuition**: The chain randomly walks through the dataset, with bias toward points far from existing centers.

Sampling Process
~~~~~~~~~~~~~~~~

For each center :math:`c_i`:

1. Start at a random point :math:`x_0`
2. For :math:`t = 1` to ``chain_length``:

   a. Propose: Move to random point :math:`y`
   b. Accept: Based on Metropolis-Hastings criterion
   c. Update: :math:`x_t = y` if accepted, else :math:`x_t = x_{t-1}`

3. Return :math:`x_{\\text{chain\_length}}` as the new center

**Key advantage**: Only compute distances for points visited by the chain (≪ n points).

Parameter Tuning
----------------

chain_length
~~~~~~~~~~~~

The most important parameter controlling quality vs speed tradeoff:

**Low (50-100)**:
  - Fastest execution
  - Chain may not fully converge
  - Slightly lower clustering quality
  - Use for quick experiments or when speed is critical

**Medium (200-300)** [**Recommended**]:
  - Good balance
  - Chain typically converges
  - Near-optimal quality
  - Default: 200

**High (500-1000)**:
  - Best quality
  - Slower execution
  - Diminishing returns beyond ~500
  - Use when quality is paramount

.. code-block:: python

   # Fast mode
   centers = afkmc2(X, n_clusters=100, chain_length=100)

   # Balanced mode (recommended)
   centers = afkmc2(X, n_clusters=100, chain_length=200)

   # Quality mode
   centers = afkmc2(X, n_clusters=100, chain_length=500)

**Rule of thumb**: Use ``chain_length ≈ O(log k)`` as a baseline, then scale based on quality needs.

Adaptive Chain Length
~~~~~~~~~~~~~~~~~~~~~

For different numbers of clusters:

.. code-block:: python

   import numpy as np

   def adaptive_chain_length(k):
       \"\"\"Suggest chain length based on number of clusters.\"\"\"
       return max(100, int(200 * np.log(k + 1)))

   k = 500
   centers = afkmc2(X, n_clusters=k,
                   chain_length=adaptive_chain_length(k))

Performance Characteristics
---------------------------

Time Complexity
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Operation
     - AFK-MC²
     - Standard k-means++
   * - Per center
     - :math:`O(m \\cdot d)`
     - :math:`O(n \\cdot d)`
   * - Total initialization
     - :math:`O(k \\cdot m \\cdot d)`
     - :math:`O(n \\cdot k \\cdot d)`
   * - Speedup
     - :math:`n / m`
     - 1×

**Example**: For n=100,000, k=100, m=200, d=50:
  - AFK-MC²: ~1 million ops
  - k-means++: ~500 million ops
  - **Speedup: ~500×**

Quality vs Speed Tradeoff
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import time
   from sklearn.cluster import KMeans
   from kmeans_seeding import afkmc2

   X = np.random.randn(50000, 100)
   chain_lengths = [50, 100, 200, 500, 1000]

   for m in chain_lengths:
       start = time.time()
       centers = afkmc2(X, n_clusters=200, chain_length=m)
       init_time = time.time() - start

       # Measure quality (k-means cost)
       kmeans = KMeans(n_clusters=200, init=centers, n_init=1, max_iter=1)
       kmeans.fit(X)
       cost = kmeans.inertia_

       print(f"m={m:4d}: {init_time:.3f}s, cost={cost:.2e}")

Comparison with Other Algorithms
---------------------------------

vs Standard k-means++
~~~~~~~~~~~~~~~~~~~~~

**Advantages**:
  - Much faster (10-100× speedup)
  - Sublinear time per center
  - Similar clustering quality

**Disadvantages**:
  - Randomness from MCMC (need longer chains for consistency)
  - Theoretical guarantees are probabilistic

vs RS-k-means++
~~~~~~~~~~~~~~~

**Advantages**:
  - Simpler: no auxiliary data structures needed
  - More predictable behavior
  - Easier to tune (only one main parameter)

**Disadvantages**:
  - Slightly slower for very large datasets
  - Less flexible (no index type options)
  - Lower quality with short chain lengths

**Recommendation**: Use AFK-MC² when simplicity is valued and dataset size is moderate (< 100K points).

Practical Tips
--------------

1. **Start with default parameters**:

   .. code-block:: python

      centers = afkmc2(X, n_clusters=k)  # Use defaults

2. **Scale chain_length with k**:

   .. code-block:: python

      m = int(200 * np.log(k + 1))
      centers = afkmc2(X, n_clusters=k, chain_length=m)

3. **Normalize your data**:

   .. code-block:: python

      from sklearn.preprocessing import StandardScaler

      X_scaled = StandardScaler().fit_transform(X)
      centers = afkmc2(X_scaled, n_clusters=k)

4. **Use multiple initializations**:

   .. code-block:: python

      from sklearn.cluster import KMeans

      best_inertia = float('inf')
      best_labels = None

      for seed in range(10):
          centers = afkmc2(X, n_clusters=k, random_state=seed)
          kmeans = KMeans(n_clusters=k, init=centers, n_init=1)
          labels = kmeans.fit_predict(X)

          if kmeans.inertia_ < best_inertia:
              best_inertia = kmeans.inertia_
              best_labels = labels

5. **Batch processing for very large datasets**:

   .. code-block:: python

      # Sample subset for initialization
      n_sample = min(50000, len(X))
      indices = np.random.choice(len(X), n_sample, replace=False)
      X_sample = X[indices]

      # Initialize on sample
      centers = afkmc2(X_sample, n_clusters=k)

      # Use on full dataset
      kmeans = KMeans(n_clusters=k, init=centers, n_init=1)
      labels = kmeans.fit_predict(X)

Theoretical Background
----------------------

AFK-MC² provides an :math:`O(\\log k)` approximation to the optimal k-means cost with high probability, assuming the Markov chain is run for :math:`O(\\log k)` steps.

**Approximation guarantee**:

.. math::

   \\mathbb{E}[\\Phi(S)] \\leq O(\\log k) \\cdot \\Phi_k(X)

with probability at least :math:`1 - 1/k`, where:

- :math:`\\Phi(S)` is the k-means cost for centers :math:`S`
- :math:`\\Phi_k(X)` is the optimal k-means cost

**Key result**: The chain converges to the D² distribution in :math:`O(\\log n)` steps (mixing time).

References
----------

.. [Bachem2016] Bachem, O., Lucic, M., Hassani, H., & Krause, A. (2016).
   "Approximate k-means++ in sublinear time."
   AAAI Conference on Artificial Intelligence.

.. [Bachem2016b] Bachem, O., Lucic, M., & Krause, A. (2016).
   "Distributed and provably good seedings for k-means in constant rounds."
   ICML 2017.

See Also
--------

- :doc:`rskmeans` - Faster alternative using rejection sampling
- :doc:`fast_lsh` - LSH-based approach
- :doc:`comparison` - Detailed algorithm comparison
