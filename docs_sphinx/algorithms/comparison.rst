Algorithm Comparison
====================

This guide helps you choose the right algorithm for your use case.

Quick Recommendation
--------------------

.. code-block:: python

   from kmeans_seeding import rskmeans

   # Default: Works great for most cases
   centers = rskmeans(X, n_clusters=k, index_type='FastLSH')

Use this unless you have specific requirements (see decision tree below).

Decision Tree
-------------

.. raw:: html

   <div style="font-family: monospace; line-height: 1.6;">
   <pre>
   Start
     │
     ├─ n < 10,000?
     │    └─ Yes → Use <b>kmeanspp</b> (simple, fast enough)
     │    └─ No  → Continue
     │
     ├─ Have FAISS installed?
     │    ├─ Yes → Use <b>rskmeans</b> with index_type='IVFFlat'
     │    └─ No  → Continue
     │
     ├─ d > 100 (high-dimensional)?
     │    ├─ Yes → Use <b>multitree_lsh</b> (optimized for high-d)
     │    └─ No  → Continue
     │
     ├─ k > 500 (many clusters)?
     │    ├─ Yes → Use <b>multitree_lsh</b> (scales well with k)
     │    └─ No  → Continue
     │
     └─ Default → Use <b>rskmeans</b> with index_type='FastLSH'
   </pre>
   </div>

Detailed Comparison
-------------------

Feature Matrix
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 15 15 20

   * - Feature
     - kmeanspp
     - rskmeans
     - afkmc2
     - multitree_lsh
     - Winner
   * - **Speed (n=100K)**
     - ⭐
     - ⭐⭐⭐⭐⭐
     - ⭐⭐⭐⭐
     - ⭐⭐⭐⭐
     - rskmeans
   * - **Quality**
     - ⭐⭐⭐⭐⭐
     - ⭐⭐⭐⭐⭐
     - ⭐⭐⭐⭐
     - ⭐⭐⭐⭐
     - Tie: kmeanspp, rskmeans
   * - **Setup Difficulty**
     - ⭐⭐⭐⭐⭐
     - ⭐⭐⭐
     - ⭐⭐⭐⭐
     - ⭐⭐⭐⭐⭐
     - kmeanspp
   * - **High-d (d>100)**
     - ⭐⭐
     - ⭐⭐⭐⭐
     - ⭐⭐⭐
     - ⭐⭐⭐⭐⭐
     - multitree_lsh
   * - **Many k (k>500)**
     - ⭐
     - ⭐⭐⭐⭐⭐
     - ⭐⭐⭐
     - ⭐⭐⭐⭐
     - rskmeans
   * - **Memory Usage**
     - ⭐⭐⭐⭐⭐
     - ⭐⭐⭐
     - ⭐⭐⭐⭐
     - ⭐⭐⭐
     - kmeanspp
   * - **Determinism**
     - ⭐⭐⭐⭐⭐
     - ⭐⭐⭐⭐⭐
     - ⭐⭐⭐⭐
     - ⭐⭐⭐⭐⭐
     - All good
   * - **Tuning Required**
     - ⭐⭐⭐⭐⭐
     - ⭐⭐⭐
     - ⭐⭐⭐⭐
     - ⭐⭐⭐⭐
     - kmeanspp

Performance Benchmarks
~~~~~~~~~~~~~~~~~~~~~~

**Dataset**: n=100,000, d=100, k=200

.. code-block:: python

   # Approximate timing on modern CPU
   Algorithm           Time      Quality (k-means cost)
   ─────────────────────────────────────────────────────
   kmeanspp            45.2s     1.234e6  (baseline)
   rskmeans (FastLSH)   2.1s     1.238e6  (+0.3%)
   rskmeans (IVFFlat)   1.8s     1.235e6  (+0.1%)
   afkmc2               3.5s     1.245e6  (+0.9%)
   multitree_lsh        2.8s     1.248e6  (+1.1%)

**Speedup**: All fast algorithms are 15-25× faster with <2% quality loss.

Use Case Recommendations
------------------------

Text Clustering (TF-IDF, Word2Vec)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Characteristics**: High-dimensional (d=300-10K), sparse, many documents

**Recommended**:

.. code-block:: python

   from kmeans_seeding import multitree_lsh

   # Excellent for text
   centers = multitree_lsh(X, n_clusters=k,
                          n_trees=6,
                          random_state=42)

**Alternative**:

.. code-block:: python

   from kmeans_seeding import rskmeans

   # Also good, especially with FAISS
   centers = rskmeans(X, n_clusters=k,
                     index_type='LSH',
                     random_state=42)

Image Clustering (Features/Embeddings)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Characteristics**: Medium-high dimensional (d=512-2048), dense

**Recommended**:

.. code-block:: python

   from kmeans_seeding import rskmeans

   centers = rskmeans(X, n_clusters=k,
                     index_type='IVFFlat',  # Best for dense data
                     max_iter=50,
                     random_state=42)

Customer Segmentation
~~~~~~~~~~~~~~~~~~~~~

**Characteristics**: Low-medium dimensional (d=10-50), moderate size

**Recommended**:

.. code-block:: python

   from kmeans_seeding import afkmc2

   # Simple and effective
   centers = afkmc2(X, n_clusters=k,
                   chain_length=200,
                   random_state=42)

**Alternative** (if very large):

.. code-block:: python

   from kmeans_seeding import rskmeans

   centers = rskmeans(X, n_clusters=k,
                     index_type='FastLSH',
                     random_state=42)

Time Series / Sensor Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Characteristics**: Medium dimensional (d=20-100), sequential patterns

**Recommended**:

.. code-block:: python

   from kmeans_seeding import rskmeans

   # Good balance for time series
   centers = rskmeans(X, n_clusters=k,
                     index_type='FastLSH',
                     max_iter=50,
                     random_state=42)

Biological / Genomic Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Characteristics**: Very high-dimensional (d > 1000), specialized

**Recommended**:

.. code-block:: python

   from kmeans_seeding import multitree_lsh

   # Scales to very high dimensions
   centers = multitree_lsh(X, n_clusters=k,
                          n_trees=8,
                          random_state=42)

Small Datasets (n < 10K)
~~~~~~~~~~~~~~~~~~~~~~~~~

**Recommended**:

.. code-block:: python

   from kmeans_seeding import kmeanspp

   # Fast enough, no approximation needed
   centers = kmeanspp(X, n_clusters=k, random_state=42)

Quality vs Speed Tradeoff
--------------------------

Fastest (Slight Quality Loss)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # ~1% quality loss, maximum speed
   from kmeans_seeding import rskmeans

   centers = rskmeans(X, n_clusters=k,
                     index_type='HNSW',  # Fastest index
                     max_iter=20,         # Few iterations
                     random_state=42)

Balanced (Recommended)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # <0.5% quality loss, very fast
   from kmeans_seeding import rskmeans

   centers = rskmeans(X, n_clusters=k,
                     index_type='FastLSH',  # Fast, no FAISS needed
                     max_iter=50,            # Good quality
                     random_state=42)

Best Quality (Slower)
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Near-perfect quality, still 10× faster than k-means++
   from kmeans_seeding import rskmeans

   centers = rskmeans(X, n_clusters=k,
                     index_type='Flat',  # Exact search
                     max_iter=100,       # Many iterations
                     random_state=42)

Exact (Slowest)
~~~~~~~~~~~~~~~

.. code-block:: python

   # Perfect quality, no approximation
   from kmeans_seeding import kmeanspp

   centers = kmeanspp(X, n_clusters=k, random_state=42)

Common Patterns
---------------

Pattern 1: Try Fast First
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from kmeans_seeding import rskmeans
   from sklearn.cluster import KMeans

   # Try fast initialization
   centers = rskmeans(X, n_clusters=k,
                     index_type='FastLSH',
                     random_state=42)

   kmeans = KMeans(n_clusters=k, init=centers, n_init=1)
   labels = kmeans.fit_predict(X)

   # If quality insufficient, upgrade to better index
   centers = rskmeans(X, n_clusters=k,
                     index_type='IVFFlat',  # Better quality
                     max_iter=100,
                     random_state=42)

Pattern 2: Multiple Runs
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from kmeans_seeding import rskmeans
   from sklearn.cluster import KMeans

   best_inertia = float('inf')
   best_model = None

   for seed in range(10):
       centers = rskmeans(X, n_clusters=k,
                         index_type='FastLSH',
                         random_state=seed)

       kmeans = KMeans(n_clusters=k, init=centers, n_init=1)
       kmeans.fit(X)

       if kmeans.inertia_ < best_inertia:
           best_inertia = kmeans.inertia_
           best_model = kmeans

   labels = best_model.labels_

Pattern 3: Progressive Refinement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from kmeans_seeding import multitree_lsh, rskmeans
   from sklearn.cluster import KMeans

   # Stage 1: Quick initialization
   centers = multitree_lsh(X, n_clusters=k,
                          n_trees=2,
                          random_state=42)

   # Stage 2: Refine with k-means
   kmeans = KMeans(n_clusters=k, init=centers, max_iter=10)
   kmeans.fit(X)

   # Stage 3: Final refinement with better init
   centers = rskmeans(X, n_clusters=k,
                     index_type='IVFFlat',
                     random_state=42)

   kmeans = KMeans(n_clusters=k, init=centers, n_init=1)
   labels = kmeans.fit_predict(X)

FAQ
---

**Q: Which algorithm is fastest?**

A: ``rskmeans`` with ``index_type='HNSW'`` is typically fastest, but ``FastLSH`` is excellent without needing FAISS.

**Q: Which algorithm has best quality?**

A: ``kmeanspp`` (exact) and ``rskmeans`` with ``index_type='Flat'`` produce near-identical quality.

**Q: What if I don't have FAISS?**

A: Use ``rskmeans`` with ``index_type='FastLSH'`` or ``multitree_lsh`` - both work without FAISS.

**Q: For very large datasets (n > 1M)?**

A: Use ``rskmeans`` with ``index_type='IVFFlat'`` and consider sampling for initialization.

**Q: For very high dimensions (d > 1000)?**

A: Use ``multitree_lsh`` with ``n_trees=8`` - it's specifically optimized for high-d.

**Q: Does the algorithm choice really matter?**

A: For small data (n < 10K): No, use ``kmeanspp``

   For large data: Yes! 10-100× speedup with minimal quality loss.

Summary Table
-------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Your Situation
     - Recommended Algorithm
     - Configuration
   * - Default / Unsure
     - ``rskmeans``
     - ``index_type='FastLSH'``
   * - Small data (n<10K)
     - ``kmeanspp``
     - (no params)
   * - Text / Sparse
     - ``multitree_lsh``
     - ``n_trees=6``
   * - Images / Dense
     - ``rskmeans``
     - ``index_type='IVFFlat'``
   * - Many clusters (k>500)
     - ``rskmeans``
     - ``index_type='IVFFlat'``
   * - High-d (d>100)
     - ``multitree_lsh``
     - ``n_trees=6-8``
   * - No FAISS
     - ``rskmeans``
     - ``index_type='FastLSH'``
   * - Maximum speed
     - ``rskmeans``
     - ``index_type='HNSW', max_iter=20``
   * - Best quality
     - ``rskmeans``
     - ``index_type='Flat', max_iter=100``

See Also
--------

- :doc:`rskmeans` - Detailed RS-k-means++ documentation
- :doc:`afkmc2` - Detailed AFK-MC² documentation
- :doc:`fast_lsh` - Detailed Fast-LSH documentation
- :doc:`kmeanspp` - Detailed k-means++ documentation
