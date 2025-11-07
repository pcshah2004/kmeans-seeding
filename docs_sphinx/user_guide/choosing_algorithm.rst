Choosing an Algorithm
=====================

This guide helps you select the right algorithm based on your specific needs.

Start Here: Quick Selection
----------------------------

Answer these questions to find your algorithm:

1. **Is your dataset small (n < 10,000)?**

   → Use ``kmeanspp`` - it's fast enough and simple

2. **Do you have FAISS installed?**

   → Yes: Use ``rskmeans`` with ``index_type='IVFFlat'``

   → No: Use ``rskmeans`` with ``index_type='FastLSH'``

3. **Is your data high-dimensional (d > 100)?**

   → Use ``multitree_lsh`` - optimized for high dimensions

4. **Default case:**

   → Use ``rskmeans`` with ``index_type='FastLSH'``

By Dataset Size
---------------

Small (n < 10,000)
~~~~~~~~~~~~~~~~~~

Use standard k-means++:

.. code-block:: python

   from kmeans_seeding import kmeanspp

   centers = kmeanspp(X, n_clusters=k, random_state=42)

**Why**: Fast enough, no approximation needed.

Medium (10,000 < n < 100,000)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use AFK-MC² or RS-k-means++:

.. code-block:: python

   from kmeans_seeding import afkmc2

   centers = afkmc2(X, n_clusters=k, chain_length=200, random_state=42)

**Why**: Good balance of speed and simplicity.

Large (n > 100,000)
~~~~~~~~~~~~~~~~~~~

Use RS-k-means++ with FAISS:

.. code-block:: python

   from kmeans_seeding import rskmeans

   centers = rskmeans(X, n_clusters=k,
                     index_type='IVFFlat',
                     random_state=42)

**Why**: Maximum speedup on large data.

By Dimensionality
-----------------

Low (d < 20)
~~~~~~~~~~~~

Any algorithm works well:

.. code-block:: python

   from kmeans_seeding import kmeanspp

   centers = kmeanspp(X, n_clusters=k, random_state=42)

Medium (20 < d < 100)
~~~~~~~~~~~~~~~~~~~~~

Use RS-k-means++ with FastLSH:

.. code-block:: python

   from kmeans_seeding import rskmeans

   centers = rskmeans(X, n_clusters=k,
                     index_type='FastLSH',
                     random_state=42)

High (d > 100)
~~~~~~~~~~~~~~

Use Fast-LSH (tree embedding):

.. code-block:: python

   from kmeans_seeding import multitree_lsh

   centers = multitree_lsh(X, n_clusters=k,
                          n_trees=6,
                          random_state=42)

**Why**: Tree embedding is optimized for high dimensions.

By Number of Clusters
----------------------

Few (k < 50)
~~~~~~~~~~~~

Any algorithm is fine:

.. code-block:: python

   from kmeans_seeding import kmeanspp

   centers = kmeanspp(X, n_clusters=k, random_state=42)

Medium (50 < k < 500)
~~~~~~~~~~~~~~~~~~~~~

Use RS-k-means++ or AFK-MC²:

.. code-block:: python

   from kmeans_seeding import rskmeans

   centers = rskmeans(X, n_clusters=k,
                     index_type='FastLSH',
                     random_state=42)

Many (k > 500)
~~~~~~~~~~~~~~

Use RS-k-means++ with IVFFlat:

.. code-block:: python

   from kmeans_seeding import rskmeans

   centers = rskmeans(X, n_clusters=k,
                     index_type='IVFFlat',
                     random_state=42)

**Why**: IVFFlat scales best with many clusters.

By Data Type
------------

Dense Numerical
~~~~~~~~~~~~~~~

Standard choice:

.. code-block:: python

   from kmeans_seeding import rskmeans

   centers = rskmeans(X, n_clusters=k,
                     index_type='FastLSH',
                     random_state=42)

Sparse (Text, One-Hot)
~~~~~~~~~~~~~~~~~~~~~~

Use Fast-LSH:

.. code-block:: python

   from kmeans_seeding import multitree_lsh

   centers = multitree_lsh(X, n_clusters=k,
                          n_trees=6,
                          random_state=42)

**Why**: Tree embedding handles sparse data efficiently.

Images/Embeddings
~~~~~~~~~~~~~~~~~

Use RS-k-means++ with IVFFlat:

.. code-block:: python

   from kmeans_seeding import rskmeans

   centers = rskmeans(X, n_clusters=k,
                     index_type='IVFFlat',
                     random_state=42)

Time Series
~~~~~~~~~~~

Use RS-k-means++ or AFK-MC²:

.. code-block:: python

   from kmeans_seeding import afkmc2

   centers = afkmc2(X, n_clusters=k,
                   chain_length=200,
                   random_state=42)

By Requirements
---------------

Maximum Speed
~~~~~~~~~~~~~

.. code-block:: python

   from kmeans_seeding import rskmeans

   centers = rskmeans(X, n_clusters=k,
                     index_type='HNSW',
                     max_iter=20,
                     random_state=42)

**Quality loss**: ~1-2%

Best Quality
~~~~~~~~~~~~

.. code-block:: python

   from kmeans_seeding import rskmeans

   centers = rskmeans(X, n_clusters=k,
                     index_type='Flat',
                     max_iter=100,
                     random_state=42)

**Slower**: ~5-10× slower than fastest options

No Dependencies (No FAISS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from kmeans_seeding import rskmeans

   centers = rskmeans(X, n_clusters=k,
                     index_type='FastLSH',
                     random_state=42)

Or:

.. code-block:: python

   from kmeans_seeding import multitree_lsh

   centers = multitree_lsh(X, n_clusters=k, random_state=42)

Simple Setup
~~~~~~~~~~~~

.. code-block:: python

   from kmeans_seeding import kmeanspp

   centers = kmeanspp(X, n_clusters=k, random_state=42)

**Limitation**: Slow for large datasets.

Summary Flowchart
-----------------

.. code-block:: text

   ┌─────────────────────────────────────┐
   │     What's your dataset size?       │
   └──────────────┬──────────────────────┘
                  │
         ┌────────┴────────┐
         │                 │
   n < 10K           n > 10K
         │                 │
         ▼                 ▼
    kmeanspp     ┌──────────────────┐
                 │ Do you have      │
                 │ FAISS?           │
                 └────┬────────┬────┘
                      │        │
                    Yes       No
                      │        │
                      ▼        ▼
               rskmeans    rskmeans
               (IVFFlat)   (FastLSH)
                      │        │
                      └────┬───┘
                           │
                ┌──────────┴──────────┐
                │ Is d > 100?         │
                └──┬──────────────┬───┘
                   │              │
                  Yes            No
                   │              │
                   ▼              ▼
            multitree_lsh    rskmeans
                             (FastLSH)

See Also
--------

- :doc:`quickstart` - Getting started guide
- :doc:`../algorithms/comparison` - Detailed comparison
- :doc:`sklearn_integration` - Advanced sklearn patterns
