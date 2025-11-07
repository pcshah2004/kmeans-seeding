Quickstart Guide
================

This guide will get you started with kmeans-seeding in 5 minutes.

Basic Usage
-----------

The simplest way to use kmeans-seeding is with the ``kmeanspp()`` function:

.. code-block:: python

   from kmeans_seeding import kmeanspp
   import numpy as np

   # Your data
   X = np.random.randn(1000, 20)  # 1000 samples, 20 features

   # Initialize centers
   centers = kmeanspp(X, n_clusters=10)

   print(f"Shape: {centers.shape}")  # (10, 20)

The returned ``centers`` are NumPy arrays that can be used directly with scikit-learn.

With Scikit-Learn
-----------------

Use initialized centers with scikit-learn's KMeans:

.. code-block:: python

   from kmeans_seeding import kmeanspp
   from sklearn.cluster import KMeans
   import numpy as np

   X = np.random.randn(5000, 50)

   # Option 1: Direct initialization
   centers = kmeanspp(X, n_clusters=100)
   kmeans = KMeans(n_clusters=100, init=centers, n_init=1)
   labels = kmeans.fit_predict(X)

   # Option 2: Let sklearn handle it (slower)
   kmeans = KMeans(n_clusters=100, init='k-means++', n_init=10)
   labels = kmeans.fit_predict(X)

.. tip::
   Using pre-computed centers with ``n_init=1`` is often faster and gives similar or better results than sklearn's default ``n_init=10``.

Trying Different Algorithms
----------------------------

kmeans-seeding provides multiple algorithms. Here's how to try each:

Standard k-means++
~~~~~~~~~~~~~~~~~~

Classic D² sampling (baseline):

.. code-block:: python

   from kmeans_seeding import kmeanspp

   centers = kmeanspp(X, n_clusters=10, random_state=42)

RS-k-means++ (Fast & Accurate)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rejection sampling with approximate nearest neighbors:

.. code-block:: python

   from kmeans_seeding import rskmeans

   # Fast mode (no FAISS needed)
   centers = rskmeans(X, n_clusters=100,
                      index_type='FastLSH',
                      random_state=42)

   # With FAISS (install first: conda install -c pytorch faiss-cpu)
   centers = rskmeans(X, n_clusters=100,
                      index_type='LSH',  # or 'IVFFlat', 'HNSW'
                      random_state=42)

AFK-MC² (MCMC Sampling)
~~~~~~~~~~~~~~~~~~~~~~~

Markov Chain Monte Carlo for D² sampling:

.. code-block:: python

   from kmeans_seeding import afkmc2

   centers = afkmc2(X, n_clusters=100,
                    chain_length=200,
                    random_state=42)

Fast-LSH k-means++ (Google 2020)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tree embedding with LSH:

.. code-block:: python

   from kmeans_seeding import multitree_lsh

   centers = multitree_lsh(X, n_clusters=100,
                           n_trees=4,
                           random_state=42)

Reproducibility
---------------

All algorithms support random seeds for reproducibility:

.. code-block:: python

   from kmeans_seeding import rskmeans
   import numpy as np

   X = np.random.randn(1000, 50)

   # Same seed = same results
   centers1 = rskmeans(X, n_clusters=10, random_state=42)
   centers2 = rskmeans(X, n_clusters=10, random_state=42)

   print(np.allclose(centers1, centers2))  # True

Complete Example
----------------

Here's a complete example with visualization:

.. code-block:: python

   import numpy as np
   from sklearn.datasets import make_blobs
   from sklearn.cluster import KMeans
   from kmeans_seeding import rskmeans
   import matplotlib.pyplot as plt

   # Generate synthetic data
   X, true_labels = make_blobs(n_samples=5000, n_features=2,
                                centers=10, random_state=42)

   # Initialize with RS-k-means++
   centers = rskmeans(X, n_clusters=10,
                      index_type='FastLSH',
                      random_state=42)

   # Run k-means
   kmeans = KMeans(n_clusters=10, init=centers, n_init=1)
   labels = kmeans.fit_predict(X)

   # Plot results
   plt.figure(figsize=(12, 5))

   plt.subplot(121)
   plt.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='tab10', alpha=0.5)
   plt.title('True Labels')

   plt.subplot(122)
   plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', alpha=0.5)
   plt.scatter(centers[:, 0], centers[:, 1],
               c='red', marker='X', s=200, edgecolors='black')
   plt.title('K-Means Clustering')

   plt.tight_layout()
   plt.show()

Comparing Algorithms
--------------------

Quick benchmark to compare speed and quality:

.. code-block:: python

   import numpy as np
   import time
   from sklearn.cluster import KMeans
   from kmeans_seeding import kmeanspp, rskmeans, afkmc2, multitree_lsh

   # Generate data
   X = np.random.randn(10000, 100)
   n_clusters = 200

   algorithms = {
       'kmeanspp': lambda: kmeanspp(X, n_clusters, random_state=42),
       'rskmeans-FastLSH': lambda: rskmeans(X, n_clusters,
                                            index_type='FastLSH',
                                            random_state=42),
       'afkmc2': lambda: afkmc2(X, n_clusters, random_state=42),
       'multitree_lsh': lambda: multitree_lsh(X, n_clusters, random_state=42),
   }

   results = {}
   for name, func in algorithms.items():
       start = time.time()
       centers = func()
       elapsed = time.time() - start

       # Compute k-means cost
       kmeans = KMeans(n_clusters=n_clusters, init=centers, n_init=1, max_iter=1)
       kmeans.fit(X)
       cost = kmeans.inertia_

       results[name] = {'time': elapsed, 'cost': cost}
       print(f"{name:20s}: {elapsed:.4f}s, cost={cost:.2e}")

Next Steps
----------

- :doc:`choosing_algorithm` - Learn which algorithm to use for your data
- :doc:`sklearn_integration` - Advanced scikit-learn integration patterns
- :doc:`../algorithms/comparison` - Detailed algorithm comparison
- :doc:`../api/initializers` - Complete API reference
