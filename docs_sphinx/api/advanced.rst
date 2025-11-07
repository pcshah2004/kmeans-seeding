Advanced Usage
==============

This page covers advanced usage patterns and customization.

Custom Distance Metrics
-----------------------

While kmeans-seeding uses Euclidean distance internally, you can preprocess your data:

.. code-block:: python

   from sklearn.metrics import pairwise_distances
   from sklearn.manifold import MDS
   from kmeans_seeding import rskmeans

   # Compute custom distance matrix
   distances = pairwise_distances(X, metric='cosine')

   # Embed in Euclidean space
   mds = MDS(n_components=50, dissimilarity='precomputed')
   X_embedded = mds.fit_transform(distances)

   # Now use Euclidean k-means++
   centers_embedded = rskmeans(X_embedded, n_clusters=10)

   # Map back to original indices
   center_indices = [np.argmin(np.linalg.norm(X_embedded - c, axis=1))
                     for c in centers_embedded]
   centers = X[center_indices]

Incremental Clustering
----------------------

Add more clusters to existing centers:

.. code-block:: python

   from kmeans_seeding import rskmeans
   from sklearn.cluster import KMeans
   import numpy as np

   # Initial clustering
   centers_10 = rskmeans(X, n_clusters=10, random_state=42)

   # Add 5 more clusters
   # Compute distances to existing centers
   kmeans_temp = KMeans(n_clusters=10, init=centers_10, n_init=1, max_iter=0)
   kmeans_temp.fit(X)
   distances = kmeans_temp.transform(X)
   min_distances_sq = np.min(distances, axis=1) ** 2

   # Sample 5 more centers using D² distribution
   probs = min_distances_sq / min_distances_sq.sum()
   new_indices = np.random.choice(len(X), size=5, replace=False, p=probs)

   # Combine
   centers_15 = np.vstack([centers_10, X[new_indices]])

Parallel Initialization
-----------------------

Initialize multiple clusterings in parallel:

.. code-block:: python

   from joblib import Parallel, delayed
   from kmeans_seeding import rskmeans
   from sklearn.cluster import KMeans

   def run_clustering(seed):
       centers = rskmeans(X, n_clusters=100,
                         index_type='FastLSH',
                         random_state=seed)
       kmeans = KMeans(n_clusters=100, init=centers, n_init=1)
       kmeans.fit(X)
       return kmeans.inertia_, kmeans.labels_

   # Run 10 clusterings in parallel
   results = Parallel(n_jobs=-1)(
       delayed(run_clustering)(seed) for seed in range(10)
   )

   # Get best result
   best_idx = np.argmin([r[0] for r in results])
   best_inertia, best_labels = results[best_idx]

Hierarchical Initialization
---------------------------

Use k-means++ for hierarchical clustering:

.. code-block:: python

   from kmeans_seeding import kmeanspp
   from sklearn.cluster import AgglomerativeClustering

   # Initialize with well-spread points
   initial_centers = kmeanspp(X, n_clusters=100, random_state=42)

   # Use as initial linkage for hierarchical clustering
   hierarchical = AgglomerativeClustering(n_clusters=10)
   labels = hierarchical.fit_predict(initial_centers)

Weighted Sampling
-----------------

Initialize with sample weights:

.. code-block:: python

   import numpy as np
   from kmeans_seeding import rskmeans

   # Define weights (e.g., based on density)
   weights = compute_density(X)  # Your density estimation

   # Sample weighted subset
   n_sample = min(50000, len(X))
   probs = weights / weights.sum()
   indices = np.random.choice(len(X), size=n_sample, p=probs)
   X_weighted = X[indices]

   # Initialize on weighted sample
   centers = rskmeans(X_weighted, n_clusters=100,
                     index_type='FastLSH')

Warm Starting
-------------

Use previous centers to initialize new clustering:

.. code-block:: python

   from sklearn.cluster import KMeans
   from kmeans_seeding import rskmeans
   import numpy as np

   # Initial clustering
   centers_old = rskmeans(X, n_clusters=10, random_state=42)
   kmeans = KMeans(n_clusters=10, init=centers_old, n_init=1)
   kmeans.fit(X)

   # New data arrives
   X_new = np.vstack([X, new_data])

   # Warm start with old centers
   centers_new = rskmeans(X_new, n_clusters=10,
                         index_type='FastLSH',
                         random_state=42)

   # Could also just use old centers directly
   kmeans = KMeans(n_clusters=10, init=centers_old, n_init=1)
   kmeans.fit(X_new)

Memory-Efficient Processing
---------------------------

For very large datasets that don't fit in memory:

.. code-block:: python

   from kmeans_seeding import rskmeans
   from sklearn.cluster import MiniBatchKMeans
   import numpy as np

   # Sample for initialization
   n_sample = 100000
   indices = np.random.choice(len(X), size=n_sample, replace=False)
   X_sample = X[indices]

   # Initialize on sample
   centers = rskmeans(X_sample, n_clusters=500,
                     index_type='FastLSH')

   # Use MiniBatchKMeans on full data
   mbkmeans = MiniBatchKMeans(n_clusters=500,
                              init=centers,
                              n_init=1,
                              batch_size=10000)

   # Process in batches
   for batch in generate_batches(X, batch_size=10000):
       mbkmeans.partial_fit(batch)

   labels = mbkmeans.predict(X)

Custom Index Parameters (RS-k-means++)
--------------------------------------

Fine-tune FAISS index behavior:

.. code-block:: python

   from kmeans_seeding import rskmeans

   # IVFFlat with custom nprobe
   # Note: This requires modifying the C++ code
   # The Python API doesn't expose all FAISS parameters yet

   # For now, use the provided index types
   centers = rskmeans(X, n_clusters=100,
                     index_type='IVFFlat',
                     max_iter=50)

   # FastLSH parameters can be tuned
   # (No FAISS needed, highly optimized)
   centers = rskmeans(X, n_clusters=100,
                     index_type='FastLSH',
                     max_iter=50)

Debugging and Profiling
-----------------------

Profile initialization time:

.. code-block:: python

   import cProfile
   import pstats
   from kmeans_seeding import rskmeans

   profiler = cProfile.Profile()
   profiler.enable()

   centers = rskmeans(X, n_clusters=100,
                     index_type='FastLSH')

   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(20)

See Also
--------

- :doc:`initializers` - API reference
- :doc:`../user_guide/sklearn_integration` - Sklearn patterns
