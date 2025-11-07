Scikit-Learn Integration
=========================

This guide shows advanced patterns for using kmeans-seeding with scikit-learn.

Basic Integration
-----------------

.. code-block:: python

   from kmeans_seeding import rskmeans
   from sklearn.cluster import KMeans

   # Initialize centers
   centers = rskmeans(X, n_clusters=10, random_state=42)

   # Use with KMeans
   kmeans = KMeans(n_clusters=10, init=centers, n_init=1)
   labels = kmeans.fit_predict(X)

Key Points
~~~~~~~~~~

- Set ``n_init=1`` when using pre-computed centers
- The ``init`` parameter accepts NumPy arrays
- All kmeans-seeding functions return compatible arrays

Multiple Initializations
-------------------------

Run k-means multiple times with different seeds:

.. code-block:: python

   from kmeans_seeding import rskmeans
   from sklearn.cluster import KMeans
   import numpy as np

   best_inertia = float('inf')
   best_model = None

   for seed in range(10):
       # Initialize with different seed
       centers = rskmeans(X, n_clusters=100,
                         index_type='FastLSH',
                         random_state=seed)

       # Run k-means
       kmeans = KMeans(n_clusters=100, init=centers, n_init=1)
       kmeans.fit(X)

       # Track best result
       if kmeans.inertia_ < best_inertia:
           best_inertia = kmeans.inertia_
           best_model = kmeans

   labels = best_model.labels_
   print(f"Best inertia: {best_inertia:.2e}")

Pipeline Integration
--------------------

Use in sklearn pipelines:

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.cluster import KMeans
   from kmeans_seeding import rskmeans
   import numpy as np

   # Custom KMeans with fast init
   class FastKMeans:
       def __init__(self, n_clusters, random_state=None):
           self.n_clusters = n_clusters
           self.random_state = random_state
           self.kmeans_ = None

       def fit(self, X, y=None):
           centers = rskmeans(X, self.n_clusters,
                            index_type='FastLSH',
                            random_state=self.random_state)

           self.kmeans_ = KMeans(n_clusters=self.n_clusters,
                                init=centers, n_init=1)
           self.kmeans_.fit(X)
           return self

       def predict(self, X):
           return self.kmeans_.predict(X)

       def fit_predict(self, X, y=None):
           self.fit(X)
           return self.predict(X)

   # Use in pipeline
   pipeline = Pipeline([
       ('scaler', StandardScaler()),
       ('kmeans', FastKMeans(n_clusters=10, random_state=42))
   ])

   labels = pipeline.fit_predict(X)

With MiniBatchKMeans
--------------------

For very large datasets:

.. code-block:: python

   from sklearn.cluster import MiniBatchKMeans
   from kmeans_seeding import rskmeans

   # Initialize with fast method
   centers = rskmeans(X, n_clusters=100,
                     index_type='FastLSH',
                     random_state=42)

   # Use with MiniBatchKMeans
   mbkmeans = MiniBatchKMeans(n_clusters=100,
                              init=centers,
                              n_init=1,
                              batch_size=1000)
   labels = mbkmeans.fit_predict(X)

Cross-Validation
----------------

With GridSearchCV:

.. code-block:: python

   from sklearn.model_selection import GridSearchCV
   from sklearn.cluster import KMeans
   from sklearn.metrics import silhouette_score
   from kmeans_seeding import rskmeans
   import numpy as np

   # Custom scorer
   def kmeans_score(estimator, X):
       labels = estimator.labels_
       return silhouette_score(X, labels)

   # Test different k values
   for k in [50, 100, 150, 200]:
       centers = rskmeans(X, n_clusters=k,
                         index_type='FastLSH',
                         random_state=42)

       kmeans = KMeans(n_clusters=k, init=centers, n_init=1)
       kmeans.fit(X)

       score = silhouette_score(X, kmeans.labels_)
       print(f"k={k:3d}: silhouette={score:.4f}")

Feature Preprocessing
---------------------

With scaling:

.. code-block:: python

   from sklearn.preprocessing import StandardScaler
   from kmeans_seeding import rskmeans
   from sklearn.cluster import KMeans

   # Scale features
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)

   # Initialize on scaled data
   centers = rskmeans(X_scaled, n_clusters=100,
                     index_type='FastLSH',
                     random_state=42)

   # Cluster
   kmeans = KMeans(n_clusters=100, init=centers, n_init=1)
   labels = kmeans.fit_predict(X_scaled)

With PCA:

.. code-block:: python

   from sklearn.decomposition import PCA
   from kmeans_seeding import multitree_lsh
   from sklearn.cluster import KMeans

   # Reduce dimensionality
   pca = PCA(n_components=100)
   X_pca = pca.fit_transform(X)

   # Initialize on reduced data
   centers = multitree_lsh(X_pca, n_clusters=50,
                          n_trees=4,
                          random_state=42)

   # Cluster
   kmeans = KMeans(n_clusters=50, init=centers, n_init=1)
   labels = kmeans.fit_predict(X_pca)

Evaluation Metrics
------------------

Compare initialization methods:

.. code-block:: python

   from sklearn.cluster import KMeans
   from sklearn.metrics import (
       silhouette_score,
       calinski_harabasz_score,
       davies_bouldin_score
   )
   from kmeans_seeding import kmeanspp, rskmeans, afkmc2
   import time

   algorithms = {
       'kmeanspp': lambda: kmeanspp(X, 100, random_state=42),
       'rskmeans': lambda: rskmeans(X, 100, index_type='FastLSH',
                                    random_state=42),
       'afkmc2': lambda: afkmc2(X, 100, random_state=42),
   }

   for name, init_func in algorithms.items():
       # Time initialization
       start = time.time()
       centers = init_func()
       init_time = time.time() - start

       # Cluster
       kmeans = KMeans(n_clusters=100, init=centers, n_init=1)
       kmeans.fit(X)

       # Evaluate
       silhouette = silhouette_score(X, kmeans.labels_)
       calinski = calinski_harabasz_score(X, kmeans.labels_)
       davies = davies_bouldin_score(X, kmeans.labels_)

       print(f"\n{name}:")
       print(f"  Init time: {init_time:.4f}s")
       print(f"  Inertia: {kmeans.inertia_:.2e}")
       print(f"  Silhouette: {silhouette:.4f}")
       print(f"  Calinski-Harabasz: {calinski:.2f}")
       print(f"  Davies-Bouldin: {davies:.4f}")

See Also
--------

- :doc:`quickstart` - Basic usage
- :doc:`../algorithms/comparison` - Algorithm comparison
