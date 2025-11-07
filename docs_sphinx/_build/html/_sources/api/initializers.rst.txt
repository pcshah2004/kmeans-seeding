API Reference
=============

Main Functions
--------------

All algorithms return cluster centers as NumPy arrays of shape ``(n_clusters, n_features)`` that can be used directly with scikit-learn's ``KMeans``.

.. currentmodule:: kmeans_seeding

rskmeans
~~~~~~~~

.. autofunction:: rskmeans

kmeanspp
~~~~~~~~

.. autofunction:: kmeanspp

afkmc2
~~~~~~

.. autofunction:: afkmc2

multitree_lsh
~~~~~~~~~~~~~

.. autofunction:: multitree_lsh

Aliases
-------

rejection_sampling
~~~~~~~~~~~~~~~~~~

Alias for :func:`rskmeans`.

.. code-block:: python

   from kmeans_seeding import rejection_sampling

   # Same as rskmeans
   centers = rejection_sampling(X, n_clusters=100)

fast_lsh
~~~~~~~~

Alias for :func:`multitree_lsh`.

.. code-block:: python

   from kmeans_seeding import fast_lsh

   # Same as multitree_lsh
   centers = fast_lsh(X, n_clusters=100)

Complete Example
----------------

.. code-block:: python

   from kmeans_seeding import (
       kmeanspp,
       rskmeans,
       afkmc2,
       multitree_lsh,
       rejection_sampling,  # alias for rskmeans
       fast_lsh,            # alias for multitree_lsh
   )
   from sklearn.cluster import KMeans
   import numpy as np

   # Generate sample data
   X = np.random.randn(10000, 50)
   k = 100

   # All algorithms have the same basic interface
   centers_kpp = kmeanspp(X, n_clusters=k, random_state=42)
   centers_rs = rskmeans(X, n_clusters=k, random_state=42)
   centers_afk = afkmc2(X, n_clusters=k, random_state=42)
   centers_lsh = multitree_lsh(X, n_clusters=k, random_state=42)

   # Use with scikit-learn
   kmeans = KMeans(n_clusters=k, init=centers_rs, n_init=1)
   labels = kmeans.fit_predict(X)

Parameter Summary
-----------------

Common Parameters
~~~~~~~~~~~~~~~~~

All functions share these parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - ``X``
     - array-like
     - Yes
     - Training data, shape (n_samples, n_features)
   * - ``n_clusters``
     - int
     - Yes
     - Number of clusters to initialize
   * - ``random_state``
     - int
     - No
     - Random seed for reproducibility

Algorithm-Specific Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**rskmeans**:

.. list-table::
   :widths: 20 15 50

   * - ``max_iter``
     - int
     - Max rejection sampling iterations (default: 50)
   * - ``index_type``
     - str
     - ANN index: 'Flat', 'LSH', 'IVFFlat', 'HNSW', 'FastLSH', 'GoogleLSH'

**afkmc2**:

.. list-table::
   :widths: 20 15 50

   * - ``chain_length``
     - int
     - Markov chain length per center (default: 200)
   * - ``index_type``
     - str
     - FAISS index for label assignment (default: 'Flat')

**multitree_lsh**:

.. list-table::
   :widths: 20 15 50

   * - ``n_trees``
     - int
     - Number of random projection trees (default: 4)
   * - ``scaling_factor``
     - float
     - Integer casting precision (default: 1.0)
   * - ``n_greedy_samples``
     - int
     - Greedy samples per center (default: 1)
   * - ``index_type``
     - str
     - FAISS index for label assignment (default: 'Flat')

Return Values
-------------

All functions return:

.. code-block:: python

   centers : ndarray of shape (n_clusters, n_features)
       The initialized cluster centers.

These can be used directly with scikit-learn:

.. code-block:: python

   from sklearn.cluster import KMeans

   kmeans = KMeans(n_clusters=k, init=centers, n_init=1)
   labels = kmeans.fit_predict(X)

Exceptions
----------

Common exceptions raised by all algorithms:

**ValueError**:
  - ``n_samples < n_clusters``
  - Invalid parameter values
  - Dimension mismatch

**RuntimeError** (rskmeans only):
  - FAISS index requested but FAISS not available
  - Invalid index type

Example:

.. code-block:: python

   from kmeans_seeding import rskmeans
   import numpy as np

   X = np.random.randn(100, 10)

   try:
       # This will fail: FAISS not installed
       centers = rskmeans(X, n_clusters=10, index_type='LSH')
   except RuntimeError as e:
       print(f"Error: {e}")
       # Use FastLSH instead (works without FAISS)
       centers = rskmeans(X, n_clusters=10, index_type='FastLSH')

Type Hints
----------

For type checkers and IDEs:

.. code-block:: python

   from typing import Optional
   import numpy as np
   from numpy.typing import ArrayLike, NDArray

   def kmeanspp(
       X: ArrayLike,
       n_clusters: int,
       *,
       random_state: Optional[int] = None
   ) -> NDArray[np.float64]:
       ...

   def rskmeans(
       X: ArrayLike,
       n_clusters: int,
       *,
       max_iter: int = 50,
       index_type: str = 'LSH',
       random_state: Optional[int] = None
   ) -> NDArray[np.float64]:
       ...

   def afkmc2(
       X: ArrayLike,
       n_clusters: int,
       *,
       chain_length: int = 200,
       index_type: str = 'Flat',
       random_state: Optional[int] = None
   ) -> NDArray[np.float64]:
       ...

   def multitree_lsh(
       X: ArrayLike,
       n_clusters: int,
       *,
       n_trees: int = 4,
       scaling_factor: float = 1.0,
       n_greedy_samples: int = 1,
       index_type: str = 'Flat',
       random_state: Optional[int] = None
   ) -> NDArray[np.float64]:
       ...

See Also
--------

- :doc:`../user_guide/quickstart` - Quick start guide
- :doc:`../algorithms/comparison` - Algorithm comparison
- :doc:`advanced` - Advanced usage patterns
