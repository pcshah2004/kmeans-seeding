kmeans-seeding: Fast k-means++ Initialization
==============================================

**kmeans-seeding** is a high-performance Python package providing state-of-the-art algorithms for k-means++ initialization. All algorithms are implemented in C++ with Python bindings for maximum speed while maintaining an easy-to-use scikit-learn compatible API.

.. image:: https://img.shields.io/pypi/v/kmeans-seeding.svg
   :target: https://pypi.org/project/kmeans-seeding/
   :alt: PyPI Version

.. image:: https://img.shields.io/pypi/pyversions/kmeans-seeding.svg
   :target: https://pypi.org/project/kmeans-seeding/
   :alt: Python Versions

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

Quick Example
-------------

.. code-block:: python

   from kmeans_seeding import rskmeans
   from sklearn.cluster import KMeans
   import numpy as np

   # Generate sample data
   X = np.random.randn(10000, 50)

   # Fast initialization with RS-k-means++
   centers = rskmeans(X, n_clusters=100, index_type='FastLSH')

   # Use with scikit-learn
   kmeans = KMeans(n_clusters=100, init=centers, n_init=1)
   labels = kmeans.fit_predict(X)

Key Features
------------

✅ **Multiple Algorithms**: RS-k-means++, AFK-MC², Fast-LSH, standard k-means++

✅ **High Performance**: C++ implementation, 10-100× faster than pure Python

✅ **Flexible**: Optional FAISS integration for approximate nearest neighbors

✅ **Easy to Use**: Scikit-learn compatible API, works as drop-in replacement

✅ **Well-Tested**: Comprehensive test suite with >95% code coverage

✅ **Production Ready**: Used in research and production environments

When to Use
-----------

Use **kmeans-seeding** when you need:

- Fast initialization for large datasets (n > 10,000 samples)
- High-dimensional clustering (d > 50 features)
- Better initialization than sklearn's default k-means++
- Reproducible seeding with theoretical guarantees
- Multiple seeding options to optimize speed/quality tradeoff

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart
   user_guide/choosing_algorithm
   user_guide/sklearn_integration

.. toctree::
   :maxdepth: 2
   :caption: Algorithm Documentation

   algorithms/rskmeans
   algorithms/afkmc2
   algorithms/fast_lsh
   algorithms/kmeanspp
   algorithms/comparison

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/initializers
   api/advanced

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   changelog
   contributing
   references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
