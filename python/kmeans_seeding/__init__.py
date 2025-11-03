"""
kmeans-seeding: Fast k-means++ Seeding Algorithms
====================================================

A library providing state-of-the-art k-means initialization algorithms
implemented in C++ with Python bindings.

Algorithms included:
- Standard k-means++
- RS-k-means++ (Rejection Sampling)
- AFK-MC² (Adaptive Fast k-MC²)
- Fast-LSH k-means++ (Google 2020)

Example usage:
    >>> from kmeans_seeding import rejection_sampling
    >>> from sklearn.cluster import KMeans
    >>>
    >>> # Get initial centers using RS-k-means++
    >>> centers = rejection_sampling(X, n_clusters=10)
    >>>
    >>> # Use with sklearn
    >>> kmeans = KMeans(n_clusters=10, init=centers, n_init=1)
    >>> kmeans.fit(X)
"""

__version__ = "0.1.0"
__author__ = "Poojan Shah, Shashwat Agrawal, Ragesh Jaiswal"
__email__ = "cs1221594@cse.iitd.ac.in"

from .initializers import (
    kmeanspp,
    rejection_sampling,
    afkmc2,
    fast_lsh,
    rejection_sampling_lsh_2020,
)

__all__ = [
    "kmeanspp",
    "rejection_sampling",
    "afkmc2",
    "fast_lsh",
    "rejection_sampling_lsh_2020",
    "__version__",
]
