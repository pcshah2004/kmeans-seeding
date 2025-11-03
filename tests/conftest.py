"""
Pytest configuration and fixtures for kmeans-seeding tests.
"""

import pytest
import numpy as np


@pytest.fixture
def small_dataset():
    """Small dataset for quick tests (100 samples, 10 features)."""
    np.random.seed(42)
    return np.random.randn(100, 10).astype(np.float64)


@pytest.fixture
def medium_dataset():
    """Medium dataset for performance tests (1000 samples, 50 features)."""
    np.random.seed(42)
    return np.random.randn(1000, 50).astype(np.float64)


@pytest.fixture
def large_dataset():
    """Large dataset for stress tests (10000 samples, 100 features)."""
    np.random.seed(42)
    return np.random.randn(10000, 100).astype(np.float64)


@pytest.fixture
def blobs_dataset():
    """Dataset with clear cluster structure."""
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(
        n_samples=500,
        n_features=20,
        centers=5,
        cluster_std=1.0,
        random_state=42
    )
    return X.astype(np.float64)


@pytest.fixture
def high_dim_dataset():
    """High-dimensional dataset (100 samples, 1000 features)."""
    np.random.seed(42)
    return np.random.randn(100, 1000).astype(np.float64)
