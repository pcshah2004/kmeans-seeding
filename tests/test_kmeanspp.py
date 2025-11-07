"""
Tests for kmeanspp (standard k-means++) initialization.
"""

import pytest
import numpy as np
from kmeans_seeding import kmeanspp


class TestKmeansPPBasic:
    """Basic functionality tests."""

    def test_returns_correct_shape(self, small_dataset):
        """Test that output has correct shape."""
        n_clusters = 5
        centers = kmeanspp(small_dataset, n_clusters=n_clusters)

        assert centers.shape == (n_clusters, small_dataset.shape[1])
        assert centers.dtype == np.float64  # C++ uses double precision

    def test_returns_actual_data_points(self, small_dataset):
        """Test that returned centers are from the dataset."""
        centers = kmeanspp(small_dataset, n_clusters=5)

        # Each center should be close to at least one data point
        for center in centers:
            distances = np.linalg.norm(small_dataset - center, axis=1)
            assert np.min(distances) < 1e-5, "Center not from dataset"

    def test_no_nan_values(self, small_dataset):
        """Test that centers don't contain NaN values."""
        centers = kmeanspp(small_dataset, n_clusters=5)
        assert not np.any(np.isnan(centers))

    def test_no_inf_values(self, small_dataset):
        """Test that centers don't contain infinite values."""
        centers = kmeanspp(small_dataset, n_clusters=5)
        assert not np.any(np.isinf(centers))


class TestKmeansPPReproducibility:
    """Test reproducibility with random_state."""

    def test_same_seed_same_result(self, small_dataset):
        """Test that same seed produces same results."""
        centers1 = kmeanspp(small_dataset, n_clusters=5, random_state=42)
        centers2 = kmeanspp(small_dataset, n_clusters=5, random_state=42)

        np.testing.assert_array_equal(centers1, centers2)

    def test_different_seed_different_result(self, small_dataset):
        """Test that different seeds produce different results."""
        centers1 = kmeanspp(small_dataset, n_clusters=5, random_state=42)
        centers2 = kmeanspp(small_dataset, n_clusters=5, random_state=123)

        # Should be different (very unlikely to be identical)
        assert not np.allclose(centers1, centers2)


class TestKmeansPPEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_k_equals_one(self, small_dataset):
        """Test with k=1 (single cluster)."""
        centers = kmeanspp(small_dataset, n_clusters=1)
        assert centers.shape == (1, small_dataset.shape[1])

    def test_k_equals_n(self, small_dataset):
        """Test with k=n (all points as centers)."""
        n = small_dataset.shape[0]
        centers = kmeanspp(small_dataset, n_clusters=n)
        assert centers.shape == (n, small_dataset.shape[1])

    def test_small_k(self, medium_dataset):
        """Test with small k."""
        centers = kmeanspp(medium_dataset, n_clusters=2)
        assert centers.shape == (2, medium_dataset.shape[1])

    def test_large_k(self, medium_dataset):
        """Test with large k (half of dataset)."""
        k = medium_dataset.shape[0] // 2
        centers = kmeanspp(medium_dataset, n_clusters=k)
        assert centers.shape == (k, medium_dataset.shape[1])


class TestKmeansPPInputValidation:
    """Test input validation and error handling."""

    def test_invalid_n_clusters_negative(self, small_dataset):
        """Test that negative n_clusters raises error."""
        with pytest.raises(ValueError):
            kmeanspp(small_dataset, n_clusters=-1)

    def test_invalid_n_clusters_zero(self, small_dataset):
        """Test that zero n_clusters raises error."""
        with pytest.raises(ValueError):
            kmeanspp(small_dataset, n_clusters=0)

    def test_invalid_n_clusters_too_large(self, small_dataset):
        """Test that n_clusters > n_samples raises error."""
        n = small_dataset.shape[0]
        with pytest.raises(ValueError):
            kmeanspp(small_dataset, n_clusters=n + 1)

    def test_1d_array_raises_error(self):
        """Test that 1D array raises error."""
        X = np.random.randn(100).astype(np.float64)
        with pytest.raises(ValueError):
            kmeanspp(X, n_clusters=5)

    def test_3d_array_raises_error(self):
        """Test that 3D array raises error."""
        X = np.random.randn(10, 10, 10).astype(np.float64)
        with pytest.raises(ValueError):
            kmeanspp(X, n_clusters=5)


class TestKmeansPPIntegration:
    """Integration tests with sklearn."""

    def test_works_with_sklearn(self, medium_dataset):
        """Test that centers work with sklearn KMeans."""
        from sklearn.cluster import KMeans

        centers = kmeanspp(medium_dataset, n_clusters=10)

        kmeans = KMeans(n_clusters=10, init=centers, n_init=1, max_iter=10)
        labels = kmeans.fit_predict(medium_dataset)

        assert len(labels) == len(medium_dataset)
        assert len(np.unique(labels)) <= 10

    def test_improves_kmeans_convergence(self, blobs_dataset):
        """Test that good initialization helps k-means converge faster."""
        from sklearn.cluster import KMeans

        centers = kmeanspp(blobs_dataset, n_clusters=5)

        # Should converge in few iterations with good init
        kmeans = KMeans(n_clusters=5, init=centers, n_init=1, max_iter=10)
        kmeans.fit(blobs_dataset)

        # Check that it actually clustered
        assert kmeans.n_iter_ < 10  # Should converge quickly

    def test_comparable_to_sklearn_kmeanspp(self, medium_dataset):
        """Test that our k-means++ is comparable to sklearn's."""
        from sklearn.cluster import KMeans

        # Our implementation
        our_centers = kmeanspp(medium_dataset, n_clusters=10, random_state=42)

        # sklearn's implementation
        sklearn_kmeans = KMeans(n_clusters=10, init='k-means++', n_init=1, random_state=42)
        sklearn_kmeans.fit(medium_dataset)

        # Both should produce valid centers
        assert our_centers.shape == sklearn_kmeans.cluster_centers_.shape
