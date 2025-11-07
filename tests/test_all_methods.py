"""
Tests comparing all seeding methods.
"""

import pytest
import numpy as np
from kmeans_seeding import (
    kmeanspp,
    rejection_sampling,
    afkmc2,
    fast_lsh,
    rejection_sampling_lsh_2020,
)


class TestAllMethodsComparison:
    """Compare all seeding methods."""

    @pytest.fixture
    def all_methods(self):
        """Return all seeding methods."""
        return {
            'k-means++': kmeanspp,
            'RS-k-means++': rejection_sampling,
            'AFK-MC²': afkmc2,
            'Fast-LSH': fast_lsh,
            'RS-LSH-2020': rejection_sampling_lsh_2020,
        }

    def test_all_methods_run_successfully(self, small_dataset, all_methods):
        """Test that all methods run without errors."""
        n_clusters = 5

        for name, method in all_methods.items():
            centers = method(small_dataset, n_clusters=n_clusters, random_state=42)
            assert centers.shape == (n_clusters, small_dataset.shape[1]), \
                f"{name} failed shape check"
            assert not np.any(np.isnan(centers)), \
                f"{name} produced NaN values"

    def test_all_methods_return_actual_points(self, small_dataset, all_methods):
        """Test that all methods return points from the dataset."""
        n_clusters = 5

        for name, method in all_methods.items():
            centers = method(small_dataset, n_clusters=n_clusters, random_state=42)

            for center in centers:
                distances = np.linalg.norm(small_dataset - center, axis=1)
                assert np.min(distances) < 1e-5, \
                    f"{name}: center not from dataset"

    def test_all_methods_reproducible(self, small_dataset, all_methods):
        """Test that all methods are reproducible with random_state."""
        n_clusters = 5

        for name, method in all_methods.items():
            centers1 = method(small_dataset, n_clusters=n_clusters, random_state=42)
            centers2 = method(small_dataset, n_clusters=n_clusters, random_state=42)

            np.testing.assert_array_equal(centers1, centers2,
                                         err_msg=f"{name} not reproducible")

    def test_all_methods_with_sklearn(self, medium_dataset, all_methods):
        """Test that all methods work with sklearn KMeans."""
        from sklearn.cluster import KMeans

        n_clusters = 10

        for name, method in all_methods.items():
            centers = method(medium_dataset, n_clusters=n_clusters, random_state=42)

            kmeans = KMeans(n_clusters=n_clusters, init=centers, n_init=1, max_iter=10)
            labels = kmeans.fit_predict(medium_dataset)

            assert len(labels) == len(medium_dataset), \
                f"{name}: wrong number of labels"
            assert len(np.unique(labels)) <= n_clusters, \
                f"{name}: too many unique labels"

    def test_all_methods_edge_case_k1(self, small_dataset, all_methods):
        """Test that all methods handle k=1."""
        n_clusters = 1

        for name, method in all_methods.items():
            centers = method(small_dataset, n_clusters=n_clusters, random_state=42)
            assert centers.shape == (1, small_dataset.shape[1]), \
                f"{name} failed k=1 test"

    def test_all_methods_edge_case_kn(self, all_methods):
        """Test that all methods handle k=n."""
        # Use smaller dataset for k=n test
        X = np.random.randn(20, 5).astype(np.float64)
        n_clusters = 20

        for name, method in all_methods.items():
            centers = method(X, n_clusters=n_clusters, random_state=42)
            assert centers.shape == (n_clusters, X.shape[1]), \
                f"{name} failed k=n test"


class TestMethodsQualityComparison:
    """Compare quality of different seeding methods."""

    def test_methods_produce_reasonable_clustering(self, blobs_dataset):
        """Test that all methods produce reasonable clustering quality."""
        from sklearn.cluster import KMeans

        n_clusters = 5
        methods = {
            'k-means++': kmeanspp,
            'RS-k-means++': rejection_sampling,
            'AFK-MC²': afkmc2,
        }

        costs = {}
        for name, method in methods.items():
            centers = method(blobs_dataset, n_clusters=n_clusters, random_state=42)

            kmeans = KMeans(n_clusters=n_clusters, init=centers, n_init=1, max_iter=100)
            kmeans.fit(blobs_dataset)

            costs[name] = kmeans.inertia_

        # All costs should be reasonably close
        cost_values = list(costs.values())
        max_cost = max(cost_values)
        min_cost = min(cost_values)

        # No method should be more than 2x worse than the best
        assert max_cost / min_cost < 2.0, \
            f"Cost variation too large: {costs}"

    def test_methods_converge_quickly(self, blobs_dataset):
        """Test that all methods help k-means converge quickly."""
        from sklearn.cluster import KMeans

        n_clusters = 5
        methods = {
            'k-means++': kmeanspp,
            'RS-k-means++': rejection_sampling,
            'AFK-MC²': afkmc2,
        }

        for name, method in methods.items():
            centers = method(blobs_dataset, n_clusters=n_clusters, random_state=42)

            kmeans = KMeans(n_clusters=n_clusters, init=centers, n_init=1, max_iter=20)
            kmeans.fit(blobs_dataset)

            # Should converge in less than 20 iterations with good init
            assert kmeans.n_iter_ < 20, \
                f"{name} took too many iterations: {kmeans.n_iter_}"


class TestCrossMethodConsistency:
    """Test consistency across methods."""

    def test_different_methods_similar_quality(self, medium_dataset):
        """Test that different methods produce similar quality results."""
        from sklearn.cluster import KMeans

        n_clusters = 10
        methods = [kmeanspp, rejection_sampling, afkmc2]

        final_costs = []
        for method in methods:
            centers = method(medium_dataset, n_clusters=n_clusters, random_state=42)

            kmeans = KMeans(n_clusters=n_clusters, init=centers, n_init=1, max_iter=50)
            kmeans.fit(medium_dataset)

            final_costs.append(kmeans.inertia_)

        # All methods should produce reasonable results
        # (within 3x of each other)
        max_cost = max(final_costs)
        min_cost = min(final_costs)
        assert max_cost / min_cost < 3.0

    def test_methods_handle_same_inputs(self, small_dataset):
        """Test that all methods accept the same input format."""
        n_clusters = 5
        methods = [kmeanspp, rejection_sampling, afkmc2, fast_lsh, rejection_sampling_lsh_2020]

        for method in methods:
            # Should accept numpy array
            centers = method(small_dataset, n_clusters=n_clusters)
            assert centers.shape == (n_clusters, small_dataset.shape[1])

            # Should accept random_state
            centers = method(small_dataset, n_clusters=n_clusters, random_state=42)
            assert centers.shape == (n_clusters, small_dataset.shape[1])


class TestAPIConsistency:
    """Test that all methods have consistent API."""

    def test_all_methods_accept_random_state(self, small_dataset):
        """Test that all methods accept random_state parameter."""
        methods = [kmeanspp, rejection_sampling, afkmc2, fast_lsh, rejection_sampling_lsh_2020]

        for method in methods:
            # Should not raise error
            centers = method(small_dataset, n_clusters=5, random_state=42)
            assert centers.shape == (5, small_dataset.shape[1])

    def test_all_methods_return_numpy_array(self, small_dataset):
        """Test that all methods return numpy arrays."""
        methods = [kmeanspp, rejection_sampling, afkmc2, fast_lsh, rejection_sampling_lsh_2020]

        for method in methods:
            centers = method(small_dataset, n_clusters=5)
            assert isinstance(centers, np.ndarray)
            assert centers.dtype == np.float64

    def test_all_methods_validate_input(self):
        """Test that all methods validate input properly."""
        methods = [kmeanspp, rejection_sampling, afkmc2, fast_lsh, rejection_sampling_lsh_2020]
        X = np.random.randn(100, 10).astype(np.float64)

        for method in methods:
            # Should raise ValueError for invalid n_clusters
            with pytest.raises(ValueError):
                method(X, n_clusters=0)

            with pytest.raises(ValueError):
                method(X, n_clusters=-1)

            with pytest.raises(ValueError):
                method(X, n_clusters=101)  # More than n_samples
