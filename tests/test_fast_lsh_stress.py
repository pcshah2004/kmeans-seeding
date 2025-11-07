"""
Optimized stress tests for FastLSH implementation.

Focuses on critical bugs:
1. SYSTEMATIC SAMPLING BUG: k > d_padded causes step=0
2. Dimension padding edge cases
3. Numerical stability
4. Basic correctness
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after path setup
try:
    from kmeans_seeding import rskmeans
    HAS_PACKAGE = True
except ImportError:
    HAS_PACKAGE = False


# Shared fixtures for efficient test data generation
@pytest.fixture(scope="module")
def small_dataset():
    """Small dataset (100x8) for quick tests."""
    np.random.seed(42)
    return np.random.randn(100, 8).astype(np.float64)


@pytest.fixture(scope="module")
def medium_dataset():
    """Medium dataset (500x32) for thorough tests."""
    np.random.seed(42)
    return np.random.randn(500, 32).astype(np.float64)


@pytest.fixture(scope="module")
def clustered_data():
    """Highly clustered data for collision testing."""
    np.random.seed(42)
    centers = np.random.randn(5, 32) * 10
    points = []
    for c in centers:
        cluster = c + np.random.randn(100, 32) * 0.5
        points.append(cluster)
    return np.vstack(points).astype(np.float64)


class TestSystematicSamplingBug:
    """
    CRITICAL BUG: Systematic sampling in fast_lsh.cpp lines 147-151

    When k > d_padded:
        step = d_padded / k  (integer division!)
        If k > d_padded, step becomes 0
        All hash indices become 0, causing severe collisions

    Example: d=3 → d_padded=4, k=10
             step = 4/10 = 0
             indices = [0,0,0,0,0,0,0,0,0,0]
    """

    @pytest.mark.skipif(not HAS_PACKAGE, reason="Package not installed")
    @pytest.mark.parametrize("d,k", [
        (3, 10),   # d_padded=4, k=10 → step=0
        (5, 20),   # d_padded=8, k=20 → step=0
        (7, 50),   # d_padded=8, k=50 → step=0
        (10, 100), # d_padded=16, k=100 → step=0
    ])
    def test_k_greater_than_d_padded(self, d, k):
        """Test the systematic sampling bug where k > d_padded."""
        X = np.random.randn(max(200, k*2), d).astype(np.float64)

        # This should work but will have poor quality due to hash collisions
        try:
            centers = rskmeans(X, n_clusters=k, index_type='FastLSH', random_state=42)
            assert centers.shape == (k, d), f"Wrong shape for d={d}, k={k}"

            # Check that we got k distinct centers (should fail if bug exists)
            unique_centers = np.unique(centers, axis=0)
            uniqueness_ratio = len(unique_centers) / k

            # Warn if uniqueness is low (indicates the bug)
            if uniqueness_ratio < 0.9:
                pytest.fail(
                    f"SYSTEMATIC SAMPLING BUG DETECTED: d={d}, k={k}\n"
                    f"d_padded={2**int(np.ceil(np.log2(d)))}, "
                    f"step={2**int(np.ceil(np.log2(d)))//k}\n"
                    f"Only {uniqueness_ratio*100:.1f}% unique centers!\n"
                    f"Expected >90% uniqueness, got {len(unique_centers)}/{k}"
                )
        except Exception as e:
            pytest.fail(f"FastLSH crashed for d={d}, k={k}: {e}")

    @pytest.mark.skipif(not HAS_PACKAGE, reason="Package not installed")
    @pytest.mark.parametrize("d,k", [
        (3, 3),   # d_padded=4, k=3 → step=1 (boundary)
        (3, 4),   # d_padded=4, k=4 → step=1 (boundary)
        (3, 5),   # d_padded=4, k=5 → step=0 (BUG)
        (8, 8),   # d_padded=8, k=8 → step=1 (boundary)
        (8, 9),   # d_padded=8, k=9 → step=0 (BUG)
    ])
    def test_boundary_k_equals_d_padded(self, d, k):
        """Test boundary case where k ≈ d_padded."""
        X = np.random.randn(max(200, k*3), d).astype(np.float64)

        d_padded = 2**int(np.ceil(np.log2(d)))
        step = d_padded // k

        try:
            centers = rskmeans(X, n_clusters=k, index_type='FastLSH', random_state=42)
            assert centers.shape == (k, d)

            if step == 0:
                # Expect poor quality
                unique_centers = np.unique(centers, axis=0)
                if len(unique_centers) < k * 0.9:
                    pytest.fail(
                        f"BUG CONFIRMED: step=0 causes collisions\n"
                        f"d={d}, d_padded={d_padded}, k={k}, step={step}"
                    )
        except Exception as e:
            pytest.fail(f"Crashed: d={d}, k={k}, step={step}: {e}")


class TestDimensionEdgeCases:
    """Test various dimension edge cases."""

    @pytest.mark.skipif(not HAS_PACKAGE, reason="Package not installed")
    @pytest.mark.parametrize("d", [1, 3, 7, 15, 31, 63, 127])
    def test_non_power_of_2_dimensions(self, d):
        """Test dimensions requiring padding (d+1 is power of 2)."""
        X = np.random.randn(100, d).astype(np.float64)
        centers = rskmeans(X, n_clusters=5, index_type='FastLSH', random_state=42)
        assert centers.shape == (5, d)

    @pytest.mark.skipif(not HAS_PACKAGE, reason="Package not installed")
    @pytest.mark.parametrize("d", [2, 4, 8, 16, 32, 64, 128])
    def test_power_of_2_dimensions(self, d):
        """Test exact power-of-2 dimensions (no padding needed)."""
        X = np.random.randn(100, d).astype(np.float64)
        centers = rskmeans(X, n_clusters=5, index_type='FastLSH', random_state=42)
        assert centers.shape == (5, d)

    @pytest.mark.skipif(not HAS_PACKAGE, reason="Package not installed")
    def test_very_high_dimension(self):
        """Test with high dimension (512)."""
        X = np.random.randn(200, 512).astype(np.float64)
        centers = rskmeans(X, n_clusters=10, index_type='FastLSH', random_state=42)
        assert centers.shape == (10, 512)


class TestNumericalStability:
    """Test numerical stability with extreme values."""

    @pytest.mark.skipif(not HAS_PACKAGE, reason="Package not installed")
    @pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 1e3, 1e6])
    def test_different_scales(self, scale, small_dataset):
        """Test with different data scales."""
        X = small_dataset * scale
        centers = rskmeans(X, n_clusters=5, index_type='FastLSH', random_state=42)
        assert centers.shape == (5, X.shape[1])
        assert np.all(np.isfinite(centers)), f"Non-finite centers at scale {scale}"

    @pytest.mark.skipif(not HAS_PACKAGE, reason="Package not installed")
    def test_mixed_magnitude_features(self, small_dataset):
        """Test with features of vastly different magnitudes."""
        X = small_dataset.copy()
        X[:, :4] *= 1e6   # Large features
        X[:, 4:] *= 1e-6  # Small features

        centers = rskmeans(X, n_clusters=5, index_type='FastLSH', random_state=42)
        assert centers.shape == (5, X.shape[1])
        assert np.all(np.isfinite(centers))


class TestBasicCorrectness:
    """Test basic correctness and determinism."""

    @pytest.mark.skipif(not HAS_PACKAGE, reason="Package not installed")
    def test_determinism(self, medium_dataset):
        """Test that same seed produces same results."""
        centers1 = rskmeans(medium_dataset, n_clusters=10, index_type='FastLSH', random_state=42)
        centers2 = rskmeans(medium_dataset, n_clusters=10, index_type='FastLSH', random_state=42)
        np.testing.assert_array_equal(centers1, centers2, err_msg="Not deterministic!")

    @pytest.mark.skipif(not HAS_PACKAGE, reason="Package not installed")
    def test_different_seeds_differ(self, medium_dataset):
        """Test that different seeds produce different results."""
        centers1 = rskmeans(medium_dataset, n_clusters=10, index_type='FastLSH', random_state=42)
        centers2 = rskmeans(medium_dataset, n_clusters=10, index_type='FastLSH', random_state=123)

        # Results should differ (not identical)
        assert not np.array_equal(centers1, centers2), "Different seeds gave identical results!"

    @pytest.mark.skipif(not HAS_PACKAGE, reason="Package not installed")
    def test_centers_from_data(self, medium_dataset):
        """Test that centers are from the dataset."""
        centers = rskmeans(medium_dataset, n_clusters=10, index_type='FastLSH', random_state=42)

        # Each center should be close to at least one data point
        for center in centers:
            min_dist = np.min(np.linalg.norm(medium_dataset - center, axis=1))
            assert min_dist < 1.0, "Center too far from all data points"

    @pytest.mark.skipif(not HAS_PACKAGE, reason="Package not installed")
    def test_distinct_centers(self, medium_dataset):
        """Test that centers are distinct."""
        centers = rskmeans(medium_dataset, n_clusters=10, index_type='FastLSH', random_state=42)

        # All centers should be distinct
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dist = np.linalg.norm(centers[i] - centers[j])
                assert dist > 1e-10, f"Centers {i} and {j} are too similar"


class TestHashCollisions:
    """Test behavior under high collision scenarios."""

    @pytest.mark.skipif(not HAS_PACKAGE, reason="Package not installed")
    def test_duplicate_points(self):
        """Test with many duplicate points."""
        base = np.random.randn(1, 32)
        X = np.repeat(base, 100, axis=0) + np.random.randn(100, 32) * 0.01
        X = X.astype(np.float64)

        centers = rskmeans(X, n_clusters=5, index_type='FastLSH', random_state=42)
        assert centers.shape == (5, 32)

    @pytest.mark.skipif(not HAS_PACKAGE, reason="Package not installed")
    def test_clustered_data(self, clustered_data):
        """Test with highly clustered data."""
        centers = rskmeans(clustered_data, n_clusters=5, index_type='FastLSH', random_state=42)
        assert centers.shape == (5, 32)

        # Should find reasonable centers despite clustering
        unique = np.unique(centers, axis=0)
        assert len(unique) >= 4, "Too many duplicate centers in clustered data"


class TestBoundaryConditions:
    """Test boundary conditions."""

    @pytest.mark.skipif(not HAS_PACKAGE, reason="Package not installed")
    def test_k_equals_n(self):
        """Test when k equals number of points."""
        X = np.random.randn(50, 16).astype(np.float64)
        centers = rskmeans(X, n_clusters=50, index_type='FastLSH', random_state=42)
        assert centers.shape == (50, 16)

    @pytest.mark.skipif(not HAS_PACKAGE, reason="Package not installed")
    def test_k_equals_1(self, small_dataset):
        """Test with k=1 (single cluster)."""
        centers = rskmeans(small_dataset, n_clusters=1, index_type='FastLSH', random_state=42)
        assert centers.shape == (1, small_dataset.shape[1])

    @pytest.mark.skipif(not HAS_PACKAGE, reason="Package not installed")
    def test_minimum_n(self):
        """Test with minimum number of points (n=k+1)."""
        X = np.random.randn(11, 16).astype(np.float64)
        centers = rskmeans(X, n_clusters=10, index_type='FastLSH', random_state=42)
        assert centers.shape == (10, 16)


class TestDataTypes:
    """Test different data types."""

    @pytest.mark.skipif(not HAS_PACKAGE, reason="Package not installed")
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_float_types(self, dtype):
        """Test with float32 and float64."""
        X = np.random.randn(100, 16).astype(dtype)
        centers = rskmeans(X, n_clusters=5, index_type='FastLSH', random_state=42)
        assert centers.shape == (5, 16)
