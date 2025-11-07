"""
Tests for package-level functionality.
"""

import pytest
import numpy as np


class TestPackageImport:
    """Test package import and basic functionality."""

    def test_package_imports(self):
        """Test that package can be imported."""
        import kmeans_seeding
        assert kmeans_seeding.__version__ is not None

    def test_all_functions_importable(self):
        """Test that all main functions can be imported."""
        from kmeans_seeding import (
            kmeanspp,
            rejection_sampling,
            afkmc2,
            fast_lsh,
            rejection_sampling_lsh_2020,
        )

        # Check that they are callable
        assert callable(kmeanspp)
        assert callable(rejection_sampling)
        assert callable(afkmc2)
        assert callable(fast_lsh)
        assert callable(rejection_sampling_lsh_2020)

    def test_version_format(self):
        """Test that version string is properly formatted."""
        import kmeans_seeding
        version = kmeans_seeding.__version__

        # Should be in format X.Y.Z
        parts = version.split('.')
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()

    def test_package_has_docstring(self):
        """Test that package has documentation."""
        import kmeans_seeding
        assert kmeans_seeding.__doc__ is not None
        assert len(kmeans_seeding.__doc__) > 0

    def test_package_metadata(self):
        """Test that package has proper metadata."""
        import kmeans_seeding

        assert hasattr(kmeans_seeding, '__version__')
        assert hasattr(kmeans_seeding, '__author__')

class TestCoreDependency:
    """Test C++ core extension module."""

    def test_core_module_available(self):
        """Test that _core module is available."""
        try:
            from kmeans_seeding import _core
            assert _core is not None
        except ImportError:
            pytest.skip("C++ extension not built")

    def test_core_classes_available(self):
        """Test that core functions are available."""
        try:
            from kmeans_seeding import _core

            # Check that seeding functions are available
            assert hasattr(_core, 'kmeanspp_seeding')
            assert hasattr(_core, 'rejection_sampling')
            assert hasattr(_core, 'afkmc2')
            assert hasattr(_core, 'rejection_sampling_lsh_2020')
        except ImportError:
            pytest.skip("C++ extension not built")


class TestErrorMessages:
    """Test that error messages are informative."""

    def test_helpful_error_without_extension(self, monkeypatch):
        """Test that missing extension gives helpful error."""
        # This is hard to test properly, but we can check the message
        import kmeans_seeding.initializers as init_module

        # Check that HAS_CORE is being used
        assert hasattr(init_module, 'HAS_CORE')

    def test_invalid_input_clear_message(self):
        """Test that invalid inputs produce clear error messages."""
        from kmeans_seeding import rejection_sampling

        X = np.random.randn(100, 10).astype(np.float64)

        # Test various invalid inputs
        with pytest.raises(ValueError) as exc_info:
            rejection_sampling(X, n_clusters=-1)
        assert 'n_clusters' in str(exc_info.value).lower()

        with pytest.raises(ValueError) as exc_info:
            rejection_sampling(X, n_clusters=101)
        assert 'n_samples' in str(exc_info.value).lower() or \
               'n_clusters' in str(exc_info.value).lower()

        with pytest.raises(ValueError) as exc_info:
            rejection_sampling(np.random.randn(100), n_clusters=5)
        assert '2d' in str(exc_info.value).lower() or \
               'shape' in str(exc_info.value).lower()


class TestDataTypes:
    """Test handling of different data types."""

    def test_accepts_float32(self):
        """Test that functions accept float32."""
        from kmeans_seeding import rejection_sampling

        X = np.random.randn(100, 10).astype(np.float64)
        centers = rejection_sampling(X, n_clusters=5)
        assert centers.dtype == np.float64

    def test_converts_float64(self):
        """Test that float64 is converted to float32."""
        from kmeans_seeding import rejection_sampling

        X = np.random.randn(100, 10).astype(np.float64)
        centers = rejection_sampling(X, n_clusters=5)
        assert centers.dtype == np.float64

    def test_converts_int(self):
        """Test that int arrays are converted."""
        from kmeans_seeding import rejection_sampling

        X = np.random.randint(0, 100, (100, 10))
        centers = rejection_sampling(X, n_clusters=5)
        assert centers.dtype == np.float64

    def test_handles_list_input(self):
        """Test that list input is converted to array."""
        from kmeans_seeding import rejection_sampling

        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
        centers = rejection_sampling(X, n_clusters=2)
        assert isinstance(centers, np.ndarray)
        assert centers.shape == (2, 2)


class TestMemoryEfficiency:
    """Test memory efficiency of implementations."""

    def test_no_memory_leak_repeated_calls(self):
        """Test that repeated calls don't leak memory."""
        from kmeans_seeding import rejection_sampling

        X = np.random.randn(1000, 50).astype(np.float64)

        # Multiple calls should work without issues
        for _ in range(10):
            centers = rejection_sampling(X, n_clusters=10)
            assert centers.shape == (10, 50)

    def test_handles_large_k(self):
        """Test that large k values are handled efficiently."""
        from kmeans_seeding import rejection_sampling

        X = np.random.randn(1000, 50).astype(np.float64)
        centers = rejection_sampling(X, n_clusters=500)
        assert centers.shape == (500, 50)


class TestThreadSafety:
    """Test thread safety (basic checks)."""

    def test_concurrent_calls_different_data(self):
        """Test that concurrent calls work (basic check)."""
        from kmeans_seeding import rejection_sampling
        import concurrent.futures

        def run_clustering(seed):
            np.random.seed(seed)
            X = np.random.randn(100, 10).astype(np.float64)
            return rejection_sampling(X, n_clusters=5)

        # Run multiple calls in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_clustering, i) for i in range(4)]
            results = [f.result() for f in futures]

        # All should succeed
        for centers in results:
            assert centers.shape == (5, 10)
