# kmeans-seeding Test Suite

Comprehensive test suite for the kmeans-seeding library.

## Test Files

### Algorithm-Specific Tests

1. **test_kmeanspp.py** - Standard k-means++ initialization
   - Basic functionality (shape, data points, no NaN/inf)
   - Reproducibility with random_state
   - Edge cases (k=1, k=n)
   - Input validation
   - sklearn integration

2. **test_rejection_sampling.py** - RS-k-means++ (rejection sampling)
   - Basic functionality
   - Reproducibility
   - Edge cases
   - Different FAISS index types (Flat, LSH)
   - Parameter variations (max_iter)
   - Input validation
   - sklearn integration

3. **test_afkmc2.py** - AFK-MC² (MCMC-based)
   - Basic functionality
   - Reproducibility
   - Edge cases
   - Parameter variations (chain_length)
   - Input validation
   - sklearn integration

4. **test_fast_lsh.py** - Fast-LSH k-means++ (Google 2020)
   - Basic functionality
   - Reproducibility
   - Edge cases
   - Parameter variations (n_trees, scaling_factor, n_greedy_samples)
   - Input validation
   - sklearn integration

5. **test_rejection_sampling_lsh_2020.py** - Rejection Sampling LSH (Google 2020)
   - Basic functionality
   - Reproducibility
   - Edge cases
   - Parameter variations (all parameters)
   - Input validation
   - sklearn integration

### Cross-Method Tests

6. **test_all_methods.py** - Compare all methods
   - All methods run successfully
   - All return actual data points
   - All are reproducible
   - All work with sklearn
   - Edge case consistency (k=1, k=n)
   - Quality comparison
   - Convergence speed
   - API consistency

7. **test_package.py** - Package-level tests
   - Import functionality
   - Version format
   - Metadata
   - Core extension availability
   - Error messages
   - Data type handling
   - Memory efficiency
   - Thread safety (basic)

### Test Configuration

8. **conftest.py** - Pytest fixtures
   - small_dataset (100 samples, 10 features)
   - medium_dataset (1000 samples, 50 features)
   - large_dataset (10000 samples, 100 features)
   - blobs_dataset (clear cluster structure)
   - high_dim_dataset (100 samples, 1000 features)

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/test_rejection_sampling.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_rejection_sampling.py::TestRejectionSamplingBasic -v
```

### Run Specific Test
```bash
pytest tests/test_rejection_sampling.py::TestRejectionSamplingBasic::test_returns_correct_shape -v
```

### Run with Coverage
```bash
pytest tests/ --cov=kmeans_seeding --cov-report=html --cov-report=term
```

### Run Fast Tests Only (skip slow tests)
```bash
pytest tests/ -v -m "not slow"
```

## Test Organization

### Test Classes

Each test file is organized into test classes by functionality:

- **TestBasic**: Basic functionality tests
- **TestReproducibility**: Random state and reproducibility
- **TestEdgeCases**: Boundary conditions (k=1, k=n, etc.)
- **TestParameters**: Different parameter values
- **TestInputValidation**: Error handling
- **TestIntegration**: Integration with sklearn

### Test Naming Convention

- `test_<functionality>` - General test
- `test_<method>_<aspect>` - Specific aspect of a method
- `test_invalid_<input>_<raises_error>` - Error handling
- `test_<comparison>_<methods>` - Comparison tests

## Test Coverage

Target: **>80% code coverage**

### Current Coverage Areas

✅ **Core Functionality** (100%)
- All seeding methods
- Return value shapes
- Data type handling
- Reproducibility

✅ **Edge Cases** (100%)
- k=1 (single cluster)
- k=n (all points)
- Small and large k values

✅ **Input Validation** (100%)
- Invalid n_clusters (negative, zero, too large)
- Invalid array shapes (1D, 3D)
- Data type conversion

✅ **Integration** (100%)
- sklearn KMeans compatibility
- Convergence behavior

✅ **API Consistency** (100%)
- All methods accept random_state
- All return numpy arrays
- All validate input similarly

⚠️ **Performance** (Partial)
- Basic performance tests included
- Full benchmark suite in benchmarks/

⚠️ **C++ Extension** (Partial)
- Tests work with and without extension
- C++ unit tests in cpp/tests/

## Test Requirements

### Required Packages
```bash
pip install pytest pytest-cov numpy scikit-learn
```

### Optional for Full Testing
```bash
pip install pytest-xdist  # Parallel testing
pip install pytest-benchmark  # Performance testing
```

## Continuous Integration

Tests run automatically on:
- Every push to main/develop
- Every pull request
- Multiple Python versions (3.9, 3.10, 3.11, 3.12)
- Multiple platforms (Linux, macOS, Windows)

See `.github/workflows/tests.yml` for CI configuration.

## Writing New Tests

### Template for Algorithm Test

```python
def test_<algorithm>_<aspect>(self, <fixture>):
    """Test that <algorithm> <description>."""
    from kmeans_seeding import <algorithm>

    # Setup
    n_clusters = 5

    # Execute
    centers = <algorithm>(<fixture>, n_clusters=n_clusters)

    # Assert
    assert centers.shape == (n_clusters, <fixture>.shape[1])
    assert not np.any(np.isnan(centers))
```

### Adding New Fixtures

Add to `conftest.py`:

```python
@pytest.fixture
def custom_dataset():
    """Description of dataset."""
    # Generate or load data
    return data.astype(np.float32)
```

## Test Statistics

- **Total Test Files**: 8
- **Total Test Classes**: ~40
- **Total Test Functions**: ~150
- **Test Coverage**: Target >80%
- **Estimated Runtime**: ~30 seconds (without slow tests)

## Debugging Tests

### Run with Verbose Output
```bash
pytest tests/ -vv
```

### Stop on First Failure
```bash
pytest tests/ -x
```

### Run Last Failed Tests
```bash
pytest tests/ --lf
```

### See Print Statements
```bash
pytest tests/ -s
```

### Debug with pdb
```bash
pytest tests/ --pdb
```

## Known Issues

1. **FAISS Dependency**: Some tests may be skipped if FAISS is not installed
2. **Random Variations**: Very rare test failures due to random initialization
3. **Memory Tests**: May fail on systems with limited memory

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure >80% coverage
3. Test edge cases
4. Test integration with sklearn
5. Add docstrings to tests
6. Run full test suite before submitting PR

## Questions?

See main documentation or open an issue on GitHub.
