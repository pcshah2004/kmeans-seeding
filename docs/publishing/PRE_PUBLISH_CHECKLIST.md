# Pre-Publication Checklist

## ✅ Before Publishing to PyPI

### 1. Code Quality
- [x] All C++ code organized in cpp/
- [x] All Python code in python/kmeans_seeding/
- [x] Python bindings module named `_core`
- [x] No hardcoded paths or secrets
- [ ] All tests passing locally

### 2. Documentation
- [x] README.md complete with examples
- [x] LICENSE file (MIT)
- [x] CITATION.cff for academic citation
- [ ] Update GitHub URLs (replace 'yourusername')
- [ ] Update ORCID IDs in CITATION.cff (optional)

### 3. Metadata
- [x] Version 0.1.0 in all files
- [ ] Update author email if needed
- [ ] Update project URLs in pyproject.toml
- [ ] Add GitHub repository URL

### 4. Build System
- [x] pyproject.toml configured
- [x] setup.py with CMake integration
- [x] CMakeLists.txt for C++ build
- [ ] Test local build: `pip install -e .`

### 5. Dependencies
- [x] Requirements specified in pyproject.toml
- [x] FAISS marked as optional
- [x] Python 3.9+ specified
- [ ] Verify all imports work

### 6. Tests
- [x] 150+ tests written
- [ ] Run tests: `pytest tests/ -v`
- [ ] Check coverage: `pytest --cov`
- [ ] All tests passing (or skipped if no C++ ext)

### 7. GitHub Setup
- [ ] Create GitHub repository
- [ ] Push code to main branch
- [ ] Add topics/tags (clustering, kmeans, python, cpp)
- [ ] Enable GitHub Pages (for docs)
- [ ] Add repository secrets:
  - PYPI_API_TOKEN
  - TEST_PYPI_API_TOKEN (optional)

### 8. CI/CD
- [x] GitHub Actions workflows created
- [ ] Test CI workflows run successfully
- [ ] Wheel builds complete for all platforms
- [ ] Tests pass on CI

### 9. PyPI Test Upload
- [ ] Register on test.pypi.org
- [ ] Get API token
- [ ] Build distributions: `python -m build`
- [ ] Upload to Test PyPI:
  ```bash
  twine upload --repository testpypi dist/*
  ```
- [ ] Test installation:
  ```bash
  pip install --index-url https://test.pypi.org/simple/ kmeans-seeding
  ```

### 10. Production PyPI Upload
- [ ] Register on pypi.org
- [ ] Get API token
- [ ] Create git tag: `git tag v0.1.0`
- [ ] Push tag: `git push origin v0.1.0`
- [ ] GitHub Action will auto-publish (or manual):
  ```bash
  twine upload dist/*
  ```

## Quick Commands

### Test Build Locally
```bash
cd "/Users/poojanshah/Desktop/Fast k means++"

# Clean previous builds
rm -rf build dist *.egg-info

# Install in development mode
pip install -e .

# Test import
python -c "import kmeans_seeding; print(f'Version: {kmeans_seeding.__version__}')"

# Run tests
pytest tests/ -v
```

### Build Distributions
```bash
# Install build tools
pip install build twine

# Build wheel and sdist
python -m build

# Check distribution
twine check dist/*
```

### Upload to Test PyPI
```bash
# First time: create account and get token from test.pypi.org

# Upload
twine upload --repository testpypi dist/*

# Test install
pip install --index-url https://test.pypi.org/simple/ --no-deps kmeans-seeding
python -c "import kmeans_seeding; print('Success!')"
```

### Upload to Production PyPI
```bash
# First time: create account and get token from pypi.org

# Tag release
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0

# Upload (or let GitHub Actions do it)
twine upload dist/*
```

## Known Issues to Fix Before Release

### Critical
- [ ] Test that C++ extension builds on macOS
- [ ] Test that C++ extension builds on Linux (via CI)
- [ ] Verify FAISS optional dependency works

### Important
- [ ] Add actual GitHub repository URL
- [ ] Test README renders correctly on PyPI
- [ ] Verify all example code runs

### Nice to Have
- [ ] Add badges to README (build status, coverage, etc.)
- [ ] Create CHANGELOG.md
- [ ] Add CONTRIBUTING.md

## Post-Publication

### Immediate
- [ ] Test install from PyPI: `pip install kmeans-seeding`
- [ ] Verify package page on pypi.org
- [ ] Check that README renders correctly
- [ ] Test on fresh environment

### Soon After
- [ ] Set up Read the Docs
- [ ] Add Zenodo DOI
- [ ] Announce on relevant forums/communities
- [ ] Write blog post/tutorial

### Later
- [ ] Monitor GitHub issues
- [ ] Respond to user feedback
- [ ] Plan v0.2.0 features

## Contact for Issues

- Email: cs1221594@cse.iitd.ac.in
- GitHub: (to be added)
