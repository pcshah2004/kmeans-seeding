# Fix PyPI Package Installation Issue

## Problem Identified

The PyPI package `kmeans-seeding` versions 0.1.0, 0.2.0, and 0.2.1 all have a **critical metadata bug** that prevents installation:

```
WARNING: Generating metadata for package kmeans-seeding produced metadata for project name unknown.
```

The package metadata shows `name: "unknown"` instead of `name: "kmeans-seeding"`, causing pip to reject all versions.

## Root Cause

In `setup.py` line 138, the `setup()` function was called without the `name` parameter:

```python
# BEFORE (BROKEN)
setup(
    version=get_version(),
    ext_modules=[...],
    ...
)
```

Even though `pyproject.toml` specifies `name = "kmeans-seeding"`, the `setup.py` file overrides this when building the source distribution, and without an explicit `name` parameter, setuptools defaults to "unknown".

## Fix Applied

**File: `setup.py` line 138**

```python
# AFTER (FIXED)
setup(
    name="kmeans-seeding",  # <-- ADDED THIS LINE
    version=get_version(),
    ext_modules=[...],
    ...
)
```

## Steps to Republish (Version 0.2.2)

### 1. Update Version Number

Edit `python/kmeans_seeding/__init__.py`:

```python
__version__ = "0.2.2"  # Changed from 0.2.1
```

### 2. Clean Previous Builds

```bash
cd "/Users/poojanshah/Desktop/Fast k means++"

# Remove old build artifacts
rm -rf build/
rm -rf dist/
rm -rf *.egg-info
rm -rf python/*.egg-info
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
```

### 3. Install Build Tools (if not already installed)

```bash
pip3 install --upgrade build twine
```

### 4. Build the Package

```bash
# Build source distribution and wheel
python3 -m build

# Verify the built packages
ls -lh dist/
```

Expected output:
```
kmeans_seeding-0.2.2.tar.gz
kmeans_seeding-0.2.2-*.whl  (platform-specific wheel)
```

### 5. Test Locally Before Publishing

```bash
# Create a fresh test environment
python3 -m venv test_install_venv
source test_install_venv/bin/activate

# Install from local build
pip install dist/kmeans_seeding-0.2.2.tar.gz

# Test the installation
python3 -c "import kmeans_seeding; print(kmeans_seeding.__version__)"
python3 -c "from kmeans_seeding import rskmeans; print('Import successful!')"

# Deactivate and remove test venv
deactivate
rm -rf test_install_venv
```

### 6. Check Package Metadata

Before uploading, verify the metadata is correct:

```bash
tar -tzf dist/kmeans_seeding-0.2.2.tar.gz | grep PKG-INFO
tar -xzf dist/kmeans_seeding-0.2.2.tar.gz -O kmeans_seeding-0.2.2/PKG-INFO | head -20
```

You should see:
```
Name: kmeans-seeding
Version: 0.2.2
...
```

**NOT** `Name: unknown`

### 7. Upload to Test PyPI (Recommended)

```bash
# Upload to Test PyPI first
python3 -m twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip3 install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ kmeans-seeding
```

### 8. Upload to PyPI (Production)

```bash
# Upload to production PyPI
python3 -m twine upload dist/*
```

You'll be prompted for your PyPI credentials or API token.

### 9. Verify Installation Works

```bash
# In a fresh environment, test installation
pip3 install --upgrade kmeans-seeding

# Verify
python3 -c "import kmeans_seeding; print(kmeans_seeding.__version__)"
```

## Additional Recommendations

### Add to pyproject.toml

To prevent this issue in the future, you can also add version to pyproject.toml dynamically. However, the current setup with `setup.py` reading from `__init__.py` is fine as long as `name` is specified.

### Update MANIFEST.in

The current `MANIFEST.in` looks good, but verify all necessary files are included:

```bash
# After building, check what's in the source distribution
tar -tzf dist/kmeans_seeding-0.2.2.tar.gz | less
```

Ensure it includes:
- All `.cpp`, `.cc`, `.c` files from `cpp/src/`
- All `.h`, `.hpp` files from `cpp/include/`
- `CMakeLists.txt` and `.cmake` files
- `README.md`, `LICENSE`, `CLAUDE.md`
- Python source files from `python/kmeans_seeding/`

## Summary of Changes

1. ✅ **Fixed**: Added `name="kmeans-seeding"` to `setup.py`
2. ⚠️ **TODO**: Bump version to 0.2.2 in `__init__.py`
3. ⚠️ **TODO**: Rebuild and republish package
4. ⚠️ **TODO**: Consider yanking versions 0.1.0, 0.2.0, 0.2.1 from PyPI

## Yanking Old Versions (Optional)

You can "yank" the broken versions from PyPI (they'll still exist but won't be installed by default):

```bash
# Log into PyPI web interface at https://pypi.org/manage/project/kmeans-seeding/
# Or use twine (requires PyPI account with maintainer access):
# This marks them as "unavailable" but keeps them for existing users
```

Navigate to: https://pypi.org/manage/project/kmeans-seeding/releases/
And click "Options" → "Yank release" for versions 0.1.0, 0.2.0, and 0.2.1.

## Contact

If you encounter issues during republishing:
- Check build logs carefully
- Verify CMake finds all dependencies (FAISS, OpenMP)
- Test in a clean virtual environment
- Ensure you have PyPI upload permissions
