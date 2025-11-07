# Publishing Guide: PyPI and Read the Docs

Complete guide for publishing kmeans-seeding to both PyPI and Read the Docs.

---

## 📦 Part 1: Publishing to PyPI

### Prerequisites

1. **PyPI Account**
   - Sign up at: https://pypi.org/account/register/
   - Enable 2FA (required)
   - Create API token: https://pypi.org/manage/account/token/

2. **Install Build Tools**
   ```bash
   pip install --upgrade build twine
   ```

### Step 1: Prepare the Package

1. **Update Version Number**

   Edit `pyproject.toml`:
   ```toml
   [project]
   version = "0.2.2"  # Update this
   ```

2. **Update Changelog**

   Edit `docs_sphinx/changelog.rst`:
   ```rst
   Version 0.2.2 (November 2025)
   -----------------------------

   **New Features**
   - Comprehensive Read the Docs documentation
   - FastLSH optimizations (20-40% faster)
   - Fixed critical hash collision bug

   **Documentation**
   - Complete algorithm documentation
   - API reference with examples
   - User guides and tutorials
   ```

3. **Verify README.md**

   Ensure README.md has:
   - Clear description
   - Installation instructions
   - Quick example
   - Link to documentation
   - Badges (optional)

   Example additions:
   ```markdown
   [![PyPI version](https://badge.fury.io/py/kmeans-seeding.svg)](https://pypi.org/project/kmeans-seeding/)
   [![Documentation Status](https://readthedocs.org/projects/kmeans-seeding/badge/?version=latest)](https://kmeans-seeding.readthedocs.io/)
   ```

4. **Clean Previous Builds**
   ```bash
   rm -rf build/ dist/ *.egg-info
   ```

### Step 2: Build the Package

**Option A: Source Distribution + Wheels (Recommended)**

```bash
# Build source distribution
python3 -m build --sdist

# Build wheel for current platform
python3 -m build --wheel
```

**Option B: Using cibuildwheel (For Multiple Platforms)**

```bash
# Install cibuildwheel
pip install cibuildwheel

# Build wheels for all platforms
cibuildwheel --platform linux
cibuildwheel --platform macos
cibuildwheel --platform windows
```

This creates wheels for:
- Linux: x86_64, aarch64 (manylinux2014)
- macOS: x86_64, arm64
- Windows: x86_64

**Verify Builds:**
```bash
ls -lh dist/
```

You should see:
```
kmeans_seeding-0.2.2.tar.gz           # Source distribution
kmeans_seeding-0.2.2-cp311-*.whl      # Platform wheel(s)
```

### Step 3: Test Upload (TestPyPI)

1. **Create TestPyPI Account**
   - https://test.pypi.org/account/register/

2. **Upload to TestPyPI**
   ```bash
   python3 -m twine upload --repository testpypi dist/*
   ```

   Enter your TestPyPI credentials or API token.

3. **Test Installation**
   ```bash
   # Create test environment
   python3 -m venv test_env
   source test_env/bin/activate

   # Install from TestPyPI
   pip install --index-url https://test.pypi.org/simple/ kmeans-seeding

   # Test it works
   python3 -c "from kmeans_seeding import rskmeans; import numpy as np; X = np.random.randn(100, 10); centers = rskmeans(X, 5); print('Success!')"

   deactivate
   rm -rf test_env
   ```

### Step 4: Publish to PyPI

**⚠️ This is permanent - double-check everything!**

```bash
# Upload to PyPI
python3 -m twine upload dist/*
```

Enter your PyPI API token:
- Username: `__token__`
- Password: `pypi-...` (your API token)

**Verify:**
```bash
# Wait 2-3 minutes for PyPI to update
pip install --upgrade kmeans-seeding

# Test
python3 -c "from kmeans_seeding import rskmeans; print('Published successfully!')"
```

### Step 5: Create GitHub Release

1. **Tag the Release**
   ```bash
   git tag v0.2.2
   git push origin v0.2.2
   ```

2. **Create Release on GitHub**
   - Go to: https://github.com/YOUR-USERNAME/kmeans-seeding/releases/new
   - Select tag: `v0.2.2`
   - Title: `v0.2.2 - FastLSH Optimizations & Documentation`
   - Description: Copy from changelog
   - Attach dist files (optional)
   - Click "Publish release"

---

## 📚 Part 2: Publishing to Read the Docs

### Step 1: Sign Up for Read the Docs

1. **Create Account**
   - Go to: https://readthedocs.org/accounts/signup/
   - Sign in with GitHub (recommended)

2. **Grant Repository Access**
   - Allow Read the Docs to access your GitHub repositories
   - Or import manually

### Step 2: Import Project

1. **Import Repository**
   - Go to: https://readthedocs.org/dashboard/import/
   - Click "Import a Repository"
   - Select `kmeans-seeding` from the list
   - Or manually enter: `https://github.com/YOUR-USERNAME/kmeans-seeding`

2. **Project Configuration**
   - Name: `kmeans-seeding`
   - Repository URL: (auto-filled)
   - Default branch: `main`
   - Language: `en`

3. **Click "Next"**

### Step 3: Configure Build

Read the Docs will automatically detect `.readthedocs.yaml`:

```yaml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.11"

sphinx:
  configuration: docs_sphinx/conf.py

python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - docs
```

**Manual Configuration (if needed):**

1. Go to: Admin → Settings
2. Set:
   - Programming Language: `Python`
   - Documentation type: `Sphinx Html`
   - Sphinx configuration file: `docs_sphinx/conf.py`

### Step 4: Build Documentation

1. **Trigger Build**
   - Go to: Builds
   - Click "Build Version"
   - Select "latest"
   - Wait 2-3 minutes

2. **Check Build Status**
   - Green checkmark = success ✅
   - Red X = failed ❌

3. **View Documentation**
   - Click "View Docs"
   - URL: `https://kmeans-seeding.readthedocs.io/en/latest/`

### Step 5: Enable Automatic Builds

**Webhook Setup (Automatic):**

Read the Docs should automatically create a GitHub webhook. Verify:

1. Go to GitHub repository → Settings → Webhooks
2. You should see: `https://readthedocs.org/api/v2/webhook/...`
3. Recent Deliveries should show successful pings

**If webhook is missing:**

1. Go to Read the Docs: Admin → Integrations
2. Click "Add integration"
3. Select "GitHub incoming webhook"
4. Copy the webhook URL
5. Add to GitHub repository webhooks

Now every push to `main` will automatically rebuild docs!

### Step 6: Configure Versions

1. **Activate Versions**
   - Go to: Admin → Versions
   - Activate:
     - `latest` (always latest commit)
     - `stable` (latest tag)
     - Specific tags (e.g., `v0.2.2`)

2. **Set Default Version**
   - Go to: Admin → Advanced Settings
   - Default version: `stable`
   - Save

### Step 7: Custom Domain (Optional)

To use `docs.yourdomain.com`:

1. **Add Domain in Read the Docs**
   - Go to: Admin → Domains
   - Add domain: `docs.yourdomain.com`

2. **Configure DNS**
   Add CNAME record:
   ```
   docs.yourdomain.com → kmeans-seeding.readthedocs.io
   ```

3. **Enable HTTPS**
   - Read the Docs will automatically provision SSL certificate
   - Takes 5-15 minutes

---

## 🚀 Part 3: Automated Publishing (GitHub Actions)

Create `.github/workflows/publish.yml`:

```yaml
name: Publish Package

on:
  release:
    types: [published]

jobs:
  build-and-publish:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine

    - name: Build package
      run: python -m build

    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

**Setup:**

1. Add PyPI API token to GitHub Secrets:
   - Go to: Repository → Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `PYPI_API_TOKEN`
   - Value: Your PyPI API token
   - Save

2. Now when you create a GitHub release, it automatically publishes to PyPI!

---

## ✅ Verification Checklist

### Before Publishing

- [ ] Version number updated in `pyproject.toml`
- [ ] Changelog updated
- [ ] README.md is current
- [ ] All tests pass: `pytest tests/`
- [ ] Documentation builds: `cd docs_sphinx && make html`
- [ ] Git committed and pushed
- [ ] No uncommitted changes

### After Publishing to PyPI

- [ ] Package appears on PyPI: https://pypi.org/project/kmeans-seeding/
- [ ] Can install: `pip install kmeans-seeding`
- [ ] Import works: `from kmeans_seeding import rskmeans`
- [ ] Version correct: `pip show kmeans-seeding`

### After Publishing to Read the Docs

- [ ] Documentation builds successfully
- [ ] Accessible at: https://kmeans-seeding.readthedocs.io/
- [ ] Search works
- [ ] All pages render correctly
- [ ] API documentation generated
- [ ] Examples run correctly
- [ ] Webhook configured (auto-rebuild on push)

### Post-Publication

- [ ] GitHub release created with tag
- [ ] PyPI link in README
- [ ] Read the Docs badge in README
- [ ] Social media announcement (optional)
- [ ] Update personal portfolio (optional)

---

## 🔧 Troubleshooting

### PyPI Upload Fails

**Error: "Invalid distribution"**
```bash
# Rebuild cleanly
rm -rf build/ dist/ *.egg-info
python3 -m build
```

**Error: "File already exists"**
```bash
# You can't re-upload same version
# Increment version in pyproject.toml
# Or delete the release from PyPI (within 24 hours)
```

**Error: "Invalid credentials"**
```bash
# Use API token, not password
# Username: __token__
# Password: pypi-...
```

### Read the Docs Build Fails

**Error: "Sphinx not found"**
- Check `.readthedocs.yaml` has correct Python version
- Ensure `docs` extra_requires includes Sphinx

**Error: "Module not found"**
- Add to `conf.py`:
  ```python
  autodoc_mock_imports = ['_core', 'faiss']
  ```

**Error: "Build timeout"**
- Simplify documentation
- Remove large assets
- Contact Read the Docs support

### Import Errors After Install

**Error: "No module named '_core'"**
- C++ extension didn't build
- Install build dependencies
- Try: `pip install --no-binary :all: kmeans-seeding`

**Error: "FAISS not available"**
- This is expected - FAISS is optional
- Either install FAISS or use FastLSH index
- See: https://kmeans-seeding.readthedocs.io/en/latest/user_guide/installation.html

---

## 📊 Post-Publication Monitoring

### PyPI Statistics

- Download stats: https://pypistats.org/packages/kmeans-seeding
- Monitor via: https://pypi.org/project/kmeans-seeding/

### Read the Docs Analytics

- Go to: Admin → Analytics
- View:
  - Page views
  - Search queries
  - Top pages
  - Traffic sources

### GitHub Insights

- Stars, forks, watchers
- Clone statistics
- Dependency graph

---

## 🎯 Quick Publishing Workflow

For regular updates:

```bash
# 1. Update version
vim pyproject.toml  # Increment version

# 2. Update changelog
vim docs_sphinx/changelog.rst

# 3. Commit
git add .
git commit -m "Release v0.2.3"
git push

# 4. Build
rm -rf dist/
python3 -m build

# 5. Test (optional)
python3 -m twine upload --repository testpypi dist/*

# 6. Publish
python3 -m twine upload dist/*

# 7. Tag
git tag v0.2.3
git push origin v0.2.3

# 8. GitHub Release
# Go to GitHub and create release from tag

# 9. Verify
# Check PyPI: https://pypi.org/project/kmeans-seeding/
# Check docs: https://kmeans-seeding.readthedocs.io/
```

Done! 🎉

---

## 📞 Support

- PyPI Support: https://pypi.org/help/
- Read the Docs Support: https://docs.readthedocs.io/
- GitHub Actions: https://docs.github.com/en/actions

Good luck with your publication! 🚀
