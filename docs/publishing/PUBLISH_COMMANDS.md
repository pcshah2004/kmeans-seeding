# 🚀 PyPI Publication Commands

Your package is **ready to publish**! Run these commands in your terminal.

## ✅ Package Status

- ✅ Code on GitHub: https://github.com/pcshah2004/kmeans-seeding
- ✅ Distributions built and validated
- ✅ Ready for PyPI upload

## 📦 Files to Upload

Located in `dist/`:
- `kmeans_seeding-0.1.0-cp313-cp313-macosx_14_0_arm64.whl` (135 KB)
- `kmeans_seeding-0.1.0.tar.gz` (49 KB)

---

## Option 1: Upload to TestPyPI (Recommended First)

### Step 1: Get TestPyPI API Token

1. Create account at: https://test.pypi.org/account/register/
2. Verify your email
3. Create API token at: https://test.pypi.org/manage/account/token/
   - Token name: "kmeans-seeding upload"
   - Scope: "Entire account" (you can restrict after first upload)
4. Copy the token (starts with `pypi-`)

### Step 2: Upload to TestPyPI

Run these commands in your terminal:

```bash
cd "/Users/poojanshah/Desktop/Fast k means++"
source build_venv/bin/activate

# Upload (you'll be prompted for username and password)
twine upload --repository testpypi dist/*

# When prompted:
# Username: __token__
# Password: pypi-... (paste your token)
```

### Step 3: Test Installation from TestPyPI

```bash
# Create fresh test environment
python3 -m venv test_pypi_install
source test_pypi_install/bin/activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple kmeans-seeding

# Test it works
python -c "from kmeans_seeding import kmeanspp; import numpy as np; X = np.random.randn(100, 10).astype(np.float64); print('✓ Got', kmeanspp(X, n_clusters=5).shape, 'centers'); print('✓ Package works!')"
```

If the test passes, proceed to publish to real PyPI!

---

## Option 2: Upload to PyPI (Production)

### Step 1: Get PyPI API Token

1. Create account at: https://pypi.org/account/register/
2. Verify your email
3. Create API token at: https://pypi.org/manage/account/token/
   - Token name: "kmeans-seeding upload"
   - Scope: "Entire account" (you can restrict after first upload)
4. Copy the token (starts with `pypi-`)

### Step 2: Upload to PyPI

Run these commands in your terminal:

```bash
cd "/Users/poojanshah/Desktop/Fast k means++"
source build_venv/bin/activate

# Upload (you'll be prompted for username and password)
twine upload dist/*

# When prompted:
# Username: __token__
# Password: pypi-... (paste your token)
```

### Step 3: Verify Publication

1. Check package page: https://pypi.org/project/kmeans-seeding/
2. Install it:
   ```bash
   pip install kmeans-seeding
   ```
3. Test it:
   ```python
   from kmeans_seeding import rejection_sampling
   import numpy as np

   X = np.random.randn(1000, 50).astype(np.float64)
   centers = rejection_sampling(X, n_clusters=10, index_type='LSH')
   print(f"✓ Got {centers.shape[0]} centers!")
   print("✓ kmeans-seeding is live on PyPI!")
   ```

---

## Alternative: Use .pypirc File (Saves Typing)

Create a file `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_PYPI_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TESTPYPI_TOKEN_HERE
```

Then you can upload without being prompted:

```bash
# To TestPyPI
twine upload --repository testpypi dist/*

# To PyPI
twine upload dist/*
```

**Security Note:** Keep your `.pypirc` file secure (`chmod 600 ~/.pypirc`)

---

## 🎉 After Publishing

### 1. Update README Badges

The PyPI badge will now work:
- PyPI page: https://pypi.org/project/kmeans-seeding/

### 2. Create GitHub Release

1. Go to: https://github.com/pcshah2004/kmeans-seeding/releases/new
2. Tag: `v0.1.0`
3. Title: `kmeans-seeding v0.1.0 - Initial Release`
4. Description: See NEXT_STEPS.md for template
5. Attach: `dist/kmeans_seeding-0.1.0.tar.gz`

### 3. Share Your Achievement!

**Twitter/LinkedIn Post:**
```
🚀 Just published kmeans-seeding v0.1.0 to PyPI!

Fast k-means++ initialization algorithms with C++ core & Python bindings.

⚡ 5 algorithms including our RS-k-means++ (0.90x-1.16x vs standard k-means++)
📦 pip install kmeans-seeding
🔗 https://github.com/pcshah2004/kmeans-seeding
📄 Paper: arXiv:2502.02085

#MachineLearning #Python #Clustering #OpenSource
```

### 4. Install and Use Your Package!

```bash
pip install kmeans-seeding

python << 'EOF'
from kmeans_seeding import rejection_sampling
from sklearn.cluster import KMeans
import numpy as np

# Your published algorithm in action!
X = np.random.randn(10000, 50)
centers = rejection_sampling(X, n_clusters=100, index_type='LSH')

kmeans = KMeans(n_clusters=100, init=centers, n_init=1)
kmeans.fit(X)
print(f"✓ Clustering complete! Used {len(centers)} initial centers")
print("✓ Your package is live and working!")
EOF
```

---

## 📊 Quick Reference

| Item | Link |
|------|------|
| **GitHub Repo** | https://github.com/pcshah2004/kmeans-seeding |
| **TestPyPI** | https://test.pypi.org/project/kmeans-seeding/ |
| **PyPI** | https://pypi.org/project/kmeans-seeding/ |
| **TestPyPI Token** | https://test.pypi.org/manage/account/token/ |
| **PyPI Token** | https://pypi.org/manage/account/token/ |
| **GitHub Release** | https://github.com/pcshah2004/kmeans-seeding/releases/new |

---

## 🆘 Troubleshooting

**If upload fails with "Invalid or non-existent authentication":**
- Make sure you're using `__token__` (with underscores) as username
- Make sure you copied the entire token (starts with `pypi-`)

**If upload fails with "File already exists":**
- Package version 0.1.0 already uploaded
- Either increment version in `pyproject.toml` and rebuild, or delete the existing release

**If upload fails with "403 Forbidden":**
- Verify your email on PyPI
- Make sure the token has upload permissions

---

## ✅ Ready to Publish!

Just open a terminal and run the commands above. Your package is ready to go! 🚀

**Recommended Order:**
1. Upload to TestPyPI first
2. Test installation from TestPyPI
3. If all works, upload to PyPI
4. Create GitHub release
5. Celebrate! 🎉
