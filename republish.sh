#!/bin/bash
# Quick script to republish kmeans-seeding to PyPI with the metadata fix

set -e  # Exit on error

echo "================================================"
echo "Republishing kmeans-seeding v0.2.2 to PyPI"
echo "================================================"
echo ""

# Navigate to project root
cd "$(dirname "$0")"

echo "Step 1: Cleaning old build artifacts..."
rm -rf build/ dist/ *.egg-info python/*.egg-info
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "✓ Cleaned"
echo ""

echo "Step 2: Installing/upgrading build tools..."
pip3 install --upgrade build twine
echo "✓ Tools ready"
echo ""

echo "Step 3: Building package..."
python3 -m build
echo "✓ Package built"
echo ""

echo "Step 4: Verifying package metadata..."
echo "Contents of dist/:"
ls -lh dist/
echo ""

echo "Checking PKG-INFO for correct name..."
tar -xzf dist/kmeans_seeding-0.2.2.tar.gz -O kmeans_seeding-0.2.2/PKG-INFO | head -10
echo ""

read -p "Does the metadata show 'Name: kmeans-seeding' (not 'unknown')? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "❌ Metadata still incorrect. Please check setup.py"
    exit 1
fi
echo ""

echo "Step 5: Testing local installation..."
python3 -m venv test_install_tmp
source test_install_tmp/bin/activate
pip install dist/kmeans_seeding-0.2.2.tar.gz > /dev/null 2>&1
python3 -c "import kmeans_seeding; print('Version:', kmeans_seeding.__version__); from kmeans_seeding import rskmeans; print('✓ Import successful')"
deactivate
rm -rf test_install_tmp
echo "✓ Local test passed"
echo ""

echo "Step 6: Ready to upload to PyPI"
echo ""
echo "Choose upload destination:"
echo "  1) Test PyPI (recommended first)"
echo "  2) Production PyPI"
echo "  3) Skip upload (just build)"
read -p "Enter choice (1/2/3): " choice

case $choice in
    1)
        echo ""
        echo "Uploading to Test PyPI..."
        python3 -m twine upload --repository testpypi dist/*
        echo ""
        echo "✓ Uploaded to Test PyPI"
        echo ""
        echo "Test installation with:"
        echo "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ kmeans-seeding"
        ;;
    2)
        echo ""
        echo "Uploading to Production PyPI..."
        python3 -m twine upload dist/*
        echo ""
        echo "✓ Uploaded to PyPI"
        echo ""
        echo "Wait 2-3 minutes, then test with:"
        echo "  pip install --upgrade kmeans-seeding"
        echo "  python3 -c 'import kmeans_seeding; print(kmeans_seeding.__version__)'"
        ;;
    3)
        echo "Skipping upload. Package is ready in dist/"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "================================================"
echo "Done! 🎉"
echo "================================================"
