# Documentation for kmeans-seeding

This directory contains the Sphinx documentation for the kmeans-seeding package.

## Building Locally

### Prerequisites

```bash
pip install sphinx sphinx-rtd-theme sphinx-copybutton
```

### Build HTML

```bash
cd docs_sphinx
make html
```

The HTML documentation will be in `_build/html/`. Open `_build/html/index.html` in your browser.

### Clean Build

```bash
make clean
make html
```

## Publishing to Read the Docs

### Setup

1. Go to https://readthedocs.org/
2. Sign in with GitHub account
3. Import the `kmeans-seeding` repository
4. Read the Docs will automatically detect `.readthedocs.yaml`

### Configuration

The `.readthedocs.yaml` file in the repository root configures:

- Python version (3.11)
- Build dependencies
- Sphinx configuration path

### Automatic Builds

Read the Docs automatically rebuilds documentation when you:

- Push to the main branch
- Create a new tag/release
- Open a pull request (for PR previews)

### Custom Domain (Optional)

To use a custom domain:

1. Go to project settings on Read the Docs
2. Add your domain under "Domains"
3. Configure DNS CNAME record
4. Enable HTTPS

### Versioning

Read the Docs automatically creates versions for:

- `latest`: Latest commit on main branch
- `stable`: Latest tagged release
- Each tag: Individual versions (e.g., `v0.2.2`)

To activate a version:
1. Go to "Versions" in project settings
2. Toggle "Active" for desired versions

## Documentation Structure

```
docs_sphinx/
├── index.rst                  # Main page
├── conf.py                    # Sphinx configuration
├── Makefile                   # Build commands
├── requirements.txt           # Build dependencies
│
├── user_guide/                # User guides
│   ├── installation.rst
│   ├── quickstart.rst
│   ├── choosing_algorithm.rst
│   └── sklearn_integration.rst
│
├── algorithms/                # Algorithm documentation
│   ├── rskmeans.rst
│   ├── afkmc2.rst
│   ├── fast_lsh.rst
│   ├── kmeanspp.rst
│   └── comparison.rst
│
├── api/                       # API reference
│   ├── initializers.rst
│   └── advanced.rst
│
├── changelog.rst              # Version history
├── contributing.rst           # Contribution guide
└── references.rst             # Academic references
```

## Writing Documentation

### Adding New Pages

1. Create `.rst` file in appropriate directory
2. Add to `toctree` in `index.rst` or parent page
3. Rebuild: `make html`

### reStructuredText Basics

```rst
Title
=====

Section
-------

Subsection
~~~~~~~~~~

**Bold** and *italic*

- Bullet list
- Item 2

1. Numbered list
2. Item 2

.. code-block:: python

   # Code example
   from kmeans_seeding import rskmeans
   centers = rskmeans(X, n_clusters=10)

:doc:`link to other page <path/to/page>`

.. note::
   This is a note box

.. warning::
   This is a warning box
```

### Cross-References

```rst
:func:`rskmeans`                    # Link to function
:class:`KMeans`                     # Link to class
:doc:`../algorithms/rskmeans`       # Link to document
```

### API Documentation

Uses autodoc to extract docstrings:

```rst
.. autofunction:: rskmeans
```

Requires NumPy-style docstrings in source code.

## Troubleshooting

### Build Fails

Check Python path in `conf.py`:

```python
sys.path.insert(0, os.path.abspath('../python'))
```

### Import Errors

If autodoc can't import modules:

```python
# In conf.py
autodoc_mock_imports = ['_core', 'faiss']
```

### Missing Modules

Install with:

```bash
pip install -r requirements.txt
```

### Theme Issues

Ensure sphinx-rtd-theme is installed:

```bash
pip install sphinx-rtd-theme
```

## Continuous Integration

The documentation build is tested in CI:

```yaml
# Example GitHub Actions
- name: Build docs
  run: |
    pip install sphinx sphinx-rtd-theme sphinx-copybutton
    cd docs_sphinx
    make html
```

## Maintenance

### Regular Updates

- Keep examples up-to-date with API changes
- Update version numbers in `conf.py`
- Add entries to `changelog.rst` for each release
- Update benchmark numbers if performance improves

### Link Checking

```bash
make linkcheck
```

This verifies all external links are still valid.

## Resources

- Sphinx documentation: https://www.sphinx-doc.org/
- Read the Docs guide: https://docs.readthedocs.io/
- reStructuredText primer: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
- NumPy docstring guide: https://numpydoc.readthedocs.io/
