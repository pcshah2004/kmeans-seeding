# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.abspath('../python'))

# -- Project information -----------------------------------------------------
project = 'kmeans-seeding'
copyright = '2025, Poojan Shah'
author = 'Poojan Shah'
release = '0.2.2'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx_copybutton',
    'sphinx_rtd_theme',
]

# Napoleon settings (for NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_mock_imports = ['_core']

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
    'display_version': True,
}

# Logo and favicon (if you have them)
# html_logo = '_static/logo.png'
# html_favicon = '_static/favicon.ico'

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    'preamble': r'''
\usepackage{amsmath}
\usepackage{amssymb}
''',
}
