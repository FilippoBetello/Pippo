# conf.py - Sphinx configuration for Read the Docs

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath(os.path.join('..', 'easy_rec')))

# -- Project information -----------------------------------------------------

project = 'Your Project Name'
copyright = '2025, Your Name'
author = 'Your Name'

release = '0.1.0'
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',        # Automatically document from docstrings
    'sphinx.ext.napoleon',       # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',       # Add links to source code
    'sphinx.ext.githubpages',    # Publish .nojekyll file for GitHub Pages
    'sphinx.ext.todo',           # Support for todo directives
    'sphinx.ext.mathjax',        # Support for LaTeX-style math
    'sphinx.ext.intersphinx',    # Link to other project's documentation
]

# -- Path setup --------------------------------------------------------------

templates_path = ['_templates']
exclude_patterns = []

# -- HTML output -------------------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Extension configurations ------------------------------------------------

# Napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Autodoc
autoclass_content = 'class'
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

# Todo
todo_include_todos = True

# Math
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'

