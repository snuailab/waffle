# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Waffle'
copyright = '2023, ZeroAct'
author = 'ZeroAct'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']


# -- Options for Path setup --------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath('../waffle_hub'))
sys.path.insert(0, os.path.abspath('../waffle_utils'))
sys.path.insert(0, os.path.abspath('../waffle_menu'))

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# Docstring Format : google style

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinxcontrib.katex',
    'sphinx.ext.autosectionlabel',
    'sphinx_copybutton',
    'sphinx_panels',
    'myst_parser',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_material'

html_theme_options = {
    # Set the name of the project to appear in the navigation.
    'nav_title': 'Waffle',

    # Set you GA account ID to enable tracking
    'google_analytics_account': 'UA-XXXXX',

    # Specify a base_url used to generate sitemap.xml. If not
    # specified, then no sitemap will be built.
    # 'base_url': 'https://project.github.io/project',

    # Set the color and the accent color
    'color_primary': 'orange',
    'color_accent': 'light-orange',

    # Set the repo location to get a badge with stats
    'repo_url': 'https://github.com/snuailab/waffle',
    'repo_name': 'waffle',

    # Visible levels of the global TOC; -1 means unlimited
    'globaltoc_depth': 3,
    # If False, expand all TOC entries
    'globaltoc_collapse': True,
    # If True, show hidden TOC entries
    'globaltoc_includehidden': True,
}

html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
}

html_static_path = ['_static']
html_favicon = '_static/icons/waffle_icon.png'

add_module_names = False
