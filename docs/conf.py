import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'SEAM'
copyright = '2023-2025, Evan Seitz, David McCandlish, Justin Kinney, Peter Koo'
author = 'Evan Seitz'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
    'myst_parser'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/seam_logo_light.png'
html_theme_options = {
    'logo_only': True,
    'display_version': True,
}

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
} 