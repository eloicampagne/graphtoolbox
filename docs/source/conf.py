# Configuration file for the Sphinx documentation builder.

import os
import sys
import inspect

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
project = 'GraphToolbox'
copyright = '2025, Eloi Campagne'
author = 'Eloi Campagne'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.linkcode",
    "myst_parser",
]

autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}


templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ['_static']

html_theme_options = {
    "light_logo": "logo_light.png",
    "dark_logo": "logo_dark.png",
    "navigation_with_keys": True,
    "sidebar_hide_name": False,
    "footer_icons": [
        {
            "name": "Home",
            "url": "https://eloicampagne.fr",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" 
                    viewBox="0 0 24 24" height="1.2em" width="1.2em">
                    <path d="M12 3l9 8h-3v9h-4v-6H10v6H6v-9H3z"></path>
                </svg>
            """,
            "class": "",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/eloicampagne/GraphToolbox",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

html_show_sourcelink = False

# -- Autodoc config ----------------------------------------------------------
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

suppress_warnings = ["autodoc.noindex"]

# -- Link to GitHub source code ----------------------------------------------
# ⚙️ Change the repository URL below if needed
github_user = "eloicampagne"
github_repo = "GraphToolbox"
github_branch = "main"  # or "master" if applicable

def linkcode_resolve(domain, info):
    """
    Return the GitHub URL corresponding to the object being documented.
    """
    if domain != 'py' or not info['module']:
        return None

    try:
        module = sys.modules[info['module']]
        obj = module
        for part in info['fullname'].split('.'):
            obj = getattr(obj, part)
        # Try to get the source file and line numbers
        fn = inspect.getsourcefile(obj)
        if not fn:
            return None
        fn = os.path.relpath(fn, start=os.path.dirname(__file__) + "/../../")
        source, lineno = inspect.getsourcelines(obj)
        return f"https://github.com/{github_user}/{github_repo}/blob/{github_branch}/{fn}#L{lineno}-L{lineno + len(source) - 1}"
    except Exception:
        return None

def setup(app):
    # Pour un fichier local
    # app.add_css_file("styles.css")
    pass
