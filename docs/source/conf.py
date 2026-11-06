# Configuration file for the Sphinx documentation builder.

import os
import sys
import inspect

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
project = 'GraphToolbox'
copyright = '2025–2026, Eloi Campagne'
author = 'Eloi Campagne'
version = '0.3'
release = '0.3.1'

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

# Copy buttons skip shell and Python prompts.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

autosummary_generate = True
# Member documentation is driven by the autosummary module template
# (_templates/autosummary/module.rst): leaf modules document their members in
# full, whereas re-exporting subpackages only list their submodules. This keeps
# each object documented once and avoids ambiguous cross-references.
autodoc_default_options = {}


templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ['_static']
html_css_files = [
    "https://fonts.bunny.net/css?family=crimson-pro:400,400i,600,700",
    "graphtoolbox.css",
]
html_title = f"GraphToolbox {release}"

html_theme_options = {
    "announcement": (
        f"GraphToolbox {release}: MinT with a shrunk error covariance, daily online updates, "
        "attention matched to edges. "
        "<a href='https://github.com/EDF-Lab/GraphToolbox/blob/main/CHANGELOG.md'>Changelog</a>"
    ),
    # Palette of eloicampagne.fr: warm paper, ink, olive links, brown accents; a warm dark variant.
    "light_css_variables": {
        "font-stack": '"Crimson Pro", Georgia, serif',
        "font-stack--headings": '"Crimson Pro", Georgia, serif',
        "color-foreground-primary": "#2E2E2E",
        "color-foreground-secondary": "rgba(46, 46, 46, 0.68)",
        "color-background-primary": "#F5F0E7",
        "color-background-secondary": "#EFE7DA",
        "color-background-hover": "rgba(139, 94, 60, 0.08)",
        "color-background-border": "rgba(139, 94, 60, 0.18)",
        "color-brand-primary": "#8B5E3C",
        "color-brand-content": "#607D3B",
        "color-brand-visited": "#607D3B",
        "color-sidebar-background": "#EFE7DA",
        "color-sidebar-link-text--top-level": "#8B5E3C",
        "color-code-background": "#FFFDF9",
        "color-inline-code-background": "rgba(139, 94, 60, 0.06)",
        "color-highlighted-background": "rgba(96, 125, 59, 0.15)",
        "color-announcement-background": "#8B5E3C",
        "color-announcement-text": "#F5F0E7",
        "color-api-name": "#8B5E3C",
        "color-api-pre-name": "#607D3B",
    },
    "dark_css_variables": {
        "font-stack": '"Crimson Pro", Georgia, serif',
        "font-stack--headings": '"Crimson Pro", Georgia, serif',
        "color-foreground-primary": "#E9E2D6",
        "color-foreground-secondary": "rgba(233, 226, 214, 0.7)",
        "color-background-primary": "#1E1A15",
        "color-background-secondary": "#191510",
        "color-background-hover": "rgba(201, 148, 106, 0.12)",
        "color-background-border": "rgba(201, 148, 106, 0.22)",
        "color-brand-primary": "#C9946A",
        "color-brand-content": "#A3BE78",
        "color-brand-visited": "#A3BE78",
        "color-sidebar-background": "#191510",
        "color-sidebar-link-text--top-level": "#C9946A",
        "color-code-background": "#26211B",
        "color-inline-code-background": "rgba(201, 148, 106, 0.1)",
        "color-highlighted-background": "rgba(163, 190, 120, 0.2)",
        "color-announcement-background": "#5E3F28",
        "color-announcement-text": "#F5F0E7",
        "color-api-name": "#C9946A",
        "color-api-pre-name": "#A3BE78",
    },
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
