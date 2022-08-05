# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import subprocess
import sys

# debug info
print("python exec:", sys.executable)
print("sys.path:", sys.path)
print("environment:", os.environ)

if "conda" in sys.executable:
    print("conda environment:")
    subprocess.run(["conda", "list"])
    subprocess.run(["conda", "info"])

else:
    print("pip environment:")
    subprocess.run([sys.executable, "-m", "pip", "list"])


# hacky for RTD - which doesn't actually call conda activate
# see: https://github.com/readthedocs/readthedocs.org/issues/5339
if os.environ.get("READTHEDOCS", False):
    import cmdstanpy

    path = os.path.join(
        os.environ["CONDA_ENVS_PATH"],
        os.environ["READTHEDOCS_VERSION_NAME"],
        "bin",
    )
    cmdstanpy.set_cmdstan_path(os.path.join(path, "cmdstan", ""))

    os.environ["CXX"] = os.path.join(path, "x86_64-conda-linux-gnu-c++")

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "SciKit Stan"
copyright = "2022, Alexey Izmailov, Brian Ward"
author = "Alexey Izmailov, Brian Ward"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "nbsphinx",
    "sphinx.ext.mathjax",
]


autosummary_generate = True
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "ArrayLike": ":term:`array-like`",
    "NDArray": "~numpy.ndarray",
    "NDArray[np.float64]": "~numpy.ndarray",
    "NDArray[Union[np.float64, np.int64]]": "~numpy.ndarray",
}
napoleon_use_admonition_for_notes = True
autodoc_typehints = "none"
napoleon_use_param = True
napoleon_use_rtype = False


nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = "_static/images/logo_icon.png"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/WardBrian/scikit-stan",
            "icon": "fab fa-github",
        },
        {
            "name": "Forums",
            "url": "https://discourse.mc-stan.org/",
            "icon": "fas fa-users",
        },
    ],
    "use_edit_page_button": True,
}

html_context = {
    "github_user": "WardBrian",
    "github_repo": "scikit-stan",
    "github_version": "main",
    "doc_path": "docsrc",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


intersphinx_mapping = {
    "python": (
        "https://docs.python.org/3/",
        None,
    ),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "cmdstanpy": ("https://mc-stan.org/cmdstanpy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}
