"""Build Cython extensions in this directory (run: cd source_alloc && python setup_cython.py build_ext --inplace)."""
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

HERE = Path(__file__).resolve().parent

ext_modules = [
    Extension(
        name="subproblem_solver",
        sources=[str(HERE / "subproblem_solver.pyx")],
        include_dirs=[np.get_include()],
    ),
    Extension(
        name="subproblem_solver_MK",
        sources=[str(HERE / "subproblem_solver_MK.pyx")],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="source_alloc_extensions",
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={"language_level": "3"},
    ),
)
