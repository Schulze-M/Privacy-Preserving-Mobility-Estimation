from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "ppme",
        sources=["ppme.pyx", "./cpp_trie/src/main.cpp"],
        language="c++",
        include_dirs=[np.get_include(), "./cpp_trie/include/"],
        extra_compile_args=["-O3", "-std=c++17", "-fopenmp"],  # Optimization flags
        extra_link_args=["-fopenmp"],  # Linking flags
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    name="ppme",
    ext_modules=cythonize(extensions, language_level=3),
)
