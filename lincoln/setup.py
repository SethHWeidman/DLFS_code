from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("lincoln/operations/conv_cy.pyx"),
    include_dirs=[numpy.get_include()]
)
