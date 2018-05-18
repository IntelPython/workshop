from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
import os

# Use if you have icc for better performance
#os.environ['CC'] = "icc"
#os.environ['LDSHARED'] = "icc -shared"
#os.environ['CFLAGS'] = "-fimf-precision=high -qopt-report=5 -fno-alias -xCORE-AVX2 -axCOMMON-AVX512 -qopenmp -pthread -fno-strict-aliasing"

setup(
    name = 'cyth_scholes',
    include_dirs = [np.get_include()],
    ext_modules = cythonize("cython_scholes.pyx"),
)
