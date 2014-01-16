from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np

extensions = [Extension("cyphi/*", ["cyphi/*.pyx"],
                        include_dirs=[np.get_include()])]

setup(
    name="cyphi",
    ext_modules=cythonize(extensions)
)
