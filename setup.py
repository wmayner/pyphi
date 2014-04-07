#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np

extensions = [Extension("cyphi/*", ["cyphi/*.pyx"],
                        include_dirs=[np.get_include()])]

setup(
    name="cyphi",
    version="0.0.0",
    description="A Cython library for computing integrated information",
    ext_modules=cythonize(extensions)
)
