#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cyphi

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

packages = ['cyphi']

requires = []

with open('README.rst') as f:
    readme = f.read()
with open('HISTORY.rst') as f:
    history = f.read()

setup(
    name="cyphi",
    version=cyphi.__version__,
    author=cyphi.__author__,
    author_email=cyphi.__author_email__,
    description=cyphi.__description__,
    include_package_data=True,
    install_requires=requires,
    packages=packages,
    package_data={'': ['LICENSE', 'NOTICE']},
    license='GNU General Public License v3.0',
    zip_safe=False,
    classifiers=(
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ),
)
