#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# setup.py

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('README.rst') as f:
    readme = f.read()

about = {}
with open('./pyphi/__about__.py') as f:
    exec(f.read(), about)

install_requires = [
    'joblib >=0.8.0',
    'numpy >=1.11.0',
    'psutil >=2.1.1',
    'pyemd >=0.3.0',
    'pymongo >=2.7.1',
    'pyyaml >=3.11',
    'redis >=2.10.5',
    'scipy >=0.13.3',
    'tblib >=1.3.2',
    'tqdm >=4.11.2',
]

setup(
    name=about['__title__'],
    version=about['__version__'],
    description=about['__description__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    url=about['__url__'],
    license=about['__license__'],
    long_description=readme,
    install_requires=install_requires,
    keywords=('causality causal-modeling causation iit information integrated '
              'integrated-information modeling neuroscience theory'),
    packages=['pyphi', 'pyphi.compute', 'pyphi.models'],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta;',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
    ]
)
