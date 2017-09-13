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
    'numpy >=1.11.0, <2.0.0',
    'scipy >=0.13.3, <1.0.0',
    'pyemd >=0.3.0, <1.0.0',
    'joblib >=0.8.0a3, <1.0.0',
    'psutil >=2.1.1, <3.0.0',
    'marbl-python >=2.0.0, <3.0.0',
    'pymongo >=2.7.1, <3.0.0',
    'pyyaml >=3.11, <4.0',
    'redis >=2.10.5, <3.0.0',
    'tqdm >=4.11.2, <5.0.0',
    'tblib >=1.3.2, <2.0.0'
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
    include_package_data=True,
    install_requires=install_requires,
    packages=['pyphi', 'pyphi.compute', 'pyphi.models'],
    package_data={'pyphi': ['data/**/*'],
                  '': ['README.rst', 'LICENSE.md', 'pyphi_config.yml',
                       'redis.conf']},
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
    ]
)
