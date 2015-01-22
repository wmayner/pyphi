#!/usr/bin/env python3
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


from pyphi import __version__


with open('README.rst') as f:
    readme = f.read()


install_requires = [
    'numpy >=1.8.1, <2.0.0',
    'scipy >=0.13.3, <1.0.0',
    'pyemd >=0.0.7, <1.0.0',
    'joblib >=0.8.0a3, <1.0.0',
    'psutil >= 2.1.1, <3.0.0',
    'marbl-python >=2.0.0, <3.0.0',
    'pymongo >=2.7.1, <3.0.0',
    'pyyaml>=3.11, < 4.0'
]

test_require = [
    'pytest',
    'coverage',
    'sphinx_rtd_theme'
]


setup(
    name="pyphi",
    version=__version__,
    description='A Python library for computing integrated information.',
    author='Will Mayner',
    author_email='wmayner@gmail.com',
    long_description=readme,
    include_package_data=True,
    install_requires=install_requires,
    tests_require=test_require,
    test_suite='test',
    packages=['pyphi'],
    package_data={'': ['README.rst', 'LICENSE.md', 'pyphi_config.yml']},
    license='GNU General Public License v3.0',
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering',
    ]
)
