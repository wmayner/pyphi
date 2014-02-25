#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CyPhi
=====


CyPhi is a Cython library for computing integrated information.


"""

__title__ = 'cyphi'
__version__ = '0.0.0'
__author__ = 'Will Mayner'
__copyright__ = 'Copyright 2014 Will Mayner'


# Import Cython modules like normal Python modules (automatically compile them)
# NOTE: this requires the environment to have Cython; compile and bundle
# everything explicitly before distributing
import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})

from .node import Node
from .network import Network
from .subsystem import Subsystem
from . import utils
from . import exceptions
