# Copyright (c) Facebook, Inc. and its affiliates.

from .utils.env import setup_environment

setup_environment()

print("siema")
# This line will be programatically read/write by setup.py.
# Leave them at the bottom of this file and don't touch them.
__version__ = "0.6"
