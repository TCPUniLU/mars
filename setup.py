"""Compatibility shim.

All package metadata, dependencies, and optional extras are declared in
``pyproject.toml`` ([project] table). This file exists only so that legacy
tooling that still invokes ``setup.py`` keeps working; setuptools reads the
configuration from ``pyproject.toml``.
"""

from setuptools import setup

setup()
