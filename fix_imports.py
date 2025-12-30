#!/usr/bin/env python
"""
Compatibility fix for Python 3.6
importlib.metadata was added in Python 3.8, so we need to use importlib_metadata backport
"""
import sys
import importlib_metadata

# Make importlib.metadata available as an alias to importlib_metadata
sys.modules['importlib.metadata'] = importlib_metadata

print("✓ importlib.metadata compatibility fix applied")
