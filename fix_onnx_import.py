"""
Fix ONNX Import Issue - Run this BEFORE importing anything else
This must be imported at the very top of main.py
"""

import sys
import importlib.machinery
import importlib.util
from types import ModuleType

# Create mock loader
class MockLoader:
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        pass

# Create proper spec
_onnx_spec = importlib.machinery.ModuleSpec(
    name='onnxruntime',
    loader=MockLoader(),
    origin='mock',
    is_package=False
)

# Create mock module
_mock_onnx = ModuleType('onnxruntime')
_mock_onnx.__spec__ = _onnx_spec
_mock_onnx.__loader__ = MockLoader()
_mock_onnx.__version__ = '0.0.0'
_mock_onnx.__file__ = None
_mock_onnx.__package__ = 'onnxruntime'

# Register in sys.modules BEFORE any imports
if 'onnxruntime' not in sys.modules:
    sys.modules['onnxruntime'] = _mock_onnx

# Patch find_spec globally
_original_find_spec = importlib.util.find_spec
def _patched_find_spec(name, package=None):
    if name == 'onnxruntime' or (isinstance(name, str) and (name == 'onnxruntime' or name.startswith('onnxruntime.'))):
        return _onnx_spec
    try:
        return _original_find_spec(name, package)
    except ValueError as e:
        if 'onnxruntime' in str(e):
            return _onnx_spec
        raise

importlib.util.find_spec = _patched_find_spec

# Also patch importlib.machinery.find_spec if it exists
if hasattr(importlib.machinery, 'find_spec'):
    _original_machinery_find_spec = importlib.machinery.find_spec
    def _patched_machinery_find_spec(name, package=None):
        if name == 'onnxruntime' or (isinstance(name, str) and (name == 'onnxruntime' or name.startswith('onnxruntime.'))):
            return _onnx_spec
        return _original_machinery_find_spec(name, package)
    importlib.machinery.find_spec = _patched_machinery_find_spec

