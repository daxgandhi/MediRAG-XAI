"""
conftest.py — pytest configuration for MEDIRAG-XAI backend tests
Sets sys.path so that modules can be found correctly
"""
import sys
import os

# Add the backend directory to sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
