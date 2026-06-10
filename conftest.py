"""Root conftest — ensures the project root is on sys.path for all test imports."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import os
import tempfile
# Force PyTensor compilation directory to temp path to satisfy sandbox requirements
os.environ["PYTENSOR_FLAGS"] = f"base_compiledir={os.path.join(tempfile.gettempdir(), 'pytensor_compile_dir')}"

