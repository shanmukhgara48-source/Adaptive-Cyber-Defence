"""
conftest.py — project-level pytest configuration.

Adds the grandparent directory to sys.path so that
`from adaptive_cyber_defense.xxx import ...` resolves correctly
when running pytest from anywhere inside the project tree.
"""
import sys
from pathlib import Path

# .../adaptive_cyber_defense/conftest.py → parent is .../adaptive_cyber_defense
# grandparent is wherever the project lives (e.g. .../Documents)
_PACKAGE_PARENT = str(Path(__file__).resolve().parent.parent)
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)
