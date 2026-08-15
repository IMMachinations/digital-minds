"""Path shim: make the frozen Day-1 package importable as `lib` from p1 code.

Import this module before any `from lib import ...`. The Day-1 tree at
../desires is never modified — reusable pieces are imported, architecture-
bound pieces (Harness) are generalized by copy in p1 instead.
"""
import sys
from pathlib import Path

DESIRES = Path(__file__).resolve().parent.parent / "desires"
if str(DESIRES) not in sys.path:
    sys.path.insert(0, str(DESIRES))
