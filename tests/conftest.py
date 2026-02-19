from __future__ import annotations

import os
import sys
from pathlib import Path

# Headless plotting backend for CI/self-hosted runner environments.
os.environ.setdefault("MPLBACKEND", "Agg")


# Ensure repository root is importable so tests can `import src...`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
