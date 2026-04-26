from pathlib import Path
import os

env_root = os.getenv("PROJECT_ROOT")

PROJECT_ROOT = Path(env_root) if env_root else Path(__file__).resolve().parents[1]