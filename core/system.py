import sys
from subprocess import run
from core.paths import PROJECT_ROOT

def run_cli(args):
    return run(
        [sys.executable, "run_training.py", *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True
    )