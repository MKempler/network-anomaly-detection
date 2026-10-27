"""Which code produced a result.

A run records the commit it ran at and whether the code differed from that
commit. Only code and config count: files under runs/ are recorded results,
which evaluations append to while they run, so changes there don't make a run
dirty.
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git_state() -> tuple[str, bool]:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    changes = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)runs"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    return commit, bool(changes)
