"""What produced a result: code version, environment, and input files.

Every NPO artefact from before 1 Sep 2026 was produced by a defective loss
(``74313dc``), and nothing in those run directories says so -- they can be told
apart from valid runs only by date. A run directory that records its own git
commit closes that gap: a later fix can name exactly which runs it invalidates.

Nothing here imports torch at module level, so it can be used by the stdlib-only
tools as well.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


def _git(*args: str, cwd: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                             timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_info(repo: Optional[Path] = None) -> Dict[str, object]:
    """Commit, branch, and whether the working tree had uncommitted changes.

    ``dirty=True`` means the recorded commit does NOT fully describe the code
    that ran. Recorded rather than refused: a quick diagnostic run on a dirty
    tree is legitimate, but it must not later pass for a clean one.
    """
    repo = repo or Path(__file__).resolve().parents[1]
    commit = _git("rev-parse", "HEAD", cwd=repo)
    if commit is None:
        return {"commit": None, "branch": None, "dirty": None,
                "note": "not a git checkout, or git unavailable"}
    status = _git("status", "--porcelain", "--untracked-files=no", cwd=repo)
    return {"commit": commit,
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo),
            "dirty": bool(status)}


def environment_info() -> Dict[str, object]:
    """Interpreter and library versions that can change a number."""
    info: Dict[str, object] = {"python": sys.version.split()[0],
                               "platform": platform.platform()}
    for mod in ("torch", "transformers", "tokenizers", "datasets", "numpy"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:                      # pragma: no cover - optional deps
            info[mod] = None
    try:
        import torch
        info["cuda"] = torch.version.cuda
        info["device"] = (torch.cuda.get_device_name(0)
                          if torch.cuda.is_available() else None)
        info["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:                          # pragma: no cover
        pass
    return info


def sha256_file(path) -> Optional[str]:
    """Content hash, so a run names the exact item set or annotation file it used."""
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
