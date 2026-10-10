"""Which saved runs may be read as results (``results/studies/run_registry.json``).

A fix that invalidates past runs -- the NPO loss defect, the breadth scorer
defect -- leaves those run directories on disk, and nothing inside them says
they are wrong. The registry says so, in one committed place, so that tools
that list or display runs cannot present a known-bad run as a result.

Stdlib only: used by the torch-free viewers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

REGISTRY_NAME = "run_registry.json"


def load_registry(studies_dir) -> Dict[str, object]:
    path = Path(studies_dir) / REGISTRY_NAME
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def run_status(registry: Dict[str, object], run_id: str,
               config: Optional[dict] = None) -> Tuple[str, str]:
    """``(status, reason)`` with status one of invalid / superseded / control / ok.

    Invalid wins over everything; a control is never a measurement whatever
    else is true of it.
    """
    invalid = registry.get("invalid") or {}
    superseded = registry.get("superseded") or {}
    if run_id in invalid:
        return "invalid", str(invalid[run_id])
    if config and config.get("control", "none") not in ("none", None):
        return "control", "T1 degradation control -- a baseline, not a measurement"
    if run_id in superseded:
        return "superseded", str(superseded[run_id])
    return "ok", ""


def protocol_of(config: Optional[dict]) -> str:
    """Runs from before protocols existed are v1."""
    return str((config or {}).get("protocol") or "v1")


def variant_of(config: Optional[dict]) -> str:
    """Training design within a protocol, as ``<loss scope>/<dtype>``.

    Two runs of one protocol that differ here (an ablation) are not replicates
    of each other, so aggregators group by it. Runs from before either field
    was recorded used full-sequence loss in bfloat16.
    """
    c = config or {}
    loss = (c.get("unlearn") or {}).get("loss_scope") or "full"
    return f"{loss}/{c.get('dtype') or 'bfloat16'}"
