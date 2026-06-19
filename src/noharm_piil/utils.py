from __future__ import annotations

from pathlib import Path
from typing import Tuple


def ensure_output_dirs(out: str | Path) -> Tuple[Path, Path]:
    out = Path(out)
    fig_dir = out / "figures"
    tab_dir = out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, tab_dir
