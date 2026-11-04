from __future__ import annotations

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

from hedgelab.config import PROJECT_ROOT, get_settings

NOTEBOOKS = sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb"))


@pytest.mark.slow
@pytest.mark.data
@pytest.mark.skipif(not get_settings().options_path.exists(), reason="course dataset not present")
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs_and_its_assertions_hold(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    NotebookClient(
        nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}
    ).execute()
    out = get_settings().output_dir / "executed" / path.name
    out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out)
