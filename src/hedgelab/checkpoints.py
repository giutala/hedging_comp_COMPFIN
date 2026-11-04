"""Boundary checkpoints: pickled DataFrames for each pipeline stage."""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path

import pandas as pd

from hedgelab.exceptions import DataLoadError

logger = logging.getLogger(__name__)


class Stage(StrEnum):
    """Pipeline stage boundaries at which data is checkpointed."""

    RAW = "raw"
    PARSED = "parsed"
    VALIDATED = "validated"


def checkpoint_path(directory: Path, stage: Stage, name: str) -> Path:
    """Return the pickle path for ``name`` at ``stage``.

    Parameters
    ----------
    directory : Path
        Root checkpoint directory.
    stage : Stage
        Pipeline stage.
    name : str
        Dataset name, e.g. ``"options"``.

    Returns
    -------
    Path
        ``<directory>/<stage>/<name>.pkl``.
    """
    return directory / stage.value / f"{name}.pkl"


def save_checkpoint(df: pd.DataFrame, directory: Path, stage: Stage, name: str) -> Path:
    """Pickle a DataFrame at a stage boundary.

    Parameters
    ----------
    df : pd.DataFrame
        Frame to persist.
    directory : Path
        Root checkpoint directory.
    stage : Stage
        Pipeline stage.
    name : str
        Dataset name.

    Returns
    -------
    Path
        The written file.
    """
    path = checkpoint_path(directory, stage, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    logger.debug("Checkpoint written: %s", path)
    return path


def load_checkpoint(directory: Path, stage: Stage, name: str) -> pd.DataFrame:
    """Load a previously pickled stage checkpoint.

    Parameters
    ----------
    directory : Path
        Root checkpoint directory.
    stage : Stage
        Pipeline stage.
    name : str
        Dataset name.

    Returns
    -------
    pd.DataFrame
        The stored frame.

    Raises
    ------
    DataLoadError
        If the checkpoint does not exist.
    """
    path = checkpoint_path(directory, stage, name)
    if not path.exists():
        raise DataLoadError(f"checkpoint not found: {path} (run the earlier stage first)")
    df = pd.read_pickle(path)
    if not isinstance(df, pd.DataFrame):
        raise DataLoadError(f"checkpoint {path} does not contain a DataFrame")
    return df
