"""Structured, validated configuration loaded from environment variables / ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings for the hedgelab pipeline.

    Values are read from environment variables prefixed with ``HEDGELAB_`` or from a
    ``.env`` file in the project root. Relative paths are resolved against the root.

    Attributes
    ----------
    data_dir : Path
        Directory containing the raw course dataset (parquet + xlsx).
    options_file : str
        File name of the unpivoted option bid/ask parquet inside ``data_dir``.
    futures_file : str
        File name of the futures historical-series workbook inside ``data_dir``.
    checkpoint_dir : Path
        Directory for pickled stage checkpoints (raw / parsed / validated).
    output_dir : Path
        Directory for figures, reports and the SQLite results database.
    db_name : str
        File name of the SQLite results database inside ``output_dir``.
    log_level : str
        Logging level name.
    mc_paths : int
        Default number of Monte Carlo paths.
    mc_seed : int
        Seed for the Monte Carlo random generator (reproducibility).

    Notes
    -----
    The dataset is local, so no API key is required. If a remote data source is ever
    added, its key belongs here as a required ``SecretStr`` field so that a missing key
    fails fast at start-up rather than mid-run.
    """

    model_config = SettingsConfigDict(
        env_prefix="HEDGELAB_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Path("data")
    options_file: str = "unpivoted_bid_offer_train_clean.parquet"
    futures_file: str = "COJ21_COZ22_historical_series.xlsx"
    checkpoint_dir: Path = Path("data/checkpoints")
    output_dir: Path = Path("outputs")
    db_name: str = "hedgelab.sqlite"
    log_level: str = "INFO"
    mc_paths: int = Field(default=200_000, gt=0)
    mc_seed: int = 20200106

    @field_validator("data_dir", "checkpoint_dir", "output_dir")
    @classmethod
    def _resolve(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @field_validator("log_level")
    @classmethod
    def _check_level(cls, value: str) -> str:
        value = value.upper()
        if value not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"invalid log level {value!r}")
        return value

    @property
    def options_path(self) -> Path:
        """Full path of the option quotes parquet."""
        return self.data_dir / self.options_file

    @property
    def futures_path(self) -> Path:
        """Full path of the futures workbook."""
        return self.data_dir / self.futures_file

    @property
    def db_path(self) -> Path:
        """Full path of the SQLite results database."""
        return self.output_dir / self.db_name


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Returns
    -------
    Settings
        Validated settings.

    Raises
    ------
    pydantic.ValidationError
        If any environment value is malformed.
    """
    return Settings()
