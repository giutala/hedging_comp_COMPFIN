from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from hedgelab.checkpoints import Stage, load_checkpoint, save_checkpoint
from hedgelab.config import Settings, get_settings
from hedgelab.exceptions import DataLoadError
from hedgelab.loaders import filter_as_of, load_futures_series, load_option_quotes

SETTINGS = get_settings()
needs_data = pytest.mark.skipif(
    not SETTINGS.options_path.exists(), reason="course dataset not present in DATA_DIR"
)


def test_settings_resolve_relative_paths() -> None:
    s = Settings()
    assert s.data_dir.is_absolute()
    assert s.db_path.parent == s.output_dir


def test_filter_as_of_drops_future_rows() -> None:
    df = pd.DataFrame({"date": pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07"])})
    out = filter_as_of(df, date(2020, 1, 6))
    assert out["date"].max() == pd.Timestamp("2020-01-06")
    assert len(out) == 2


def test_filter_as_of_on_index() -> None:
    df = pd.DataFrame({"x": [1, 2]}, index=pd.to_datetime(["2020-01-06", "2020-01-08"]))
    assert len(filter_as_of(df, date(2020, 1, 7))) == 1


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1.0, 2.0]})
    save_checkpoint(df, tmp_path, Stage.PARSED, "demo")
    pd.testing.assert_frame_equal(load_checkpoint(tmp_path, Stage.PARSED, "demo"), df)


def test_missing_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(DataLoadError):
        load_checkpoint(tmp_path, Stage.RAW, "nope")


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DataLoadError):
        load_option_quotes(tmp_path / "missing.parquet")


@pytest.mark.data
@needs_data
def test_dataset_loads_and_has_training_legs() -> None:
    opts = load_option_quotes(SETTINGS.options_path)
    assert opts["date"].min() == pd.Timestamp("2020-01-06")
    assert {"COJ21", "COK21"} <= set(opts["future_contract_code"])
    assert opts["sofr_rate"].max() < 0.05  # converted from percent
    fut = load_futures_series(SETTINGS.futures_path)
    assert "COJ21" in fut
    # Apr21 futures must be available through the whole February 2021 fixing window.
    assert fut["COJ21"].index.max() >= pd.Timestamp("2021-02-26")
