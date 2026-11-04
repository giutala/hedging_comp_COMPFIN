"""Phase 0 orchestration: load data, run checks, compute analytics, build the report."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from hedgelab import eda, plots
from hedgelab.checkpoints import Stage, save_checkpoint
from hedgelab.config import Settings
from hedgelab.loaders import build_futures_panel, load_futures_series, load_option_quotes

logger = logging.getLogger(__name__)

INCEPTION = date(2020, 1, 6)
K_CALL = 78.0
LEGS: tuple[str, str] = ("COJ21", "COK21")
LEG_OPTIONS: tuple[str, str] = ("COJ1", "COK1")
FIXING_START = date(2021, 2, 1)
FIXING_END = date(2021, 2, 26)
CURVE_DATES: tuple[date, ...] = (
    date(2020, 1, 6),
    date(2020, 4, 21),
    date(2020, 9, 1),
    date(2021, 2, 1),
)
CORR_WINDOW = 60


@dataclass(frozen=True)
class EdaResult:
    """Everything phase 0 produces, for the CLI, notebook and tests.

    Attributes
    ----------
    options : pd.DataFrame
        Raw option quotes.
    enriched : pd.DataFrame
        Options with implied vols and analytics columns.
    panel : pd.DataFrame
        Merged futures panel (long).
    settle : pd.DataFrame
        Date x contract settlement matrix.
    checks : pd.DataFrame
        Data-quality check table.
    returns : pd.DataFrame
        Daily log-returns of the two legs.
    return_stats : pd.DataFrame
        Descriptive statistics of ``returns``.
    liquidity : pd.DataFrame
        Liquidity by moneyness bucket.
    expiries : pd.DataFrame
        Option expiry / last future date per contract.
    sections : list[plots.ReportSection]
        Figures in report order.
    """

    options: pd.DataFrame
    enriched: pd.DataFrame
    panel: pd.DataFrame
    settle: pd.DataFrame
    checks: pd.DataFrame
    returns: pd.DataFrame
    return_stats: pd.DataFrame
    liquidity: pd.DataFrame
    expiries: pd.DataFrame
    sections: list[plots.ReportSection]


def run_eda(settings: Settings) -> EdaResult:
    """Run the full exploratory analysis on the training dataset.

    Parameters
    ----------
    settings : Settings
        Paths and options.

    Returns
    -------
    EdaResult
        Checks, analytics and figures.

    Raises
    ------
    DataLoadError
        If an input file is missing.
    DataValidationError
        If the two futures sources disagree.

    Notes
    -----
    The enriched option table (with implied vols) is checkpointed at the ``PARSED``
    stage so later phases can start from it without recomputing ~360k inversions.
    """
    options = load_option_quotes(settings.options_path)
    futures = load_futures_series(settings.futures_path)
    panel = build_futures_panel(options, futures)
    enriched = eda.enrich_options(options)
    save_checkpoint(options, settings.checkpoint_dir, Stage.RAW, "options")
    save_checkpoint(panel, settings.checkpoint_dir, Stage.RAW, "futures_panel")
    save_checkpoint(enriched, settings.checkpoint_dir, Stage.PARSED, "options_enriched")

    checks = eda.checks_to_frame(
        eda.run_quality_checks(options, enriched, panel, LEGS, FIXING_START)
    )
    settle = eda.settlement_matrix(panel)
    returns = eda.log_returns(panel, LEGS)
    rv = eda.realised_vol(returns)
    labels = {LEGS[0]: "Apr21 (COJ21)", LEGS[1]: "May21 (COK21)"}
    atm = {labels[f]: eda.atm_vol(enriched, o) for f, o in zip(LEGS, LEG_OPTIONS, strict=True)}
    rv_map = {labels[f]: rv[f] for f in LEGS}
    liquidity = eda.liquidity_by_moneyness(enriched)
    sofr = options.groupby("date")["sofr_rate"].first()

    sections = [
        plots.ReportSection(
            "Structure legs",
            "Apr21 and May21 move almost in lockstep. On inception both sit near 61, far below "
            "the 78 call strike; the March–April 2020 crash takes them to the low 30s.",
            plots.fig_legs_history(settle, LEGS, FIXING_START, FIXING_END, INCEPTION, K_CALL),
        ),
        plots.ReportSection(
            "Calendar spread",
            "The only risk that separates the two legs: daily moves of ~2 cents against ~50 cents "
            "for each leg. Note the expiry effect: the spread jumps from -0.8 to -1.7 USD on "
            "26 Feb 2021, the last trading day of Apr21 and a fixing day.",
            plots.fig_calendar_spread(settle, *LEGS),
        ),
        plots.ReportSection(
            "Futures curve",
            "Mild backwardation at inception, super-contango at the 21 Apr 2020 trough (storage "
            "glut), back to backwardation by the fixing month. The option file only carries "
            "chains from Dec20/Feb21 onward, so the early curves start a year out.",
            plots.fig_term_structure(settle, CURVE_DATES),
        ),
        plots.ReportSection(
            "Implied vs realised volatility",
            "Implied vol trades above realised most of the time (variance risk premium), except "
            "in March-April 2020, when realised vol spikes far above implied. The May21 spike "
            "in March 2021 is the last days before option expiry, where ATM IV is noisy.",
            plots.fig_vol_vs_realised(atm, rv_map),
        ),
        plots.ReportSection(
            "Correlation between the legs",
            f"Rolling {CORR_WINDOW}-day and EWMA correlations stay above 0.998 until the Apr21 "
            "expiry, then dip as the expiring contract jumps on its last day. Away from expiry "
            "ρ is effectively 1: the basket behaves like a single future plus a spread.",
            plots.fig_correlation(
                eda.rolling_correlation(returns, *LEGS, CORR_WINDOW),
                eda.ewma_correlation(returns, *LEGS),
                CORR_WINDOW,
            ),
        ),
        plots.ReportSection(
            "Smiles at inception",
            "Downside skew on both legs, flattening and turning up in the far call wing. The 78 "
            "call sits between the 75 and 80 listed strikes, where bid–ask IV bands widen.",
            plots.fig_smiles(
                {
                    labels[f]: eda.smile_snapshot(enriched, o, INCEPTION)
                    for f, o in zip(LEGS, LEG_OPTIONS, strict=True)
                },
                K_CALL,
                INCEPTION,
            ),
        ),
        plots.ReportSection(
            "ATM vol term structure at inception",
            "ATM vol falls gently with maturity (Samuelson effect); Apr21 and May21 differ by only "
            "about 0.3 vol points.",
            plots.fig_atm_term(eda.atm_term_structure(enriched, INCEPTION), LEG_OPTIONS, INCEPTION),
        ),
        plots.ReportSection(
            "Strike coverage, Apr21 options",
            "From March to December 2020 no Apr21 strike with a real bid reaches 78, so the "
            "call-wing vol must be extrapolated from the smile on most hedge dates.",
            plots.fig_strike_coverage(
                eda.strike_coverage(enriched, LEG_OPTIONS[0]), "Apr21 options", K_CALL
            ),
        ),
        plots.ReportSection(
            "Where quotes are missing",
            "Missing bid/ask are concentrated on low strikes of long-dated chains (deep ITM calls "
            "and their OTM puts). Near the money, quotes are almost always present.",
            plots.fig_missing_heatmap(eda.missing_share_grid(enriched)),
        ),
        plots.ReportSection(
            "Bid–ask width in vol",
            "About 1.5 vol points at the money, more than 3 in the wings: an obvious liquidity "
            "filter criterion and the main cost of a vega hedge.",
            plots.fig_iv_width(liquidity),
        ),
        plots.ReportSection(
            "Return distribution",
            "Strong negative skew and excess kurtosis around 11: a constant-vol GBM understates "
            "tail moves, so daily delta-hedge P&L will have fat tails too.",
            plots.fig_return_hist(returns[LEGS[0]], labels[LEGS[0]]),
        ),
        plots.ReportSection(
            "Discount rate",
            "SOFR is a single overnight rate (no term curve). It drops from about 1.5% to "
            "near zero in March 2020, so discounting and cash interest are second order "
            "after that.",
            plots.fig_sofr(sofr),
        ),
    ]
    return EdaResult(
        options=options,
        enriched=enriched,
        panel=panel,
        settle=settle,
        checks=checks,
        returns=returns,
        return_stats=eda.summarise_returns(returns),
        liquidity=liquidity,
        expiries=eda.expiry_table(options, panel),
        sections=sections,
    )


def write_eda_report(result: EdaResult, path: Path) -> Path:
    """Render the EDA result to a self-contained HTML file.

    Parameters
    ----------
    result : EdaResult
        Output of :func:`run_eda`.
    path : Path
        Destination file.

    Returns
    -------
    Path
        The written file.
    """
    n_err = int((~result.checks["passed"] & (result.checks["severity"] == "error")).sum())
    intro = (
        f"ICE Brent futures options, {result.options['date'].min():%d %b %Y} – "
        f"{result.options['date'].max():%d %b %Y}: {len(result.options):,} option rows over "
        f"{result.options['date'].nunique()} trading days and "
        f"{result.options['contract_code'].nunique()} chains. "
        f"{len(result.checks)} checks run, {n_err} blocking failures."
    )
    return plots.write_html_report(path, "Brent collar EDA", intro, result.checks, result.sections)
