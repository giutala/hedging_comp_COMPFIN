"""Generate ``notebooks/00_eda.ipynb`` from source cells.

Keeping the notebook source in a script makes reviews diff-friendly and guarantees the
committed notebook matches the code. Regenerate and execute with::

    uv run python scripts/build_eda_notebook.py
    uv run pytest -m slow            # executes the notebook and checks every assertion

Notes
-----
All computation lives in the tested ``hedgelab`` package; notebook cells only call it,
display results, pin the numbers later phases rely on with ``assert``, and interpret.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

CELLS = [
    # ------------------------------------------------------------------ header
    md(
        """# 00 — Exploratory data analysis

**Purpose.** Before calibrating anything, establish that the training data (ICE Brent futures options, 6 Jan 2020 – 6 Apr 2021) is internally consistent. Find anything that looks like a data error, and identify the market features that should drive modelling and hedging choices for the Apr21/May21 Asian collar.

**How to read this notebook**

* Every computation is a call into the tested `hedgelab` package. The same pipeline backs `uv run hedgelab eda`, which also writes `outputs/eda/eda_report.html`.
* Every number a later phase relies on is pinned with an `assert` against a named constant (cell 1.2). Re-running on modified data fails loudly instead of silently changing conclusions.
* Each result is followed by an **Interpretation** cell. Section 3 collects everything that could signal wrong data, with a verdict for each item: data artefact or genuine market feature.

**Conventions.** Prices are in USD/bbl. Time to expiry is ACT/365. Implied vols come from the *undiscounted* Black-76 formula (r = 0). Log-moneyness is `ln(K/F)`."""
    ),
    md("## 1. Setup\n\n### 1.1 Imports and display options"),
    code(
        """from __future__ import annotations

import pandas as pd
import plotly.io as pio
from IPython.display import Markdown, display

from hedgelab import eda
from hedgelab.config import get_settings
from hedgelab.eda_report import INCEPTION, LEG_OPTIONS, LEGS, run_eda
from hedgelab.log import setup_logging

setup_logging("WARNING")  # keep the notebook free of INFO log noise
pio.renderers.default = "notebook_connected+vscode"  # renders in Jupyter and VS Code
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 90)
pd.set_option("display.float_format", "{:,.4f}".format)"""
    ),
    md(
        """### 1.2 Expected values

The values this notebook pins, all in one place. If the dataset changes, this is the only cell to review."""
    ),
    code(
        """# Dataset shape
N_OPTION_ROWS = 138_660
N_TRADING_DAYS = 323
N_CHAINS = 27
FIRST_DATE, LAST_DATE = "2020-01-06", "2021-04-06"

# Structure legs at inception (settlement, USD/bbl) and around the fixing window
F0_APR21, F0_MAY21 = 61.40, 61.05
APR21_LAST_SETTLE = 66.13  # 26 Feb 2021, last trading day of COJ21
N_FIXING_DAYS = 20  # business days in February 2021

# Implied volatility at inception (ATM, settlement IV) and tolerance
ATM_IV_APR21, ATM_IV_MAY21, ATM_IV_TOL = 0.249, 0.246, 2e-3

# Correlation floor away from the Apr21 expiry week
RHO_FLOOR = 0.998
PRE_EXPIRY_END = "2021-02-19"

# Liquidity
N_TICK_FLOOR_BIDS = 1_559
TICK = eda.TICK"""
    ),
    md(
        """### 1.3 Run the phase-0 pipeline

`run_eda` loads both raw files, merges the futures sources (asserting that they agree), inverts about 360k implied vols, runs every data-quality check, and builds the figures. It also writes raw/parsed checkpoints to `data/checkpoints/`, so later phases start from the validated data."""
    ),
    code(
        """res = run_eda(get_settings())
figures = {section.title: section.figure for section in res.sections}
opts, enriched, panel, settle = res.options, res.enriched, res.panel, res.settle"""
    ),
    # ------------------------------------------------------------------ 2. overview + checks
    md("## 2. Dataset overview and data-quality checks\n\n### 2.1 Shape of the dataset"),
    code(
        """summary = pd.Series(
    {
        "option rows": len(opts),
        "trading days": opts["date"].nunique(),
        "option chains": opts["contract_code"].nunique(),
        "first date": f"{opts['date'].min():%Y-%m-%d}",
        "last date": f"{opts['date'].max():%Y-%m-%d}",
        "rows with bid/ask": int(opts[["option_bid", "option_ask"]].notna().all(axis=1).sum()),
    },
    name="value",
)
assert summary["option rows"] == N_OPTION_ROWS
assert summary["trading days"] == N_TRADING_DAYS
assert summary["option chains"] == N_CHAINS
assert (summary["first date"], summary["last date"]) == (FIRST_DATE, LAST_DATE)
summary.to_frame()"""
    ),
    md(
        """**Interpretation.** Each row is one (date, option) with its settlement, bid, ask, the underlying future's settlement and bid/ask, and the overnight SOFR. 27 monthly chains are listed, Dec20 to Feb23. 88% of rows carry a two-sided quote; section 6 shows where the missing ones are."""
    ),
    md(
        """### 2.2 Automated checks

* `error` checks guard invariants the pipeline relies on: calendar, uniqueness, no-arbitrage bounds, consistency between the two futures sources, and coverage of the fixing window. Any failure stops the notebook.
* `info` checks measure known imperfections; section 3 examines each one."""
    ),
    code(
        """checks = res.checks.set_index("check")
blocking = checks[(checks["severity"] == "error") & ~checks["passed"]]
assert blocking.empty, f"blocking data-quality failures:\\n{blocking}"
checks[["severity", "passed", "failed", "tested", "comment"]]"""
    ),
    code(
        """# Facts the later phases build on, pinned individually.
assert checks.loc["put_call_parity_r0", "failed"] == 0  # C - P = F - K without discounting
assert checks.loc["settle_above_intrinsic", "failed"] == 0  # extrinsic >= 0 everywhere
assert checks.loc["butterfly_convexity", "failed"] == 0  # no static arbitrage across strikes
assert checks.loc["fixing_window_COJ21", "tested"] == N_FIXING_DAYS
assert checks.loc["fixing_window_COJ21", "failed"] == 0
assert checks.loc["fixing_window_COK21", "failed"] == 0
assert checks.loc["tick_floor_bids", "failed"] == N_TICK_FLOOR_BIDS"""
    ),
    md(
        """**Interpretation.**

* **No blocking failure.** The calendar contains only trading days; the missing weekdays are exactly the ICE Brent closures (Good Friday, Christmas, New Year). The two futures sources agree to the cent wherever they overlap.
* **Undiscounted put-call parity `C − P = F − K` holds to one tick** on all 60,910 strike pairs. That is the signature of futures-style margined options, priced by Black-76 with r = 0 (Asay, 1982). For such options early exercise is never optimal, so American = European and no early-exercise premium needs modelling.
* **Settlements are free of static arbitrage:** monotone in strike, convex within one tick (settlements are rounded to 0.01), and never below intrinsic.
* Both legs have a settlement on all 20 business days of February 2021, so the averaging period is fully observable in sample."""
    ),
    # ------------------------------------------------------------------ 3. anomalies
    md(
        """## 3. Possible data errors and anomalies

Each item below is a pattern that *could* signal bad data. For each one we show the evidence, give a verdict, and state how the downstream code handles it. The summary table is at the end of the section."""
    ),
    md("### 3.1 `underlying_spread` column inconsistent with bid/ask"),
    code(
        """spread_mismatch = eda.spread_column_mismatch(opts)
n_future_days = opts.drop_duplicates(["date", "future_contract_code"]).shape[0]
assert set(spread_mismatch["difference"].abs().round(4)) == {TICK}  # always exactly one tick
display(
    Markdown(
        f"{len(spread_mismatch):,} of {n_future_days:,} future-days "
        f"({len(spread_mismatch) / n_future_days:.0%}) disagree, always by exactly one tick."
    )
)
spread_mismatch.head()"""
    ),
    md(
        """**Verdict: data artefact (vendor rounding).** The stored spread looks computed from unrounded bid/ask. **Handling:** the column is never used; spreads are always recomputed as `ask − bid`."""
    ),
    md("### 3.2 Zero-width futures quotes (bid = ask)"),
    code(
        """zero_width = eda.zero_width_futures(panel)
share_by_source = (panel["bid"] >= panel["ask"]).groupby(panel["source"]).mean()
display(share_by_source.rename("share bid >= ask").to_frame())
zero_width.sort_values("date").tail()"""
    ),
    md(
        """**Verdict: partly artefact.** On the option-file side (4.5% of prints) a locked market at settlement is plausible. In the workbook (14%) the zero-width prints cluster in the COZ22 series in October 2022, where bid = ask = settlement for consecutive days. That looks like missing quotes **filled with the settlement price**.

**Handling:** the hedging cost model floors the half-spread at half a tick, so trading is never free. This matters most for the out-of-sample test (Dec22 leg, October 2022)."""
    ),
    md("### 3.3 Missing interior strikes"),
    code(
        """holes = eda.strike_holes(opts)
assert set(holes["contract_code"]) == {"COU1", "COV1"}  # only two chains are affected
assert not holes["contract_code"].isin(LEG_OPTIONS).any()  # never the structure legs
display(holes.groupby("contract_code")["date"].agg(["count", "min", "max"]))
holes.head(3)"""
    ),
    md(
        """**Verdict: listing gap in two non-structure chains** (the Sep21 and Oct21 chains, COU1 and COV1, lack the 25 and 30 strikes from March to December 2020). **Handling:** smile fits interpolate in log-moneyness and never assume a complete grid. No impact on the Apr21/May21 legs."""
    ),
    md("### 3.4 Quotes that do not invert to an implied vol"),
    code(
        """non_invertible = eda.non_invertible_quotes(enriched)
assert (non_invertible["option_bid"] <= TICK + 1e-9).all()  # all at the 0.01 price floor
assert (non_invertible["tau"] < 0.05).all()  # all within ~2.5 weeks of option expiry
display(Markdown(f"{len(non_invertible)} OTM quotes fail to invert; all are tick-floor bids."))
non_invertible.head()"""
    ),
    md(
        """**Verdict: expected, not an error.** A 0.01 bid on an almost-expired, far-OTM option sits below any meaningful Black price, so no vol reproduces it. The same mechanism explains the 62 strikes where call and put IV differ: deep-ITM options priced at intrinsic, where IV is undefined. **Handling:** the calibration filter drops tick-floor bids and very short-dated quotes."""
    ),
    md(
        """### 3.5 Frozen quotes: unchanged although the future moved

A quote whose bid, ask and settlement are identical to the previous day while the underlying moved may be **stale**, i.e. carried forward by the vendor. But an unchanged price can also be genuine, when a vol move offsets the delta move. The test: did the quote's implied vol move *in line with its own chain*? A frozen quote whose IV change deviates from the chain's median IV change by more than one vol point is flagged `suspect`."""
    ),
    code(
        """frozen = eda.frozen_quotes(enriched, min_expected_move=0.05, max_iv_gap_pts=1.0)
suspects = frozen[frozen["suspect"]]
assert (len(frozen), len(suspects)) == (128, 26)
display(
    Markdown(
        f"{checks.loc['fresh_quotes', 'failed']:,} quotes are unchanged day-on-day; "
        f"{len(frozen)} of them should have moved by more than 5 cents (|delta·dF| > 0.05). "
        f"Of those, **{len(suspects)}** have an IV change inconsistent with their chain "
        f"(> 1 vol pt) and are likely stale."
    )
)
suspects.sort_values("iv_gap_pts", key=abs, ascending=False).head(8)"""
    ),
    code(
        """# Where do the suspects sit? A one-tick change is worth several vol points on cheap or
# nearly expired options, so the IV-consistency test is noisy there.
cheap_or_short = (suspects["settlement_price"] <= 0.25) | (suspects["tau"] < 0.1)
assert int((~cheap_or_short).sum()) == 7  # the only real stale-quote candidates
display(
    Markdown(
        f"{cheap_or_short.mean():.0%} of suspects are cheap (settle <= 0.25) or within "
        f"~5 weeks of expiry; {int((~cheap_or_short).sum())} are not."
    )
)
display(suspects.loc[~cheap_or_short])
legs_suspect = suspects[suspects["contract_code"].isin(LEG_OPTIONS)]
display(Markdown(f"Suspect frozen quotes on the structure legs: {len(legs_suspect)}"))
legs_suspect"""
    ),
    md(
        """**Verdict: no material stale-data problem; a handful of doubtful quotes.**

* More than half of the unchanged quotes sit at the 0.01 price floor, where staying put is expected.
* Of the 128 that should have moved, about 80% did "move" in vol space consistently with their chain: a vol change offset the delta move.
* The ~26 inconsistent ones are mostly cheap or close to expiry. There, a one-tick rounding of the settlement is worth several vol points, so the IV test is noisy. The few remaining cases are the only real stale-quote candidates.
* They all pass the no-arbitrage checks, so bounds alone cannot catch them.

**Handling:** calibration weights each quote by its inverse bid-ask width and drops quotes flagged `suspect`. With only 6–11 strikes per smile, a single stale quote would otherwise bend the fitted wing."""
    ),
    md("### 3.6 Front-month decoupling in the Apr21 expiry week (market feature)"),
    code(
        """spread_series = (settle["COK21"] - settle["COJ21"]).dropna().rename("COK21 - COJ21")
expiry_week = pd.concat(
    [settle.loc["2021-02-19":"2021-02-26", list(LEGS)], spread_series.loc["2021-02-19":]], axis=1
)
assert spread_series.loc["2021-02-26"] < -1.5
assert settle.loc["2021-02-26", "COJ21"] == APR21_LAST_SETTLE
expiry_week"""
    ),
    md(
        """**Verdict: genuine market behaviour, not an error.** Backwardation had been building since January, and the spread already sits around −0.8 to −0.9 USD in the run-up. On **26 Feb, Apr21's last trading day**, it jumps to −1.71: a 0.94 move in a day, in a market where the spread normally moves about 2 cents. That is the usual convergence/squeeze of an expiring Brent contract. Apr21 is priced from the workbook on 24–26 Feb (its options expired on 23 Feb), and the workbook agrees with the option file on every overlapping day, which supports it being real.

**Handling:** it falls inside the fixing window, so the **roll rule** (on which fixing days Apr21 vs May21 is the reference) directly changes the payoff. This is why the n1/n2 split must be confirmed."""
    ),
    md("### 3.7 Implied-vol blow-up just before option expiry"),
    code(
        """atm_may21 = eda.atm_vol(enriched, LEG_OPTIONS[1])
last_days = atm_may21.loc["2021-03-15":].rename("COK1 ATM IV").to_frame()
assert last_days.iloc[-1, 0] > 0.45  # vs ~0.35 two weeks earlier
last_days.T"""
    ),
    md(
        """**Verdict: numerical artefact of short maturity.** With a few days to expiry, a one-tick change in price moves the inverted vol by several points. The 65% readings say little about the vol level. **Handling:** exclude options with fewer than about 5 trading days to expiry from smile calibration, and take vega hedges in longer-dated contracts in the final week."""
    ),
    md("### 3.8 Extreme daily returns"),
    code(
        """top_moves = eda.largest_moves(res.returns, "COJ21", n=6)
expected_event_days = pd.to_datetime(
    ["2020-03-09", "2020-04-21", "2020-03-18", "2020-11-09", "2020-03-19", "2020-03-16"]
)
assert set(top_moves["date"]) == set(expected_event_days)
top_moves.assign(pct=lambda d: 100 * d["log_return"])"""
    ),
    md(
        """**Verdict: genuine, matching documented events.**

| date | event |
|---|---|
| 9 Mar 2020 (−17%) | OPEC+ talks collapse, Saudi–Russia price war |
| 16–19 Mar 2020 | COVID demand shock, global risk-off |
| 21 Apr 2020 (−11%) | storage glut, the day after WTI settled negative |
| 9 Nov 2020 (+7%) | first COVID-vaccine efficacy announcement |

**Handling:** keep them. They are the reason a constant-vol model underestimates hedging P&L tails."""
    ),
    md(
        """### 3.9 Minor items

* **`atm_strike` rounding:** at exact half-strikes (e.g. F = 52.50) the column rounds up to 55 rather than to even. This is a convention only, and the column is not used.
* **Workbook starts earlier:** the COJ21 sheet begins on 2 Jan 2020, two days before the training window. The `as_of` filter makes those rows irrelevant.
* **SOFR:** a single overnight rate per day, not a curve. Its moves (1.55% → about 0 in March 2020) match the Fed cuts."""
    ),
    md(
        """### 3.10 Summary of anomalies

| # | item | count | verdict | handling |
|---|---|---|---|---|
| 3.1 | `underlying_spread` off by 1 tick | 3,060 future-days | vendor artefact | recompute `ask − bid` |
| 3.2 | bid = ask on futures | 4.5% (option file), 14% (workbook) | partly filled with settlement | floor half-spread at ½ tick |
| 3.3 | missing interior strikes | 204 chain-days (COU1, COV1) | listing gap, not our legs | interpolate, never assume grid |
| 3.4 | non-invertible / IV mismatch | 72 / 62 quotes | tick floor near expiry | filter floor bids and short τ |
| 3.5 | frozen quotes | 26 doubtful of 128 | mostly tick noise; few stale candidates | drop `suspect`, weight by spread |
| 3.6 | Apr21 last-day jump vs May21 | 26 Feb 2021 | **genuine** | roll rule matters; confirm n1/n2 |
| 3.7 | IV blow-up near expiry | last ~5 days | short-τ artefact | exclude τ < ~5 trading days |
| 3.8 | extreme returns | 6 days | **genuine** events | keep; explains fat-tailed P&L |

**Conclusion:** no evidence of systematic data corruption. The real artefacts are small, identifiable and handled by the calibration filters. The two things that look odd but are real (3.6 and 3.8) are exactly the ones that matter most for pricing and hedging."""
    ),
    # ------------------------------------------------------------------ 4. calendar
    md(
        """## 4. Contract calendar

ICE Brent options expire three business days before their future. The option file carries a future only while its options trade. The **workbook therefore supplies COJ21 for 24–26 Feb 2021**, the last three fixing days of Apr21."""
    ),
    code(
        """expiries = res.expiries.loc[["COH21", "COJ21", "COK21", "COM21"]]
assert expiries.loc["COJ21", "option_expiry"] == pd.Timestamp("2021-02-23")
assert expiries.loc["COJ21", "last_future_date"] == pd.Timestamp("2021-02-26")
assert expiries.loc["COK21", "option_expiry"] == pd.Timestamp("2021-03-26")
expiries"""
    ),
    md(
        """**Interpretation.** No Apr21 option exists after 23 Feb, but Apr21 still fixes on (at least some of) 24–26 Feb. In the last days, the Apr21 vega of the collar cannot be hedged with Apr21 options; May21 options are the natural substitute."""
    ),
    # ------------------------------------------------------------------ 5. futures
    md("## 5. Futures\n\n### 5.1 The two structure legs"),
    code(
        """assert settle.loc[pd.Timestamp(INCEPTION), "COJ21"] == F0_APR21
assert settle.loc[pd.Timestamp(INCEPTION), "COK21"] == F0_MAY21
figures["Structure legs"].show()"""
    ),
    md(
        """**Interpretation.** At inception both legs trade near 61, 17 USD below the 78 call strike, so the call starts well out of the money. Through 2020 the legs fall into the low 30s and recover into the 60s by the fixing month. A hedger short this collar faces both large directional moves and a strike that drifts in and out of reach."""
    ),
    md("### 5.2 Calendar spread: the only risk separating the legs"),
    code(
        """leg_move = settle["COJ21"].diff().abs().median()
spread_move = spread_series.diff().abs().median()
assert spread_move < 0.1 * leg_move  # leg-specific risk is an order of magnitude smaller
display(
    Markdown(
        f"Median absolute daily move: leg **{leg_move:.2f}**, spread **{spread_move:.2f}** USD/bbl."
    )
)
figures["Calendar spread"].show()"""
    ),
    md(
        """**Interpretation.** The spread flips from backwardation to contango in the March 2020 collapse and back in early 2021. Its daily moves are about 4% of a leg's. Once the common level is hedged, the residual two-leg risk is small, apart from the expiry week (3.6)."""
    ),
    md("### 5.3 Term structure"),
    code('figures["Futures curve"].show()'),
    md(
        """**Interpretation.** Mild backwardation at inception, then a super-contango at the April 2020 trough (the storage glut: about 6 USD from front to back over 15 months), then back to backwardation by February 2021. The cost of carry is itself a moving risk factor; this is relevant to the optional spot-driven model extension."""
    ),
    md("### 5.4 Return distribution"),
    code(
        """stats = res.return_stats
assert stats["ann_vol"].between(0.30, 0.45).all()
assert (stats["excess_kurtosis"] > 5).all()
assert (stats["skew"] < -1).all()
figures["Return distribution"].show()
stats"""
    ),
    md(
        """**Interpretation.** Annualised vol is about 37%, with strong negative skew (about −1.8) and excess kurtosis about 11. Returns are far from Gaussian. A daily delta hedge under a constant-vol model will leave fat-tailed P&L, which the ranking metric (mean squared daily P&L) penalises heavily. This argues for also hedging convexity (gamma/vega) with listed options."""
    ),
    md("### 5.5 Correlation between the legs"),
    code(
        """rolling_rho = eda.rolling_correlation(res.returns, *LEGS, window=60).dropna()
ewma_rho = eda.ewma_correlation(res.returns, *LEGS)
assert rolling_rho.loc[:PRE_EXPIRY_END].min() > RHO_FLOOR
assert ewma_rho.loc[:PRE_EXPIRY_END].min() > RHO_FLOOR
assert ewma_rho.iloc[-1] < rolling_rho.loc[:PRE_EXPIRY_END].min()  # drops into expiry
figures["Correlation between the legs"].show()"""
    ),
    md(
        """**Interpretation.** Both estimators stay above 0.998 until the expiry week, then dip as Apr21 jumps on its last day. For pricing, ρ is effectively 1: the basket behaves like a single future plus a small spread, and the choice of estimator barely moves the price. For hedging, the near-unit correlation means the two leg deltas are almost interchangeable. Re-estimating ρ at each rebalance (as the task requires) mainly matters in February 2021."""
    ),
    # ------------------------------------------------------------------ 6. implied vol
    md("## 6. Implied volatility\n\n### 6.1 Implied vs realised"),
    code(
        """atm_apr21 = eda.atm_vol(enriched, LEG_OPTIONS[0])
t0 = pd.Timestamp(INCEPTION)
assert abs(atm_apr21[t0] - ATM_IV_APR21) < ATM_IV_TOL
assert abs(atm_may21[t0] - ATM_IV_MAY21) < ATM_IV_TOL
realised = eda.realised_vol(res.returns)["COJ21"].dropna()
vol_premium = (atm_apr21.reindex(realised.index) - realised).dropna()
share_premium = float((vol_premium > 0).mean())
assert share_premium > 0.6  # implied above realised on most days
display(Markdown(f"Implied vol exceeds 21-day realised vol on **{share_premium:.0%}** of days."))
figures["Implied vs realised volatility"].show()"""
    ),
    md(
        """**Interpretation.** ATM implied vol is about 25% at inception. It sits above realised vol on about three quarters of days: the usual variance risk premium, which a hedger long gamma pays and one short gamma earns. In March–April 2020 the relation inverts sharply (realised above 80%, implied about 45%). This is the period that will dominate the backtest P&L."""
    ),
    md("### 6.2 Smiles at inception"),
    code(
        """smile_apr21 = eda.smile_snapshot(enriched, LEG_OPTIONS[0], INCEPTION).set_index("strike")
assert smile_apr21.loc[45.0, "iv_settle"] > smile_apr21.loc[75.0, "iv_settle"]  # put skew
assert smile_apr21.loc[85.0, "iv_settle"] > smile_apr21.loc[80.0, "iv_settle"]  # call wing up
assert smile_apr21.index.min() < 78 < smile_apr21.index.max()  # 78 bracketed at t0
figures["Smiles at inception"].show()
(100 * smile_apr21[["iv_bid", "iv_settle", "iv_ask"]]).round(2)"""
    ),
    md(
        """**Interpretation.** There is a clear downside skew (about 29% at K = 45 vs 21% at K = 80), and the call wing turns up beyond 80. Apr21 and May21 smiles are nearly parallel, about 0.4 vol points apart. At inception, 78 lies between listed strikes, so the call vol is interpolated. The relevant put strike (solved in phase 5) will sit on the steeper skew side, which makes the zero-cost condition sensitive to the skew fit."""
    ),
    md("### 6.3 ATM term structure at inception"),
    code('figures["ATM vol term structure at inception"].show()'),
    md(
        """**Interpretation.** ATM vol declines gently with maturity (Samuelson effect: distant futures react less to news). The two legs are adjacent on the curve, so a single vol level per leg (as in the note's moment-matching formula) is a reasonable first approximation."""
    ),
    # ------------------------------------------------------------------ 7. liquidity
    md("## 7. Liquidity and strike coverage"),
    code(
        """liquidity = res.liquidity
near_atm = liquidity.loc[["(-0.1, 0.0]", "(0.0, 0.1]"]]
assert (near_atm["missing_share"] < 0.01).all()
assert (near_atm["median_iv_width"] < 2.0).all()
assert liquidity.loc["(-2.0, -0.4]", "missing_share"] > 0.5
liquidity"""
    ),
    code(
        """coverage = eda.strike_coverage(enriched, LEG_OPTIONS[0])
share_78_quoted = float((coverage["k_max_quoted"] >= 78).mean())
assert share_78_quoted < 0.35
display(
    Markdown(
        f"78 lies inside the quoted Apr21 strike range on only **{share_78_quoted:.0%}** of days."
    )
)
figures["Strike coverage, Apr21 options"].show()"""
    ),
    code(
        """figures["Where quotes are missing"].show()
figures["Bid–ask width in vol"].show()"""
    ),
    md(
        """**Interpretation.**

* Near the money, quotes are essentially always present and about 1.5 vol points wide. In the wings the width grows to 2–4.5 points, and half of the deep low-strike quotes of long-dated chains are missing. This sets the calibration filter: two-sided quotes, bid above the floor, weighted by inverse width.
* **Strike coverage is the key modelling risk.** From March to December 2020 no Apr21 strike with a real bid reaches 78. On those hedge dates the call vol comes purely from **extrapolating** the smile, so the wing model (flat vs. SVI-style) is a first-order choice and should be sensitivity-tested.
* A vega hedge pays about 1.5 vol points of round-trip cost ATM, much more in the wings. Hedging with near-ATM options and rebalancing only when needed will matter for the P&L metric."""
    ),
    # ------------------------------------------------------------------ 8. rates
    md("## 8. Discount rate"),
    code(
        """sofr = opts.groupby("date")["sofr_rate"].first()
assert sofr.iloc[0] == 0.0155
assert sofr.min() >= 0
assert sofr.loc["2020-04-01":].max() < 0.002
figures["Discount rate"].show()"""
    ),
    md(
        """**Interpretation.** SOFR falls from 1.55% to near zero with the March 2020 Fed cuts and stays there. Only an overnight rate is provided, so discounting uses a flat rate and the cash account accrues daily (ACT/360). After March 2020 interest is a second-order P&L term."""
    ),
    # ------------------------------------------------------------------ 9. takeaways
    md(
        """## 9. Takeaways for the next phases

| finding | consequence |
|---|---|
| Undiscounted parity holds; call and put IVs coincide | Black-76 with r = 0 (Asay); no early-exercise premium; invert the OTM side only |
| Bid/ask about 1.5 vol pts ATM, 2–4.5 in the wings; floor bids, stale quotes and short-τ noise identified (§3) | Filter: two-sided, bid > 0.01, τ ≥ ~5 trading days, drop `suspect`; weight by inverse bid-ask width |
| 78 outside the quoted Apr21 range on about 76% of days | Smile **extrapolation** in the call wing is a first-order choice; test sensitivity |
| ρ > 0.998 except in the expiry week | ρ barely affects price; leg deltas nearly interchangeable; track residual spread risk |
| Apr21 jumps vs May21 on its last day, 26 Feb 2021 (genuine) | The **roll rule** (n1/n2 split) changes the payoff; confirm it |
| No Apr21 options after 23 Feb 2021 | Hedge Apr21 vega with May21 options in the final days |
| Fat tails (kurtosis about 11), realised far above implied in Mar–Apr 2020 | Delta-only hedging leaves large squared P&L; consider gamma/vega hedging |
| Workbook futures quotes partly filled with settlement | Floor transaction costs at ½ tick, especially out of sample |
| SOFR overnight only, about 0 after March 2020 | Flat-rate discounting; cash interest is a small P&L term |

**Open questions:** the exact roll split for February 2021 (n1 = 18 vs 19 Apr21 fixings), and the settlement/payment date of the collar used for discounting."""
    ),
]


def main() -> None:
    """Write the notebook to ``notebooks/00_eda.ipynb``."""
    nb = nbf.v4.new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb.metadata["language_info"] = {"name": "python"}
    out = Path(__file__).resolve().parents[1] / "notebooks" / "00_eda.ipynb"
    nbf.write(nb, out)


if __name__ == "__main__":
    main()
