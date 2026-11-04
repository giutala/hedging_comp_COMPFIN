"""Generate ``notebooks/00_eda.ipynb`` from source cells (keeps the notebook diff-friendly).

Run with ``uv run python scripts/build_eda_notebook.py``; then execute the notebook in
Jupyter/VS Code or via ``uv run pytest -m slow``.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

CELLS = [
    md(
        "# 00 — Exploratory data analysis\n\n"
        "**Goal:** before any calibration, establish that the training data is internally "
        "consistent and identify the features of the market that drive modelling and hedging "
        "choices.\n\n"
        "All numbers come from `hedgelab.eda_report.run_eda`, the same function behind "
        "`uv run hedgelab eda` (which also writes `outputs/eda/eda_report.html`). Every "
        "finding the later phases rely on is enforced with an `assert`, so re-running this "
        "notebook on a modified dataset fails loudly."
    ),
    code(
        """from __future__ import annotations

import pandas as pd
import plotly.io as pio

from hedgelab import eda
from hedgelab.config import get_settings
from hedgelab.eda_report import INCEPTION, LEG_OPTIONS, LEGS, run_eda
from hedgelab.log import setup_logging

setup_logging("WARNING")
pio.renderers.default = "notebook_connected+vscode"
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 90)

res = run_eda(get_settings())
fig = {s.title: s.figure for s in res.sections}"""
    ),
    md("## 1. Shape of the dataset"),
    code(
        """opts = res.options
summary = pd.Series(
    {
        "option rows": len(opts),
        "trading days": opts["date"].nunique(),
        "option chains": opts["contract_code"].nunique(),
        "first date": f"{opts['date'].min():%Y-%m-%d}",
        "last date": f"{opts['date'].max():%Y-%m-%d}",
        "rows with bid/ask": int(opts[["option_bid", "option_ask"]].notna().all(axis=1).sum()),
    }
)
assert summary["option rows"] == 138_660
assert summary["trading days"] == 323
assert summary["option chains"] == 27
assert (summary["first date"], summary["last date"]) == ("2020-01-06", "2021-04-06")
summary"""
    ),
    md(
        "## 2. Data-quality checks\n\n"
        "`error` checks guard invariants the pipeline relies on (calendar, no-arbitrage, "
        "source consistency, fixing-window coverage). `info` checks quantify known "
        "imperfections that the calibration phase must filter."
    ),
    code(
        """checks = res.checks.set_index("check")
blocking = checks[(checks["severity"] == "error") & ~checks["passed"]]
assert blocking.empty, blocking
checks[["severity", "failed", "tested", "comment"]]"""
    ),
    code(
        """# The facts later phases build on, pinned explicitly.
assert checks.loc["put_call_parity_r0", "failed"] == 0  # undiscounted parity -> r = 0
assert checks.loc["settle_above_intrinsic", "failed"] == 0  # extrinsic >= 0 everywhere
assert checks.loc["fixing_window_COJ21", "tested"] == 20  # 20 business days in Feb 2021
assert checks.loc["fixing_window_COK21", "failed"] == 0
assert checks.loc["tick_floor_bids", "failed"] == 1_559  # floor bids to drop in calibration
assert checks.loc["underlying_spread_column", "failed"] > 0  # never use that column"""
    ),
    md(
        "## 3. Contract calendar\n\n"
        "Options expire three ICE business days before their future. The option file "
        "carries a future only while its options trade, so the **workbook supplies COJ21 "
        "for 24-26 Feb 2021**, which lie inside the fixing window."
    ),
    code(
        """exp = res.expiries.loc[["COH21", "COJ21", "COK21", "COM21"]]
assert exp.loc["COJ21", "option_expiry"] == pd.Timestamp("2021-02-23")
assert exp.loc["COJ21", "last_future_date"] == pd.Timestamp("2021-02-26")
assert exp.loc["COK21", "option_expiry"] == pd.Timestamp("2021-03-26")
exp"""
    ),
    md("## 4. Futures: the two legs"),
    code(
        """s = res.settle
assert s.loc[pd.Timestamp(INCEPTION), "COJ21"] == 61.40
assert s.loc[pd.Timestamp(INCEPTION), "COK21"] == 61.05
assert s.loc["2021-02-26", "COJ21"] == 66.13
fig["Structure legs"].show()
fig["Calendar spread"].show()"""
    ),
    code(
        """spread = (s["COK21"] - s["COJ21"]).dropna()
leg_move = s["COJ21"].diff().abs().median()
spread_move = spread.diff().abs().median()
assert spread_move < 0.1 * leg_move  # leg-specific risk is an order of magnitude smaller
assert spread.loc["2021-02-26"] < -1.5  # expiry-week decoupling of the front month
print(f"median |daily move|: leg {leg_move:.3f}, spread {spread_move:.3f} USD/bbl")
spread.loc["2021-02-19":"2021-02-26"].rename("COK21 - COJ21").to_frame()"""
    ),
    code('fig["Futures curve"].show()'),
    md("### Returns: fat tails"),
    code(
        """rs = res.return_stats
assert rs["ann_vol"].between(0.30, 0.45).all()
assert (rs["excess_kurtosis"] > 5).all()
assert (rs["skew"] < -1).all()
fig["Return distribution"].show()
rs.round(3)"""
    ),
    md(
        "### Correlation\n\n"
        "Away from the Apr21 expiry week ρ > 0.998. In the bivariate lognormal model the "
        "estimate therefore barely moves the price, but it sets how much spread risk "
        "remains once delta is hedged with a single contract."
    ),
    code(
        """r = res.returns
roll = eda.rolling_correlation(r, *LEGS, 60).dropna()
ew = eda.ewma_correlation(r, *LEGS)
assert roll.loc[:"2021-02-19"].min() > 0.998
assert ew.loc[:"2021-02-19"].min() > 0.998
assert ew.iloc[-1] < roll.loc[:"2021-02-19"].min()  # correlation drops into expiry
fig["Correlation between the legs"].show()"""
    ),
    md("## 5. Implied volatility"),
    code(
        """atm_j = eda.atm_vol(res.enriched, LEG_OPTIONS[0])
atm_k = eda.atm_vol(res.enriched, LEG_OPTIONS[1])
t0 = pd.Timestamp(INCEPTION)
assert abs(atm_j[t0] - 0.249) < 2e-3 and abs(atm_k[t0] - 0.246) < 2e-3
rv = eda.realised_vol(r)["COJ21"].dropna()
premium = (atm_j.reindex(rv.index) - rv).dropna()
assert (premium > 0).mean() > 0.6  # implied above realised on most days
print(f"implied > realised on {100 * (premium > 0).mean():.0f}% of days")
fig["Implied vs realised volatility"].show()"""
    ),
    code(
        """smile = eda.smile_snapshot(res.enriched, LEG_OPTIONS[0], INCEPTION).set_index("strike")
assert smile.loc[45.0, "iv_settle"] > smile.loc[75.0, "iv_settle"]  # downside skew
assert smile.loc[85.0, "iv_settle"] > smile.loc[80.0, "iv_settle"]  # call wing turns up
assert smile.index.min() < 78 < smile.index.max()  # 78 bracketed at inception
fig["Smiles at inception"].show()
fig["ATM vol term structure at inception"].show()
(100 * smile[["iv_bid", "iv_settle", "iv_ask"]]).round(2)"""
    ),
    md("## 6. Liquidity and strike coverage"),
    code(
        """liq = res.liquidity
near = liq.loc[["(-0.1, 0.0]", "(0.0, 0.1]"]]
assert (near["missing_share"] < 0.01).all()
assert (near["median_iv_width"] < 2.0).all()
assert liq.loc["(-2.0, -0.4]", "missing_share"] > 0.5
cov = eda.strike_coverage(res.enriched, LEG_OPTIONS[0])
share_78 = float((cov["k_max_quoted"] >= 78).mean())
assert share_78 < 0.35
print(f"78 is inside the quoted Apr21 strike range on only {100 * share_78:.0f}% of days")
fig["Strike coverage, Apr21 options"].show()
fig["Where quotes are missing"].show()
fig["Bid–ask width in vol"].show()
liq.round(3)"""
    ),
    md("## 7. Discount rate"),
    code(
        """sofr = res.options.groupby("date")["sofr_rate"].first()
assert sofr.iloc[0] == 0.0155
assert sofr.min() >= 0
assert sofr.loc["2020-04-01":].max() < 0.002
fig["Discount rate"].show()"""
    ),
    md(
        """## 8. Takeaways for the next phases

| finding | consequence |
|---|---|
| Undiscounted put-call parity holds to 1 tick; call and put IVs coincide | Black-76 with r = 0 (Asay); invert the OTM side only. Futures-style margining means there is no early-exercise premium, so American = European |
| Settlements are smooth (exchange surface); bid/ask about 1.5 vol pts wide ATM, 2-4.5 in the wings | Calibrate on settlement IVs or mids, weighted by inverse bid-ask width; drop tick-floor bids, missing quotes and IVs that fail to invert |
| 78 is outside the quoted Apr21 range on most days | Smile **extrapolation** in the call wing is a first-order modelling choice: flat vs. SVI-style wings. Test sensitivity |
| ρ > 0.998 away from expiry; spread moves are a few % of leg moves | ρ estimation is not a big price driver; deltas on the two legs are nearly interchangeable. Track the residual spread risk |
| Apr21 decouples in 24-26 Feb 2021 (spread -1.7, correlation drops) | The **roll rule** (how many fixings use Apr21) matters; confirm the n1/n2 split |
| Workbook needed for COJ21 after 23 Feb 2021; no Apr21 options after that | Hedge Apr21 vega with May21 options, or stop vega-hedging that leg in the last days |
| Fat tails (excess kurtosis about 11, worst day -17%) | Daily delta hedging leaves jump P&L; the ranking metric (mean squared P&L) punishes it, which argues for vega/gamma hedging with listed options |
| About 5% of futures prints have bid == ask | Floor the transaction-cost half-spread at half a tick |
| SOFR is overnight only, about 0 after March 2020 | Flat-rate discounting; cash interest is a small P&L term |"""
    ),
]


def main() -> None:
    """Write the notebook next to the other phase notebooks."""
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
