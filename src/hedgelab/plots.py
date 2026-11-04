"""Plotly figure builders and a self-contained HTML report writer.

Styling follows one fixed system: a light chart surface, hairline grid, one y-axis
per chart, and categorical colours assigned in a fixed order so an entity keeps its
colour across every figure (Apr21 leg = slot 1 blue, May21 leg = slot 2 orange).
"""

from __future__ import annotations

import html
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

logger = logging.getLogger(__name__)

SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
)
SEQ_BLUE: tuple[str, ...] = (
    "#f0efec",
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#104281",
)
TEMPLATE_PATH = Path(__file__).parent / "templates" / "report.html"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _layout(fig: go.Figure, title: str, y_title: str, x_title: str = "") -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 15, "color": INK}},
        font={"family": FONT, "size": 12, "color": INK_2},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        margin={"l": 60, "r": 30, "t": 60, "b": 50},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 1, "xanchor": "right"},
        hoverlabel={"bgcolor": SURFACE, "font": {"color": INK}},
    )
    axis = {
        "gridcolor": GRID,
        "linecolor": AXIS,
        "zeroline": False,
        "tickfont": {"color": MUTED},
        "title": {"font": {"color": INK_2}},
    }
    fig.update_xaxes(**axis, title_text=x_title, showline=True)
    fig.update_yaxes(**axis, title_text=y_title)
    return fig


def _ref_hline(fig: go.Figure, y: float, label: str, right: bool = False) -> None:
    """Draw a thin horizontal reference line labelled at its left (or right) end."""
    fig.add_shape(
        type="line", xref="paper", x0=0, x1=1, yref="y", y0=y, y1=y,
        line={"color": MUTED, "width": 1},
    )  # fmt: skip
    fig.add_annotation(
        xref="paper", x=1 if right else 0, yref="y", y=y, text=label, showarrow=False,
        xanchor="right" if right else "left", yanchor="bottom",
        font={"color": INK_2, "size": 11},
    )  # fmt: skip


def _ref_vline(fig: go.Figure, x: object, label: str, top: bool = True) -> None:
    """Draw a thin vertical reference line labelled at its top (or bottom)."""
    fig.add_shape(
        type="line", yref="paper", y0=0, y1=1, xref="x", x0=x, x1=x,
        line={"color": MUTED, "width": 1},
    )  # fmt: skip
    fig.add_annotation(
        yref="paper", y=1 if top else 0, xref="x", x=x, text=label, showarrow=False,
        xanchor="left", yanchor="top" if top else "bottom", xshift=4,
        font={"color": INK_2, "size": 11},
    )  # fmt: skip


def fig_legs_history(
    settle: pd.DataFrame,
    legs: Sequence[str],
    fixing_start: date,
    fixing_end: date,
    inception: date,
    k_call: float,
) -> go.Figure:
    """Settlement history of the structure legs with fixing window and call strike.

    Parameters
    ----------
    settle : pd.DataFrame
        Date x contract settlement matrix.
    legs : Sequence[str]
        Leg contract codes in categorical order.
    fixing_start, fixing_end : date
        Averaging window bounds (shaded).
    inception : date
        Trade date (vertical reference).
    k_call : float
        Collar call strike (horizontal reference).

    Returns
    -------
    go.Figure
        Line chart.
    """
    fig = go.Figure()
    for i, leg in enumerate(legs):
        s = settle[leg].dropna()
        fig.add_trace(
            go.Scatter(
                x=s.index, y=s, name=leg, mode="lines", line={"color": SERIES[i], "width": 2}
            )
        )
    fig.add_vrect(
        x0=fixing_start,
        x1=fixing_end,
        fillcolor=GRID,
        opacity=0.6,
        line_width=0,
        annotation={"text": "fixing window", "font": {"color": INK_2, "size": 11}},
        annotation_position="bottom left",
    )
    _ref_hline(fig, k_call, f"K call = {k_call:g}", right=True)
    _ref_vline(fig, pd.Timestamp(inception), "inception")
    return _layout(fig, "Structure legs: settlement prices", "USD/bbl")


def fig_calendar_spread(settle: pd.DataFrame, near: str, far: str) -> go.Figure:
    """Far-minus-near calendar spread of the two legs.

    Parameters
    ----------
    settle : pd.DataFrame
        Date x contract settlement matrix.
    near, far : str
        Contract codes.

    Returns
    -------
    go.Figure
        Single-series line chart (positive = contango).
    """
    s = (settle[far] - settle[near]).dropna()
    fig = go.Figure(
        go.Scatter(
            x=s.index, y=s, mode="lines", line={"color": SERIES[0], "width": 2}, name="spread"
        )
    )
    fig.add_hline(y=0, line={"color": AXIS, "width": 1})
    fig.update_layout(showlegend=False)
    return _layout(fig, f"Calendar spread {far} − {near} (> 0 = contango)", "USD/bbl")


def fig_term_structure(settle: pd.DataFrame, dates: Sequence[date]) -> go.Figure:
    """Futures curve on a few dates, plotted against months to maturity.

    Parameters
    ----------
    settle : pd.DataFrame
        Date x contract settlement matrix, columns in maturity order.
    dates : Sequence[date]
        Snapshot dates (at most four, to stay within the validated palette slots).

    Returns
    -------
    go.Figure
        Line chart, one curve per date.
    """
    month_codes = "FGHJKMNQUVXZ"
    fig = go.Figure()
    for i, d in enumerate(dates):
        row = settle.loc[pd.Timestamp(d)].dropna()
        delivery = [
            pd.Timestamp(2000 + int(c[3:]), month_codes.index(c[2]) + 1, 1) for c in row.index
        ]
        months = [(m.year - d.year) * 12 + m.month - d.month for m in delivery]
        fig.add_trace(
            go.Scatter(
                x=months,
                y=row.to_numpy(),
                name=f"{d:%d %b %Y}",
                mode="lines+markers",
                line={"color": SERIES[i], "width": 2},
                marker={"size": 6},
                text=list(row.index),
                hovertemplate="%{text}: %{y:.2f}<extra></extra>",
            )
        )
    fig.update_layout(hovermode="closest")
    return _layout(fig, "Brent futures curve", "USD/bbl", "months to delivery")


def fig_sofr(rate: pd.Series[float]) -> go.Figure:
    """Overnight SOFR history.

    Parameters
    ----------
    rate : pd.Series[float]
        Decimal rate indexed by date.

    Returns
    -------
    go.Figure
        Step line in percent.
    """
    fig = go.Figure(
        go.Scatter(
            x=rate.index,
            y=100 * rate,
            mode="lines",
            line={"color": SERIES[0], "width": 2, "shape": "hv"},
        )
    )
    fig.update_layout(showlegend=False)
    return _layout(fig, "Overnight SOFR (single rate per day, no term curve)", "%")


def fig_vol_vs_realised(
    atm: dict[str, pd.Series[float]], rv: dict[str, pd.Series[float]]
) -> go.Figure:
    """ATM implied vol vs 21-day realised vol, one panel per leg.

    Parameters
    ----------
    atm : dict[str, pd.Series[float]]
        Leg label -> ATM implied vol.
    rv : dict[str, pd.Series[float]]
        Leg label -> realised vol (same keys).

    Returns
    -------
    go.Figure
        Two stacked panels sharing the time axis.
    """
    labels = list(atm)
    fig = make_subplots(
        rows=len(labels), cols=1, shared_xaxes=True, subplot_titles=labels, vertical_spacing=0.08
    )
    for r, lab in enumerate(labels, start=1):
        for j, (name, s) in enumerate((("ATM implied", atm[lab]), ("21d realised", rv[lab]))):
            fig.add_trace(
                go.Scatter(
                    x=s.index,
                    y=100 * s,
                    name=name,
                    mode="lines",
                    legendgroup=name,
                    showlegend=r == 1,
                    line={"color": SERIES[j], "width": 2},
                ),
                row=r,
                col=1,
            )
    fig.update_layout(height=520)
    fig = _layout(fig, "Implied vs realised volatility", "vol (%)")
    for ann in fig.layout.annotations:
        ann.update(font={"color": INK_2, "size": 12}, x=0, xanchor="left")
    return fig


def fig_correlation(rolling: pd.Series[float], ewma: pd.Series[float], window: int) -> go.Figure:
    """Plot the rolling and EWMA correlation of the two legs' daily log-returns.

    Parameters
    ----------
    rolling : pd.Series[float]
        Rolling-window correlation.
    ewma : pd.Series[float]
        EWMA correlation.
    window : int
        Rolling window length (for the legend).

    Returns
    -------
    go.Figure
        Line chart.
    """
    fig = go.Figure()
    for i, (name, s) in enumerate(
        ((f"rolling {window}d", rolling.dropna()), ("EWMA λ=0.94", ewma))
    ):
        fig.add_trace(
            go.Scatter(
                x=s.index, y=s, name=name, mode="lines", line={"color": SERIES[i], "width": 2}
            )
        )
    return _layout(fig, "Correlation of daily log-returns, Apr21 vs May21", "ρ")


def fig_smiles(snapshots: dict[str, pd.DataFrame], k_call: float, on: date) -> go.Figure:
    """OTM implied-vol smiles with bid/ask IV bands.

    Parameters
    ----------
    snapshots : dict[str, pd.DataFrame]
        Chain label -> output of :func:`hedgelab.eda.smile_snapshot`.
    k_call : float
        Collar call strike (vertical reference).
    on : date
        Snapshot date (title).

    Returns
    -------
    go.Figure
        Settlement IV lines with bid-ask error bars.
    """
    fig = go.Figure()
    for i, (lab, sm) in enumerate(snapshots.items()):
        sm = sm.dropna(subset=["iv_settle"])
        fig.add_trace(
            go.Scatter(
                x=sm["strike"],
                y=100 * sm["iv_settle"],
                name=lab,
                mode="lines+markers",
                line={"color": SERIES[i], "width": 2},
                marker={"size": 8},
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "thickness": 1,
                    "width": 4,
                    "array": 100 * (sm["iv_ask"] - sm["iv_settle"]),
                    "arrayminus": 100 * (sm["iv_settle"] - sm["iv_bid"]),
                },
                customdata=np.c_[100 * sm["iv_bid"], 100 * sm["iv_ask"]],
                hovertemplate=(
                    "K=%{x}: %{y:.2f}% (bid %{customdata[0]:.2f}, ask %{customdata[1]:.2f})"
                ),
            )
        )
    fwd = float(next(iter(snapshots.values()))["forward"].iloc[0])
    _ref_vline(fig, fwd, f"F ≈ {fwd:.2f}")
    _ref_vline(fig, k_call, f"K call = {k_call:g}", top=False)
    return _layout(
        fig,
        f"OTM implied-vol smiles on {on:%d %b %Y} (bars = bid/ask IV)",
        "implied vol (%)",
        "strike",
    )


def fig_atm_term(term: pd.DataFrame, highlight: Sequence[str], on: date) -> go.Figure:
    """ATM implied vol across listed expiries, highlighting the structure legs.

    Parameters
    ----------
    term : pd.DataFrame
        Output of :func:`hedgelab.eda.atm_term_structure`.
    highlight : Sequence[str]
        Option codes to emphasise.
    on : date
        Snapshot date.

    Returns
    -------
    go.Figure
        Line with emphasised markers.
    """
    fig = go.Figure(
        go.Scatter(
            x=term["expiration"],
            y=100 * term["atm_iv"],
            mode="lines+markers",
            name="ATM IV",
            line={"color": AXIS, "width": 2},
            marker={"size": 8, "color": AXIS},
            text=term["contract_code"],
            hovertemplate="%{text}: %{y:.2f}%<extra></extra>",
        )
    )
    for i, code in enumerate(highlight):
        r = term[term["contract_code"] == code]
        fig.add_trace(
            go.Scatter(
                x=r["expiration"],
                y=100 * r["atm_iv"],
                mode="markers+text",
                name=code,
                marker={"size": 11, "color": SERIES[i], "line": {"color": SURFACE, "width": 2}},
                text=[code],
                textposition="top center",
                textfont={"color": INK_2},
                hovertemplate=f"{code}: %{{y:.2f}}%<extra></extra>",
            )
        )
    fig.update_layout(hovermode="closest")
    return _layout(
        fig, f"ATM implied-vol term structure on {on:%d %b %Y}", "implied vol (%)", "option expiry"
    )


def fig_strike_coverage(cov: pd.DataFrame, label: str, k_call: float) -> go.Figure:
    """Range of strikes with a real bid over time vs the forward and the call strike.

    Parameters
    ----------
    cov : pd.DataFrame
        Output of :func:`hedgelab.eda.strike_coverage`.
    label : str
        Chain label (title).
    k_call : float
        Collar call strike.

    Returns
    -------
    go.Figure
        Band (quoted strikes) plus forward line.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cov.index,
            y=cov["k_max_quoted"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cov.index,
            y=cov["k_min_quoted"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(42,120,214,0.15)",
            name="strikes with bid > 0.01",
            customdata=cov[["k_min_quoted", "k_max_quoted"]].to_numpy(),
            hovertemplate="quoted %{customdata[0]:.0f}–%{customdata[1]:.0f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cov.index,
            y=cov["forward"],
            name="future",
            mode="lines",
            line={"color": SERIES[0], "width": 2},
        )
    )
    _ref_hline(fig, k_call, f"K call = {k_call:g}", right=True)
    return _layout(fig, f"{label}: quoted strike range vs future", "USD/bbl")


def fig_missing_heatmap(grid: pd.DataFrame) -> go.Figure:
    """Heatmap of missing-quote share by time to expiry and log-moneyness.

    Parameters
    ----------
    grid : pd.DataFrame
        Output of :func:`hedgelab.eda.missing_share_grid`.

    Returns
    -------
    go.Figure
        Sequential single-hue heatmap.
    """
    z = 100 * grid.to_numpy()
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(grid.columns),
            y=list(grid.index),
            zmin=0,
            zmax=100,
            colorscale=[[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)],
            colorbar={"title": {"text": "% missing"}, "outlinewidth": 0},
            xgap=2,
            ygap=2,
            text=np.round(z).astype(int).astype(str),
            texttemplate="%{text}",
            hovertemplate="τ %{y}, ln(K/F) %{x}: %{z:.1f}% missing<extra></extra>",
        )
    )
    fig.update_layout(hovermode="closest")
    fig = _layout(
        fig,
        "Missing quotes: deep ITM-call / OTM-put wing of long-dated chains",
        "time to expiry",
        "ln(K/F)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def fig_iv_width(liq: pd.DataFrame) -> go.Figure:
    """Median bid-ask width in vol points by log-moneyness bucket (OTM quotes).

    Parameters
    ----------
    liq : pd.DataFrame
        Output of :func:`hedgelab.eda.liquidity_by_moneyness`.

    Returns
    -------
    go.Figure
        Single-series bar chart.
    """
    fig = go.Figure(
        go.Bar(
            x=list(liq.index),
            y=liq["median_iv_width"],
            marker={"color": SERIES[0], "cornerradius": 4},
            customdata=100 * liq["median_rel_spread"],
            hovertemplate="%{x}: %{y:.2f} vol pts (rel. spread %{customdata:.1f}%)<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False, hovermode="closest", bargap=0.35)
    return _layout(
        fig, "Bid-ask width in implied vol, OTM quotes", "median width (vol pts)", "ln(K/F)"
    )


def fig_return_hist(returns: pd.Series[float], label: str) -> go.Figure:
    """Histogram of daily log-returns with the fitted normal density.

    Parameters
    ----------
    returns : pd.Series[float]
        Daily log-returns.
    label : str
        Series label.

    Returns
    -------
    go.Figure
        Histogram (density) and normal pdf line.
    """
    x = returns.to_numpy()
    grid = np.linspace(x.min(), x.max(), 300)
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=100 * x,
            histnorm="probability density",
            nbinsx=80,
            name="empirical",
            marker={"color": SERIES[0], "line": {"color": SURFACE, "width": 1}},
        )
    )
    pdf = stats.norm.pdf(grid, x.mean(), x.std(ddof=1)) / 100
    fig.add_trace(
        go.Scatter(
            x=100 * grid,
            y=pdf,
            name="normal fit",
            mode="lines",
            line={"color": SERIES[1], "width": 2},
        )
    )
    fig.update_layout(hovermode="closest", bargap=0)
    return _layout(
        fig, f"{label} daily log-returns: fat tails vs normal", "density", "daily return (%)"
    )


@dataclass(frozen=True)
class ReportSection:
    """One figure in the HTML report.

    Attributes
    ----------
    title : str
        Section heading.
    caption : str
        What to read from the figure.
    figure : go.Figure
        The chart.
    """

    title: str
    caption: str
    figure: go.Figure


def write_html_report(
    path: Path, title: str, intro: str, checks: pd.DataFrame, sections: Sequence[ReportSection]
) -> Path:
    """Write a single self-contained HTML report (plotly.js inlined once).

    Parameters
    ----------
    path : Path
        Output file.
    title : str
        Page title.
    intro : str
        Intro paragraph (plain text).
    checks : pd.DataFrame
        Output of :func:`hedgelab.eda.checks_to_frame`.
    sections : Sequence[ReportSection]
        Figures in display order.

    Returns
    -------
    Path
        The written file.
    """
    marks = {"pass": "&#10003;", "fail": "&#10007;", "note": "i"}
    rows = []
    for _, c in checks.iterrows():
        status = "pass" if c["passed"] else ("fail" if c["severity"] == "error" else "note")
        cells = (
            f"<td class='mark'>{marks[status]}</td>",
            f"<td><code>{html.escape(str(c['check']))}</code></td>",
            f"<td>{html.escape(str(c['description']))}</td>",
            f"<td class='num'>{c['failed']:,} / {c['tested']:,}</td>",
            f"<td>{html.escape(str(c['comment']))}</td>",
        )
        rows.append(f"<tr class='{status}'>{''.join(cells)}</tr>")
    body = []
    for i, sec in enumerate(sections):
        div = sec.figure.to_html(
            full_html=False, include_plotlyjs=(i == 0), config={"displaylogo": False}
        )
        body.append(
            f"<section><h2>{html.escape(sec.title)}</h2>"
            f"<p class='cap'>{html.escape(sec.caption)}</p>{div}</section>"
        )
    template = Template(TEMPLATE_PATH.read_text(encoding="utf-8"))
    page = template.substitute(
        title=html.escape(title),
        intro=html.escape(intro),
        rows="\n".join(rows),
        body="\n".join(body),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    logger.info("Report written: %s", path)
    return path
