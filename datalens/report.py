"""Render a self-contained HTML analysis report."""

from __future__ import annotations

from datetime import datetime, timezone

from jinja2 import Environment, PackageLoader, select_autoescape

from . import __version__
from .insights import METRIC_LABELS, dataset_insights, executive_summary, model_insights, statistics_insights

_env = Environment(
    loader=PackageLoader("datalens", "templates"),
    autoescape=select_autoescape(["html"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def _num(value, digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.{digits}g}" if abs(value) < 1 else f"{value:,.{digits - 1}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _pct(value, digits: int = 1) -> str:
    return "–" if value is None else f"{value:.{digits}f}%"


_env.filters["num"] = _num
_env.filters["pct"] = _pct


def render_report(session) -> str:
    profile = session.profile
    model_summary = session.model.summary if session.model else None
    stats = session.statistics
    findings = dataset_insights(profile)
    if stats:
        findings += statistics_insights(stats)
    if model_summary:
        findings += model_insights(model_summary)
    template = _env.get_template("report.html")
    return template.render(
        name=session.name,
        generated=datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC"),
        version=__version__,
        summary=executive_summary(profile, model_summary),
        profile=profile,
        findings=findings,
        stats=stats,
        model=model_summary,
        metric_labels=METRIC_LABELS,
    )
