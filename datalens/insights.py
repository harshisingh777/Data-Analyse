"""Turn numeric results into short, plain-language findings for business readers."""

from __future__ import annotations

from .target import CLASSIFICATION

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "f1_weighted": "F1 (weighted)",
    "f1_macro": "F1 (macro)",
    "roc_auc": "ROC AUC",
    "r2": "R²",
    "mae": "MAE",
    "rmse": "RMSE",
    "median_ae": "Median AE",
    "mape": "MAPE",
    "baseline_accuracy": "Baseline accuracy",
    "train_accuracy": "Train accuracy",
    "baseline_mae": "Baseline MAE",
    "baseline_r2": "Baseline R²",
    "train_r2": "Train R²",
}


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _fmt_num(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.3g}"


def dataset_insights(profile: dict) -> list[dict]:
    ov = profile["overview"]
    quality = profile["quality"]
    findings = [
        {
            "kind": "quality",
            "title": f"Data quality grade {quality['grade']} ({quality['score']:.0f}/100)",
            "detail": (
                f"{ov['rows']:,} rows × {ov['columns']} columns; "
                f"{ov['missing_pct']:.1f}% of cells are missing and {ov['duplicate_rows']:,} rows are duplicates."
            ),
        }
    ]
    worst_missing = profile["missing"][:3]
    if worst_missing:
        cols = ", ".join(f"{m['column']} ({m['pct']:.0f}%)" for m in worst_missing)
        findings.append(
            {
                "kind": "missing",
                "title": "Missing data is concentrated in a few columns",
                "detail": f"Most incomplete: {cols}. Imputation is applied automatically during modelling.",
            }
        )
    pairs = profile["correlations"].get("top_pairs", [])
    strong = [p for p in pairs if abs(p["pearson"]) >= 0.5]
    if strong:
        p = strong[0]
        direction = "positive" if p["pearson"] > 0 else "negative"
        findings.append(
            {
                "kind": "correlation",
                "title": f"Strongest relationship: {p['a']} and {p['b']}",
                "detail": f"A {direction} linear correlation of r = {p['pearson']:.2f} "
                f"({len(strong)} pair(s) with |r| ≥ 0.5).",
            }
        )
    skewed = [
        c
        for c in profile["columns"]
        if abs((c.get("numeric") or {}).get("skewness") or 0) >= 2 and c["semantic_type"] == "numeric"
    ]
    if skewed:
        names = ", ".join(c["name"] for c in skewed[:4])
        findings.append(
            {
                "kind": "distribution",
                "title": f"{len(skewed)} heavily skewed numeric column(s)",
                "detail": f"{names}. Prefer medians over means when summarising these, or consider a log transform.",
            }
        )
    return findings


def statistics_insights(stats: dict) -> list[dict]:
    tests = stats.get("tests", [])
    if not tests:
        return []
    significant = [t for t in tests if t["significant"]]
    findings = [
        {
            "kind": "significance",
            "title": f"{len(significant)} of {len(tests)} features are significantly associated with {stats['target']}",
            "detail": f"Tested at α = {stats['alpha']} after {stats['correction']} correction.",
        }
    ]
    for t in significant[:3]:
        findings.append(
            {
                "kind": "driver",
                "title": f"{t['feature']}: {t['effect']} effect",
                "detail": f"{t['test']}, effect size {t['effect_size']:.3f}, adjusted p = {t['p_adjusted']:.2g}.",
            }
        )
    return findings


def model_insights(summary: dict) -> list[dict]:
    ev = summary["evaluation"]["metrics"]
    best = summary["best_model"]["name"]
    target = summary["target"]
    findings = []
    if summary["task"] == CLASSIFICATION:
        acc, base = ev.get("accuracy"), ev.get("baseline_accuracy")
        lift = (acc / base) if acc and base else None
        findings.append(
            {
                "kind": "performance",
                "title": f"{best} predicts {target} with {_fmt_pct(acc)} accuracy",
                "detail": f"A naive majority-class guess scores {_fmt_pct(base)}"
                + (f", so the model is {lift:.2f}× better." if lift else ".")
                + (f" ROC AUC is {ev['roc_auc']:.3f}." if ev.get("roc_auc") is not None else ""),
            }
        )
        if ev.get("balanced_accuracy") is not None and acc is not None and acc - ev["balanced_accuracy"] > 0.1:
            findings.append(
                {
                    "kind": "imbalance",
                    "title": "Classes are imbalanced",
                    "detail": f"Balanced accuracy ({_fmt_pct(ev['balanced_accuracy'])}) is well below accuracy; "
                    "minority classes are predicted less reliably.",
                }
            )
    else:
        findings.append(
            {
                "kind": "performance",
                "title": f"{best} explains {_fmt_pct(ev.get('r2'))} of the variance in {target}",
                "detail": f"Typical error (MAE) is {_fmt_num(ev.get('mae'))} versus {_fmt_num(ev.get('baseline_mae'))} "
                "for always predicting the average.",
            }
        )
    gap = summary.get("overfit_gap")
    if gap is not None and gap > 0.1:
        findings.append(
            {
                "kind": "overfitting",
                "title": "Signs of overfitting",
                "detail": f"Training score exceeds the cross-validated score by {gap:.2f}. "
                "More data or stronger regularisation may help.",
            }
        )
    top = [f for f in summary["feature_importance"] if f["importance"] > 0][:3]
    if top:
        drivers = ", ".join(f"{f['feature']} ({f['share']:.0f}%)" for f in top)
        findings.append(
            {
                "kind": "drivers",
                "title": "Key drivers",
                "detail": f"Permutation importance ranks {drivers} as the most influential inputs.",
            }
        )
    tuning = summary.get("tuning")
    if tuning:
        verdict = "improved" if tuning["improved"] else "did not improve"
        findings.append(
            {
                "kind": "tuning",
                "title": f"Hyper-parameter search {verdict} the model",
                "detail": f"{tuning['iterations']} random configurations; CV score "
                f"{tuning['score_before']:.3f} → {tuning['score_after']:.3f}.",
            }
        )
    return findings


def executive_summary(profile: dict, model_summary: dict | None = None) -> str:
    ov = profile["overview"]
    q = profile["quality"]
    text = (
        f"The dataset contains {ov['rows']:,} records across {ov['columns']} fields and earns a data-quality "
        f"grade of {q['grade']} ({q['score']:.0f}/100)."
    )
    if model_summary:
        ev = model_summary["evaluation"]["metrics"]
        best = model_summary["best_model"]["name"]
        if model_summary["task"] == CLASSIFICATION:
            text += (
                f" A {best} model classifies {model_summary['target']} with {_fmt_pct(ev.get('accuracy'))} "
                f"accuracy on unseen data, against a {_fmt_pct(ev.get('baseline_accuracy'))} baseline."
            )
        else:
            text += (
                f" A {best} model explains {_fmt_pct(ev.get('r2'))} of the variation in "
                f"{model_summary['target']} with a mean absolute error of {_fmt_num(ev.get('mae'))}."
            )
        top = [f["feature"] for f in model_summary["feature_importance"] if f["importance"] > 0][:3]
        if top:
            text += f" The most influential factors are {', '.join(top)}."
    return text
