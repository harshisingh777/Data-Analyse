"""Command-line interface.

python -m datalens profile data.csv [--json out.json]
python -m datalens train data.csv [--target col] [--task classification] [--tune] [--report out.html]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .insights import METRIC_LABELS, executive_summary
from .io import load_path
from .modeling import TrainConfig, train_models
from .profiling import profile_dataframe, semantic_types
from .report import render_report
from .store import DatasetSession
from .target import infer_target
from .utils import DataLensError, to_jsonable


def _print_profile(profile: dict) -> None:
    ov, q = profile["overview"], profile["quality"]
    print(
        f"\nRows {ov['rows']:,} | Columns {ov['columns']} | Missing {ov['missing_pct']:.1f}% | "
        f"Duplicates {ov['duplicate_rows']:,} | Quality {q['score']} ({q['grade']})"
    )
    print("\nColumn                         Type          Missing   Unique")
    print("-" * 64)
    for c in profile["columns"]:
        print(f"{c['name'][:30]:<30} {c['semantic_type']:<13} {c['missing_pct']:>6.1f}%  {c['unique']:>7,}")
    if profile["alerts"]:
        print("\nAlerts")
        for a in profile["alerts"]:
            print(f"  [{a['severity']:<8}] {a['message']}")


def cmd_profile(args) -> int:
    df = load_path(args.file)
    profile = profile_dataframe(df)
    _print_profile(profile)
    if args.json:
        Path(args.json).write_text(json.dumps(to_jsonable(profile), indent=2), encoding="utf-8")
        print(f"\nProfile written to {args.json}")
    return 0


def cmd_train(args) -> int:
    df = load_path(args.file)
    profile = profile_dataframe(df)
    semantic = semantic_types(profile)
    target = args.target or infer_target(df, semantic)
    if not target:
        raise DataLensError("Could not infer a target column; pass --target.")
    config = TrainConfig(target=target, task=args.task, tune=args.tune, cv_folds=args.cv, test_size=args.test_size)
    print(f"Training models to predict '{target}' ...")
    result = train_models(df, config, semantic)
    summary = result.summary
    primary = summary["primary_metric"]

    print(f"\nTask: {summary['task']} | rows used {summary['rows_used']:,} | {summary['cv_folds']}-fold CV\n")
    print(f"{'Rank':<5}{'Model':<26}{METRIC_LABELS[primary]:>16}{'Time (s)':>10}")
    print("-" * 57)
    for row in summary["leaderboard"]:
        cv = row["cv"][primary]
        score = f"{cv['mean']:.4f} ± {cv['std']:.3f}" if cv["mean"] is not None else "n/a"
        print(f"{row.get('rank', '-')!s:<5}{row['name']:<26}{score:>16}{row['fit_seconds']:>10}")

    if summary["tuning"]:
        tuning = summary["tuning"]
        print(
            f"\nTuning ({tuning['iterations']} iterations): {tuning['score_before']:.4f} -> "
            f"{tuning['score_after']:.4f} ({'kept' if tuning['improved'] else 'discarded'}) {tuning['best_params']}"
        )
    print(f"\nHold-out metrics for {summary['best_model']['name']}:")
    for k, v in summary["evaluation"]["metrics"].items():
        if v is not None:
            print(f"  {METRIC_LABELS.get(k, k):<20} {v:.4f}")
    print("\nTop features (permutation importance):")
    for f in summary["feature_importance"][:10]:
        print(f"  {f['feature']:<30} {f['importance']:.4f}  ({f['share']:.1f}%)")
    print("\n" + executive_summary(profile, summary))

    if args.report:
        session = DatasetSession(
            id="cli", name=Path(args.file).name, source="cli", df=df, profile=profile, created_at="", model=result
        )
        Path(args.report).write_text(render_report(session), encoding="utf-8")
        print(f"\nReport written to {args.report}")
    if args.model:
        Path(args.model).write_bytes(result.to_bytes())
        print(f"Model written to {args.model}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="datalens", description="Profile datasets and train AutoML models.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("profile", help="Profile a dataset and print data-quality findings.")
    p.add_argument("file")
    p.add_argument("--json", help="Write the full profile as JSON to this path.")
    p.set_defaults(func=cmd_profile)

    t = sub.add_parser("train", help="Compare models and evaluate the best one.")
    t.add_argument("file")
    t.add_argument("--target", help="Target column (inferred when omitted).")
    t.add_argument("--task", choices=["classification", "regression"])
    t.add_argument("--cv", type=int, default=5, help="Cross-validation folds (default 5).")
    t.add_argument("--test-size", type=float, default=0.2)
    t.add_argument("--tune", action="store_true", help="Run a randomised hyper-parameter search on the winner.")
    t.add_argument("--report", help="Write an HTML report to this path.")
    t.add_argument("--model", help="Write the fitted pipeline (joblib) to this path.")
    t.set_defaults(func=cmd_train)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DataLensError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
