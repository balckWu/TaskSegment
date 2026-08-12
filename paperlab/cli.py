from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .dashboard import dashboard_once, monitor_loop
from .results import connect, export_results
from .runner import (
    collect_all,
    create_plan,
    launch_detached,
    load_plan,
    preflight,
    reset_artifacts,
    status_report,
    worker,
)
from .statistics import paired_statistics, write_statistics


DEFAULT_CONFIG = "configs/paper_3seeds.json"


def comma_values(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def comma_ints(value: str | None) -> list[int] | None:
    values = comma_values(value)
    return [int(item) for item in values] if values else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified TaskSegment paper experiment runner and result database"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--config", default=DEFAULT_CONFIG)

    validate = subparsers.add_parser("validate", help="Check datasets, local weights, repos and Conda envs")
    add_config(validate)
    validate.add_argument("--allow-missing", action="store_true")

    plan = subparsers.add_parser("plan", help="Create the deterministic six-GPU job plan")
    add_config(plan)
    plan.add_argument("--experiments", default=None, help="Comma-separated experiment names")
    plan.add_argument("--seeds", default=None, help="Comma-separated seeds")
    plan.add_argument("--show-commands", action="store_true")

    launch = subparsers.add_parser("launch", help="Launch one SSH-safe worker for each GPU")
    add_config(launch)
    launch.add_argument("--fresh", action="store_true", help="Delete only paper_artifacts before launching")
    launch.add_argument("--force", action="store_true", help="Recreate each selected run directory")
    launch.add_argument("--skip-preflight", action="store_true")
    launch.add_argument("--experiments", default=None)
    launch.add_argument("--seeds", default=None)

    work = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    add_config(work)
    work.add_argument("--gpu", type=int, required=True)
    work.add_argument("--force", action="store_true")

    status = subparsers.add_parser("status", help="Show worker and run status")
    add_config(status)
    status.add_argument("--json", action="store_true")

    dashboard = subparsers.add_parser(
        "dashboard",
        help="Print one unified six-GPU training dashboard and refresh monitor files",
    )
    add_config(dashboard)
    dashboard.add_argument("--all", action="store_true", help="Also list every run")

    watch = subparsers.add_parser("watch", help="Continuously refresh the unified dashboard")
    add_config(watch)
    watch.add_argument("--interval", type=float, default=20.0)
    watch.add_argument("--all", action="store_true", help="Also list every run")
    watch.add_argument("--no-clear", action="store_true")

    monitor = subparsers.add_parser("monitor", help=argparse.SUPPRESS)
    add_config(monitor)
    monitor.add_argument("--interval", type=float, default=30.0)

    collect = subparsers.add_parser("collect", help="Rebuild the SQLite database from run folders")
    add_config(collect)

    export = subparsers.add_parser("export", help="Export paper-ready CSV summaries from SQLite")
    add_config(export)

    stats = subparsers.add_parser("stats", help="Paired case-level bootstrap and Wilcoxon tests")
    add_config(stats)
    stats.add_argument("--reference", default="tasksegment_full")
    stats.add_argument("--comparisons", default=None)
    stats.add_argument("--iterations", type=int, default=10000)

    reset = subparsers.add_parser("reset", help="Delete only the configured paper_artifacts directory")
    add_config(reset)
    return parser


def print_plan(plan: dict, show_commands: bool) -> None:
    print(f"Plan: {sum(len(v) for v in plan['assignments'].values())} runs")
    for gpu in plan["gpus"]:
        queue = plan["assignments"].get(str(gpu), [])
        print(f"GPU {gpu}: {len(queue)} runs, estimated load={plan['estimated_load'][str(gpu)]:.1f}")
        for run_id in queue:
            print(f"  - {run_id}")
            if show_commands:
                commands = plan["jobs"][run_id]["commands"]
                print("    train:", " ".join(commands["train"]))
                print("    predict:", " ".join(commands["predict"]))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    if args.command == "validate":
        checks = preflight(config, strict=not args.allow_missing)
        for check in checks:
            marker = "OK" if check["ok"] else "MISSING"
            print(f"[{marker:7s}] {check['name']}: {check['path']}")
        return 0 if all(item["ok"] for item in checks if item["required"]) else 1

    if args.command == "plan":
        plan = create_plan(
            config,
            selected_names=comma_values(args.experiments),
            selected_seeds=comma_ints(args.seeds),
        )
        print_plan(plan, args.show_commands)
        print("Plan JSON:", config.artifacts_root / "plan.json")
        print("All commands:", config.artifacts_root / "plan.tsv")
        return 0

    if args.command == "launch":
        if args.fresh:
            reset_artifacts(config)
        if not args.skip_preflight:
            preflight(config, strict=True)
        plan = create_plan(
            config,
            selected_names=comma_values(args.experiments),
            selected_seeds=comma_ints(args.seeds),
        )
        print_plan(plan, False)
        pids = launch_detached(config, force=args.force)
        print("Workers launched:")
        for gpu, pid in sorted(pids.items()):
            print(f"  GPU {gpu}: PID {pid}")
        print("Scheduler logs:", config.artifacts_root / "scheduler")
        print("Canonical results database:", config.database_path)
        print("Live dashboard: python scripts/paper.py watch --interval 20")
        print("Aggregated monitor log:", config.artifacts_root / "monitor" / "events.log")
        return 0

    if args.command == "worker":
        worker(config, args.gpu, force=args.force)
        return 0

    if args.command == "status":
        report = status_report(config)
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
            return 0
        print("Run counts:", report["counts"])
        for gpu in config.gpus:
            state = report["workers"].get(str(gpu))
            if state is None:
                print(f"GPU {gpu}: not launched")
                continue
            current = state.get("current", "-")
            position = state.get("position", "-")
            total = state.get("total", len(state.get("jobs", [])))
            print(f"GPU {gpu}: {state.get('status')} current={current} {position}/{total}")
        failed = [row for row in report["runs"] if row["status"] == "failed"]
        if failed:
            print("Failed runs:")
            for row in failed:
                print(f"  - {row['run_id']}: {str(row.get('error') or '')[-300:]}")
        return 0

    if args.command == "dashboard":
        print(dashboard_once(config, show_all=args.all), end="")
        print("Current dashboard:", config.artifacts_root / "monitor" / "current.txt")
        print("Aggregated epoch log:", config.artifacts_root / "monitor" / "events.log")
        return 0

    if args.command == "watch":
        try:
            monitor_loop(
                config,
                interval=args.interval,
                print_to_stdout=True,
                clear_screen=not args.no_clear,
                show_all=args.all,
                stop_when_finished=False,
            )
        except KeyboardInterrupt:
            return 0
        return 0

    if args.command == "monitor":
        monitor_loop(
            config,
            interval=args.interval,
            print_to_stdout=False,
            stop_when_finished=True,
        )
        return 0

    if args.command == "collect":
        report = collect_all(config)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if not report["failures"] else 1

    if args.command == "export":
        connection = connect(config.database_path)
        try:
            paths = export_results(connection, config.artifacts_root)
        finally:
            connection.close()
        for name, path in paths.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "stats":
        comparisons = comma_values(args.comparisons)
        if comparisons is None:
            comparisons = [
                experiment["name"]
                for experiment in config.experiments
                if experiment["name"] != args.reference
            ]
        connection = connect(config.database_path)
        try:
            rows = paired_statistics(
                connection,
                args.reference,
                comparisons,
                iterations=args.iterations,
            )
        finally:
            connection.close()
        path = config.artifacts_root / "paired_statistics.csv"
        write_statistics(path, rows)
        print("Paired statistics:", path)
        return 0

    if args.command == "reset":
        reset_artifacts(config)
        print("Reset:", config.artifacts_root)
        return 0

    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
