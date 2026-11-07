from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from industrial_pm.data.ingest import prepare_official_subset
from industrial_pm.modeling.training import train_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="industrial-pm")
    subcommands = parser.add_subparsers(dest="command", required=True)

    data = subcommands.add_parser("data", help="Dataset operations")
    data_subcommands = data.add_subparsers(dest="data_command", required=True)
    prepare = data_subcommands.add_parser("prepare", help="Download and validate NASA C-MAPSS")
    prepare.add_argument("--subset", default="FD001")
    prepare.add_argument("--data-root", type=Path, default=Path("data"))
    prepare.add_argument("--rul-cap", type=int, default=125)

    train = subcommands.add_parser("train", help="Run the bounded benchmark")
    train.add_argument("--config", type=Path, default=Path("configs/fd001.yaml"))
    train.add_argument("--data-root", type=Path, default=Path("data"))
    train.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    train.add_argument("--registry-root", type=Path, default=Path("model_registry"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "data" and args.data_command == "prepare":
        manifest = prepare_official_subset(args.data_root, args.subset, args.rul_cap)
        print(json.dumps(asdict_safe(manifest), indent=2))
        return 0
    if args.command == "train":
        metrics = train_benchmark(
            args.config, args.data_root, args.artifact_root, args.registry_root
        )
        print(json.dumps(metrics, indent=2))
        return 0
    raise AssertionError("Unreachable command")


def asdict_safe(value: object) -> object:
    from dataclasses import asdict, is_dataclass

    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
