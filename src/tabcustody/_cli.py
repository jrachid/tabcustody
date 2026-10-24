"""The `tabcustody` command: scan tells whether files carry training data (exit 1), seal and unseal move that data under a key."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections.abc import Sequence
from pathlib import Path

from tabcustody._report import scan_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tabcustody")
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("scan", help="tell whether model files carry their training data")
    scan.add_argument("files", nargs="+", help="pickle, joblib or .tabpfn_fit files")
    scan.add_argument("--json", action="store_true", help="print one JSON report per file")
    scan.add_argument(
        "--min-rows", type=int, default=20, help="smallest table reported (default: 20)"
    )
    seal = commands.add_parser(
        "seal",
        help="write a model file with its training data encrypted (loads, so runs, the model)",
    )
    seal.add_argument("model", help="pickle of the fitted model to seal")
    seal.add_argument("output", help="sealed file to write")
    seal.add_argument("--key-file", help="file whose bytes are the key; else TABCUSTODY_KEY")
    unseal = commands.add_parser("unseal", help="rebuild a sealed model, given its key")
    unseal.add_argument("input", help="sealed file")
    unseal.add_argument("output", help="pickle of the rebuilt model to write")
    unseal.add_argument("--key-file", help="file whose bytes are the key; else TABCUSTODY_KEY")
    return parser


def _key(key_file: str | None) -> bytes | None:
    if key_file is not None:
        return Path(key_file).read_bytes()
    from_env = os.environ.get("TABCUSTODY_KEY")
    return from_env.encode() if from_env else None


def _scan(arguments: argparse.Namespace) -> int:
    reports = [scan_file(path, min_rows=arguments.min_rows) for path in arguments.files]
    if arguments.json:
        print(json.dumps([report.to_dict() for report in reports], indent=2))
    else:
        print("\n\n".join(report.to_text() for report in reports))
        carrying = sum(report.carries_training_data for report in reports)
        unreadable = sum(report.error is not None for report in reports)
        print(
            f"\n{len(reports)} file(s) scanned: {carrying} carry training data, {unreadable} unreadable."
        )
    if any(report.error for report in reports):
        return 2
    return 1 if any(report.carries_training_data for report in reports) else 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "scan":
        return _scan(arguments)

    from tabcustody._seal import CannotSeal, CannotUnseal, seal, unseal

    key = _key(arguments.key_file)
    if key is None:
        print("no key: pass --key-file or set TABCUSTODY_KEY", file=sys.stderr)
        return 2
    try:
        if arguments.command == "seal":
            # Loading runs the model's own pickle; it is the user's file, the same trust `unseal` takes.
            with open(arguments.model, "rb") as file:
                model = pickle.load(file)
            seal(model, arguments.output, key)
            print(f"sealed {arguments.model} -> {arguments.output}")
        else:
            model = unseal(arguments.input, key)
            Path(arguments.output).write_bytes(pickle.dumps(model))
            print(f"unsealed {arguments.input} -> {arguments.output}")
    except (CannotSeal, CannotUnseal) as error:
        print(str(error), file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
