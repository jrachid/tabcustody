"""The `tabcustody` command: exits 1 when a file carries training data, 2 when a file cannot be read, 0 otherwise."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
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


if __name__ == "__main__":
    sys.exit(main())
