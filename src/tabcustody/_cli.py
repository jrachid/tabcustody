"""The `tabcustody` command: scan tells whether files carry training data (exit 1), seal and unseal move that data under a key, id, prove and check-proof tie a past prediction to the rows of a sealed file, verify checks that this environment predicts as the sealing one, check --absent tells whether given rows left a file."""

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
    check = commands.add_parser(
        "check", help="tell whether given rows are absent from a model file (never prints them)"
    )
    check.add_argument(
        "--absent", required=True, metavar="ROWS", help="CSV or .npy file of the rows to look for"
    )
    check.add_argument("file", help="model file to search, which is never run")
    check.add_argument("--json", action="store_true", help="print the report as JSON")
    check.add_argument(
        "--min-rows", type=int, default=20, help="smallest table searched (default: 20)"
    )
    seal = commands.add_parser(
        "seal",
        help="write a model file with its training data encrypted (loads, so runs, the model)",
    )
    seal.add_argument("model", help="pickle of the fitted model to seal")
    seal.add_argument("output", help="sealed file to write")
    seal.add_argument("--key-file", help="file whose bytes are the key; else TABCUSTODY_KEY")
    seal.add_argument(
        "--probes",
        type=int,
        default=0,
        help="run the model on this many probe inputs and seal its predictions, for verify",
    )
    unseal = commands.add_parser("unseal", help="rebuild a sealed model, given its key")
    unseal.add_argument("input", help="sealed file")
    unseal.add_argument("output", help="pickle of the rebuilt model to write")
    unseal.add_argument("--key-file", help="file whose bytes are the key; else TABCUSTODY_KEY")
    ident = commands.add_parser(
        "id", help="print a sealed file's identifier, to log beside each prediction (no key)"
    )
    ident.add_argument("input", help="sealed file")
    prove = commands.add_parser(
        "prove", help="print the JSON proof that one row was in a sealed file's training table"
    )
    prove.add_argument("input", help="sealed file")
    prove.add_argument("--position", type=int, required=True, help="row position in the table")
    prove.add_argument("--table", type=int, default=0, help="committed table (default: 0)")
    prove.add_argument("--key-file", help="file whose bytes are the key; else TABCUSTODY_KEY")
    audit = commands.add_parser(
        "audit-key", help="print, in hex, the key that checks proofs and decrypts nothing"
    )
    audit.add_argument("--key-file", help="file whose bytes are the key; else TABCUSTODY_KEY")
    verify = commands.add_parser(
        "verify",
        help="tell whether this environment predicts as the one that sealed the file (exit 1 if not)",
    )
    verify.add_argument("input", help="sealed file, sealed with --probes")
    verify.add_argument(
        "--tolerance", type=float, default=0.0, help="largest deviation accepted (default: 0)"
    )
    verify.add_argument("--key-file", help="file whose bytes are the key; else TABCUSTODY_KEY")
    proof_check = commands.add_parser(
        "check-proof", help="tell whether a proof holds for a logged identifier (exit 1 if not)"
    )
    proof_check.add_argument("proof", help="JSON file written from `tabcustody prove`")
    proof_check.add_argument(
        "--id", required=True, help="the identifier logged with the prediction"
    )
    proof_check.add_argument(
        "--audit-key-file", required=True, help="file holding the hex audit key"
    )
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


def _check_proof(arguments: argparse.Namespace) -> int:
    from tabcustody._trace import verify_proof

    proof = json.loads(Path(arguments.proof).read_text())
    audit = bytes.fromhex(Path(arguments.audit_key_file).read_text().strip())
    if verify_proof(proof, audit, arguments.id):
        print(f"the proof holds: row {proof['position']} was in table {proof['table']}")
        return 0
    print("the proof does not hold for this identifier and audit key", file=sys.stderr)
    return 1


def _check(arguments: argparse.Namespace) -> int:
    from tabcustody._absence import check_absent, read_rows

    try:
        rows = read_rows(arguments.absent)
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2
    report = check_absent(arguments.file, rows, min_rows=arguments.min_rows)
    print(json.dumps(report.to_dict(), indent=2) if arguments.json else report.to_text())
    if report.error is not None:
        return 2
    if report.present:
        return 1
    return 0 if report.verified else 3


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "scan":
        return _scan(arguments)
    if arguments.command == "check":
        return _check(arguments)
    if arguments.command == "check-proof":
        return _check_proof(arguments)

    from tabcustody._seal import (
        CannotSeal,
        CannotTrace,
        CannotUnseal,
        audit_key,
        file_id,
        prove,
        seal,
        unseal,
        verify,
    )

    if arguments.command == "id":
        try:
            print(file_id(arguments.input))
        except CannotTrace as error:
            print(str(error), file=sys.stderr)
            return 3
        return 0
    key = _key(arguments.key_file)
    if key is None:
        print("no key: pass --key-file or set TABCUSTODY_KEY", file=sys.stderr)
        return 2
    try:
        if arguments.command == "seal":
            # Loading runs the model's own pickle; it is the user's file, the same trust `unseal` takes.
            with open(arguments.model, "rb") as file:
                model = pickle.load(file)
            seal(model, arguments.output, key, probes=arguments.probes)
            print(f"sealed {arguments.model} -> {arguments.output}")
        elif arguments.command == "unseal":
            model = unseal(arguments.input, key)
            Path(arguments.output).write_bytes(pickle.dumps(model))
            print(f"unsealed {arguments.input} -> {arguments.output}")
        elif arguments.command == "prove":
            proof = prove(arguments.input, key, arguments.position, arguments.table)
            print(json.dumps(proof, indent=2))
        elif arguments.command == "verify":
            result = verify(arguments.input, key, arguments.tolerance)
            print(result.to_text())
            return 0 if result.holds else 1
        else:
            print(audit_key(key).hex())
    except (CannotSeal, CannotUnseal, CannotTrace) as error:
        print(str(error), file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
