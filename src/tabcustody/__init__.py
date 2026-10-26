"""Package tabcustody tells whether a saved model file carries its training data, without running the file, seals it so the data travels only under a key, proves which rows a sealed file was fitted on, and erases a person's rows with a receipt."""

from tabcustody._absence import AbsenceReport, check_absent
from tabcustody._detect import Finding
from tabcustody._erase import CannotErase, Erasure, erase, erase_file, verify_receipt
from tabcustody._report import Report, scan_file
from tabcustody._seal import (
    CannotSeal,
    CannotTrace,
    CannotUnseal,
    Verification,
    audit_key,
    file_id,
    opens,
    prove,
    seal,
    unseal,
    verify,
)
from tabcustody._trace import verify_proof

__all__ = [
    "AbsenceReport",
    "CannotErase",
    "CannotSeal",
    "CannotTrace",
    "CannotUnseal",
    "Erasure",
    "Finding",
    "Report",
    "Verification",
    "audit_key",
    "check_absent",
    "erase",
    "erase_file",
    "file_id",
    "opens",
    "prove",
    "scan_file",
    "seal",
    "unseal",
    "verify",
    "verify_proof",
    "verify_receipt",
]
