"""Package tabcustody tells whether a saved model file carries its training data, without running the file, seals it so the data travels only under a key, and proves which rows a sealed file was fitted on."""

from tabcustody._detect import Finding
from tabcustody._report import Report, scan_file
from tabcustody._seal import (
    CannotSeal,
    CannotTrace,
    CannotUnseal,
    Verification,
    audit_key,
    file_id,
    prove,
    seal,
    unseal,
    verify,
)
from tabcustody._trace import verify_proof

__all__ = [
    "CannotSeal",
    "CannotTrace",
    "CannotUnseal",
    "Finding",
    "Report",
    "Verification",
    "audit_key",
    "file_id",
    "prove",
    "scan_file",
    "seal",
    "unseal",
    "verify",
    "verify_proof",
]
