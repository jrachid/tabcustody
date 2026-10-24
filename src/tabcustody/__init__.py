"""Package tabcustody tells whether a saved model file carries its training data, without running the file, and seals it so the data travels only under a key."""

from tabcustody._detect import Finding
from tabcustody._report import Report, scan_file
from tabcustody._seal import CannotSeal, CannotUnseal, seal, unseal

__all__ = ["CannotSeal", "CannotUnseal", "Finding", "Report", "scan_file", "seal", "unseal"]
