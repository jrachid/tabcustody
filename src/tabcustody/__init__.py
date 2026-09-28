"""Package tabcustody tells whether a saved model file carries its training data, without running the file."""

from tabcustody._detect import Finding
from tabcustody._report import Report, scan_file

__all__ = ["Finding", "Report", "scan_file"]
