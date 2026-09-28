"""Runs the first Python block of the README exactly as written, so the quick start cannot drift from the package."""

import re
import sys
from pathlib import Path

readme = Path(sys.argv[1]).read_text()
block = re.search(r"```python\n(.*?)```", readme, re.DOTALL)
if block is None:
    sys.exit("no Python block in the README")
exec(compile(block.group(1), "README.md", "exec"), {"__name__": "__main__"})
print("the README quick start ran as written")
