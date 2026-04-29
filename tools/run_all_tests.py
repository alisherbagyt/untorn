"""Run every tools/test_*.py script and report a single pass/fail summary."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

EXCLUDE = {"run_all_tests.py"}


def main() -> int:
    tests = sorted(p for p in TOOLS.glob("test_*.py") if p.name not in EXCLUDE)
    if not tests:
        print("no test_*.py files in tools/")
        return 1

    failures: list[tuple[str, int]] = []
    for t in tests:
        print(f"\n{'─' * 60}\n {t.name}\n{'─' * 60}")
        proc = subprocess.run([sys.executable, str(t)], cwd=ROOT)
        if proc.returncode != 0:
            failures.append((t.name, proc.returncode))

    print(f"\n{'═' * 60}")
    if failures:
        print(f"FAILED ({len(failures)}/{len(tests)}):")
        for name, rc in failures:
            print(f"  {name}: exit {rc}")
        return 1
    print(f"PASSED  {len(tests)}/{len(tests)} test files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
