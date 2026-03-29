#!/usr/bin/env python3
"""
run_tests.py
=============
Master test runner for the FALCON project.
Fully Windows-compatible (UTF-8 forced, no Unicode symbols).

Usage:
    python run_tests.py                  # all modules
    python run_tests.py --module 1       # only Module 1
    python run_tests.py --module 5,6,7   # modules 5, 6, 7
"""

import sys
import os
import subprocess
import time

# Force UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SUITES = [
    ("Module 1   Tensor + Autograd",      "tests/test_tensor_autograd.py"),
    ("Module 2   Operations Library",      "tests/test_ops.py"),
    ("Module 3+4 nn.Module + Training",    "tests/test_nn_training.py"),
    ("Module 5   Privacy & Security",      "tests/test_privacy.py"),
    ("Module 6+7 AnomalyAE + FL",          "tests/test_models_fl.py"),
    ("Module 8   Dashboard Backend",       "tests/test_dashboard.py"),
    ("Integration Full FALCON Pipeline",   "tests/test_integration.py"),
]

SEP  = "=" * 65
SEP2 = "-" * 65


def run_suite(name, path):
    t0 = time.time()

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    res = subprocess.run(
        [sys.executable, "-u", path],
        capture_output=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=env,
    )
    dur = time.time() - t0

    stdout = res.stdout.decode("utf-8", errors="replace")
    stderr = res.stderr.decode("utf-8", errors="replace")
    out = stdout + stderr

    passed = total = 0
    for line in out.splitlines():
        if "Results:" in line and "passed" in line:
            for token in line.split():
                if "/" in token:
                    try:
                        a, b = token.split("/")
                        passed, total = int(a), int(b)
                    except ValueError:
                        pass
                    break

    return passed, total, dur, out, res.returncode


def main():
    modules_filter = None
    if "--module" in sys.argv:
        idx = sys.argv.index("--module")
        modules_filter = set(int(m.strip()) for m in sys.argv[idx + 1].split(","))

    print()
    print(SEP)
    print("  FALCON  picograd Test Suite")
    print(SEP)
    print()

    all_passed = all_total = 0
    failures = []

    for i, (name, path) in enumerate(SUITES, 1):
        if modules_filter and i not in modules_filter:
            continue

        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        if not os.path.exists(full_path):
            print("  [SKIP] {} (not found: {})".format(name, path))
            continue

        sys.stdout.write("  Running {:46s}... ".format(name))
        sys.stdout.flush()

        passed, total, dur, out, rc = run_suite(name, full_path)
        all_passed += passed
        all_total  += total

        if rc == 0 and passed == total and total > 0:
            print("[PASS]  {}/{}  ({:.1f}s)".format(passed, total, dur))
        else:
            print("[FAIL]  {}/{}  ({:.1f}s)".format(passed, total, dur))
            failures.append((name, out))

    print()
    print(SEP)
    if not failures:
        status = "[PASS] ALL PASS"
    else:
        status = "[FAIL] {} SUITE(S) FAILED".format(len(failures))
    print("  TOTAL: {}/{} tests  {}".format(all_passed, all_total, status))
    print(SEP)
    print()

    if failures:
        print(SEP2)
        print("FAILURE DETAILS")
        print(SEP2)
        for name, out in failures:
            print()
            print("[{}]".format(name))
            for line in out.splitlines():
                low = line.lower()
                if any(k in low for k in ("[fail]", "error", "traceback", "assertionerror")):
                    print("  " + line)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
