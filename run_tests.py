#!/usr/bin/env python3
"""
run_tests.py
=============
Master test runner for the FALCON project.

Runs all test suites and prints a consolidated summary.
Exit code 0 if all pass, 1 if any fail.

Usage:
    python3 run_tests.py                  # all modules
    python3 run_tests.py --module 1       # only Module 1
    python3 run_tests.py --module 5,6,7   # modules 5, 6, 7
"""

import sys, os, subprocess, time
sys.path.insert(0, os.path.dirname(__file__))

SUITES = [
    ("Module 1 — Tensor + Autograd",       "tests/test_tensor_autograd.py"),
    ("Module 2 — Operations Library",       "tests/test_ops.py"),
    ("Module 3+4 — nn.Module + Training",   "tests/test_nn_training.py"),
    ("Module 5 — Privacy & Security",       "tests/test_privacy.py"),
    ("Module 6+7 — AnomalyAE + FL",         "tests/test_models_fl.py"),
    ("Module 8  — Dashboard Backend",       "tests/test_dashboard.py"),
]

BANNER = "=" * 65


def run_suite(name: str, path: str) -> tuple:
    """Run a test suite, return (passed, total, duration, output)."""
    t0  = time.time()
    res = subprocess.run(
        [sys.executable, path],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    dur = time.time() - t0
    out = res.stdout + res.stderr

    # Parse "Results: X/Y passed"
    passed = total = 0
    for line in out.splitlines():
        if "Results:" in line and "passed" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if "/" in p:
                    passed, total = map(int, p.split("/"))
                    break

    return passed, total, dur, out, res.returncode


def main():
    # Parse --module flag
    modules_filter = None
    if "--module" in sys.argv:
        idx = sys.argv.index("--module")
        modules_filter = set(int(m) for m in sys.argv[idx+1].split(","))

    print(f"\n{BANNER}")
    print("  FALCON — picograd Test Suite")
    print(f"{BANNER}\n")

    all_passed = all_total = 0
    failures   = []

    for i, (name, path) in enumerate(SUITES, 1):
        if modules_filter and i not in modules_filter:
            continue

        # Check file exists
        full_path = os.path.join(os.path.dirname(__file__), path)
        if not os.path.exists(full_path):
            print(f"  ⚠  {name}: test file not found ({path})")
            continue

        sys.stdout.write(f"  Running {name} ... ")
        sys.stdout.flush()

        passed, total, dur, out, rc = run_suite(name, full_path)
        all_passed += passed
        all_total  += total

        if rc == 0 and passed == total:
            print(f"✓  {passed}/{total}  ({dur:.1f}s)")
        else:
            print(f"✗  {passed}/{total}  ({dur:.1f}s)  ← FAILURES")
            failures.append((name, out))

    print(f"\n{BANNER}")
    status = "✓ ALL PASS" if not failures else f"✗ {len(failures)} SUITE(S) FAILED"
    print(f"  TOTAL: {all_passed}/{all_total} tests passed  {status}")
    print(f"{BANNER}\n")

    if failures:
        print("── Failure details ──────────────────────────────────────────")
        for name, out in failures:
            print(f"\n[{name}]")
            # Print only failing test lines
            for line in out.splitlines():
                if "✗" in line or "Error" in line or "assert" in line.lower():
                    print(f"  {line}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
