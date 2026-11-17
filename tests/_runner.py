"""Tiny test runner shared by the test files (same style as the existing project): run every test_* function, print PASS n/n."""
import sys
import traceback


def run(globs):
    names = [n for n in globs if n.startswith("test_") and callable(globs[n])]
    only = sys.argv[1:]
    n_ok = 0
    for n in names:
        if only and n not in only:
            continue
        try:
            globs[n]()
            n_ok += 1
            print("PASS", n, flush=True)
        except Exception:
            print("FAIL", n, flush=True)
            traceback.print_exc()
    n_all = len([n for n in names if not only or n in only])
    print(f"{'PASS' if n_ok == n_all else 'FAIL'} {n_ok}/{n_all}")
    sys.exit(0 if n_ok == n_all else 1)
