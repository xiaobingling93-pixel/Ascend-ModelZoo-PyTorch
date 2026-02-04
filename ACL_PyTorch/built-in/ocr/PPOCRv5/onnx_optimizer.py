#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Optimize ONNX with auto_optimizer; fallback to copy on failure."
    )
    parser.add_argument("input", help="Path to input .onnx")
    parser.add_argument("output", nargs="?", help="Path to output .onnx (optional)")
    args = parser.parse_args()

    inp = args.input
    out = args.output or (os.path.splitext(inp)[0] + "_opt.onnx")

    if not os.path.isfile(inp):
        print(f"[ERROR] input file not found: {inp}", file=sys.stderr)
        return 2

    out_dir = os.path.dirname(out) or "."
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] input : {inp}")
    print(f"[INFO] output: {out}")

    cmd = [sys.executable, "-m", "auto_optimizer", "optimize", inp, out]
    ret = subprocess.call(cmd)

    if ret != 0 or (not os.path.isfile(out)) or os.path.getsize(out) == 0:
        print(f"[WARN] auto_optimizer failed or produced empty output (ret={ret})")
        print("[WARN] fallback: copy input -> output")
        shutil.copyfile(inp, out)
    else:
        print(f"[OK] opt file generated: {out}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
