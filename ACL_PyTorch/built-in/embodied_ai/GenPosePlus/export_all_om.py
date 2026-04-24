# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import subprocess
import sys
import argparse

from pathlib import Path

# onnx_filename: om_filename (without .om extension)
MODELS = [
    ("pointnet2_from_score.onnx", "pointnet2_from_score"),
    ("pointnet2_from_energy.onnx", "pointnet2_from_energy"),
    ("scorenet.onnx", "scorenet"),
    ("energynet.onnx", "energynet"),
    ("scalenet.onnx", "scalenet"),
    ("dinov2_vits14.onnx", "dinov2_vits14"),
]


def main():
    parser = argparse.ArgumentParser(description="Batch convert all ONNX models to OM")
    parser.add_argument("--onnx_dir", type=str, default="./onnx_models")
    parser.add_argument("--output_dir", type=str, default="./om_models")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--soc_version", type=str, default="Ascend310P3")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"ONNX directory:  {args.onnx_dir}")
    print(f"OM output directory: {args.output_dir}")

    converted = 0
    skipped = 0

    script_path = Path(__file__).parent.absolute()

    for onnx_filename, om_name in MODELS:
        onnx_path = os.path.join(args.onnx_dir, onnx_filename)
        om_path = os.path.join(args.output_dir, om_name)

        if not os.path.isfile(onnx_path):
            print(f"[SKIP] {onnx_filename} not found, run export_all_onnx.py first")
            skipped += 1
        elif os.path.isfile(om_path):
            print(f"[SKIP] {om_name}.om already exists")
            skipped += 1
        else:
            print(f"\n>>> Converting {onnx_filename}...")
            cmd = [
                sys.executable, f"{script_path}/onnx2om.py",
                "--onnx_path", onnx_path,
                "--output", args.output_dir,
                "--batch_size", str(args.batch_size),
                "--soc_version", args.soc_version,
            ]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"[FAIL] {onnx_filename} conversion failed")
            else:
                converted += 1

    print(f"\nDone. Converted: {converted}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
