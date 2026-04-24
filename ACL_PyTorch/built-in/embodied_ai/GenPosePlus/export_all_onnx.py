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


MODELS = [
    ("pointnet2_from_score", "pointnet2_from_score.onnx"),
    ("pointnet2_from_energy", "pointnet2_from_energy.onnx"),
    ("score", "scorenet.onnx"),
    ("energy", "energynet.onnx"),
    ("scale", "scalenet.onnx"),
    ("dinov2", "dinov2_vits14.onnx"),
]


def main():
    parser = argparse.ArgumentParser(description="Batch export all ONNX models")
    parser.add_argument("--output_dir", type=str, default="./onnx_models")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    exported = 0
    skipped = 0

    script_path = Path(__file__).parent.absolute()

    for agent_type, filename in MODELS:
        filepath = os.path.join(args.output_dir, filename)
        if os.path.isfile(filepath):
            print(f"[SKIP] {filename} already exists")
            skipped += 1
        else:
            print(f"\n>>> Exporting {agent_type}...")
            cmd = [
                sys.executable, f"{script_path}/export_onnx.py",
                "--agent_type", agent_type,
                "--output_dir", args.output_dir,
                "--batch_size", str(args.batch_size)
            ]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"[FAIL] {agent_type} export failed")
            else:
                exported += 1

    print(f"\nDone. Exported: {exported}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
