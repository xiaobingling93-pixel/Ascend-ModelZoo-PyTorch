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

import argparse
import os
import time
from pathlib import Path

from tqdm import tqdm

from paddleocr import PaddleOCRVL


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PDF_EXTS = {".pdf"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="OmniDocBenchV1.5/pdfs",
        help="Input path: a directory, a pdf file, or an image file",
    )
    parser.add_argument("--output_path", type=str, default="OmniDocBenchV1.5_out_pdf")
    parser.add_argument("--layout_detection_model_name", type=str, default="PP-DocLayoutV2")
    parser.add_argument("--layout_detection_model_dir", type=str, default="PP-DocLayoutV2")
    parser.add_argument("--vllm_ip", type=str, default="http://127.0.0.1:8000/v1")

    return parser.parse_args()


def is_supported_input_file(p: Path) -> bool:
    if not p.is_file():
        return False
    return p.suffix.lower() in (IMG_EXTS | PDF_EXTS)


def list_input_files(input_path: str) -> list[str]:
    """
    Accepts either:
      - a single pdf/image file path, or
      - a directory containing pdf/image files (non-recursive)

    Returns a sorted list of file paths.
    """
    input_path_obj = Path(input_path)
    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path_obj.is_file():
        if not is_supported_input_file(input_path_obj):
            raise ValueError(f"Unsupported file type: {input_path_obj}")
        return [str(input_path_obj)]

    candidate_files = [p for p in input_path_obj.iterdir() if is_supported_input_file(p)]
    candidate_files.sort(key=lambda p: p.name)
    return [str(p) for p in candidate_files]


if __name__ == "__main__":
    args = parse_args()

    pipeline = PaddleOCRVL(
        layout_detection_model_name=args.layout_detection_model_name,
        layout_detection_model_dir=args.layout_detection_model_dir,
        vl_rec_backend="vllm-server",
        vl_rec_server_url=args.vllm_ip,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        device="npu",
    )

    inputs = list_input_files(args.data_dir)
    print(f"Found {len(inputs)} file(s) from: {args.data_dir}")

    start_time = time.time()
    for res in tqdm(pipeline.predict_iter(inputs)):
        res.save_to_markdown(args.output_path)
    end_time = time.time()

    total_time = end_time - start_time

    total_files = len(inputs)
    print("\n========== Summary ==========")
    print(f"Total files processed: {total_files}")
    print(f"Total time (s): {total_time:.2f}")
    if total_files > 0:
        print(f"Avg time per file (s): {total_time / total_files:.3f}")

