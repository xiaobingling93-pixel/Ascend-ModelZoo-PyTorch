# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from bidi.algorithm import get_display

import paddlex
from paddleocr import PaddleOCR
from paddlex.inference.common.reader import ReadImage
from paddlex.inference.models.base import BasePredictor
from paddlex.inference.models.text_detection import TextDetPredictor
from paddlex.inference.models.text_recognition import TextRecPredictor

from ais_bench.infer.interface import InferSession


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="general_ocr_002.png")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--det_model_name", type=str, default="PP-OCRv5_server_det")
    parser.add_argument(
        "--det_model_dir", type=str, default="PP-OCRv5_server_det_infer"
    )
    parser.add_argument("--rec_model_name", type=str, default="PP-OCRv5_server_rec")
    parser.add_argument(
        "--rec_model_dir", type=str, default="PP-OCRv5_server_rec_infer"
    )
    parser.add_argument(
        "--custom_size",
        type=str,
        default=10000000,
        help="aisbench buffer size for dynamic-shape inference",
    )

    return parser.parse_args()


### rewrite functions of the TextDetPredictor in paddlex.inference.models.text_detection
class AscendTextDetPredictor:
    def __init__(
        self,
        limit_side_len: int | None = None,
        limit_type: str | None = None,
        thresh: float | None = None,
        box_thresh: float | None = None,
        unclip_ratio: float | None = None,
        input_shape=None,
        max_side_limit: int = 4000,
        *args,
        **kwargs,
    ):
        BasePredictor.__init__(self, *args, **kwargs)

        self.limit_side_len = limit_side_len
        self.limit_type = limit_type
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.input_shape = input_shape
        self.max_side_limit = max_side_limit
        self.pre_tfs, self.infer, self.post_op = self._build()

        om_path = os.path.join(
            self.model_dir, f"{self.MODEL_FILE_PREFIX}_linux_aarch64.om"
        )
        if not os.path.exists(om_path):
            raise FileNotFoundError(f"Model file {om_path} not found")
        self.om_sess = InferSession(args_ppocr.device_id, om_path)

    def _build(self):
        pre_tfs = {"Read": ReadImage(format="RGB")}

        for cfg in self.config["PreProcess"]["transform_ops"]:
            tf_key = list(cfg.keys())[0]
            func = self._FUNC_MAP[tf_key]
            args = cfg.get(tf_key, {})
            name, op = func(self, **args) if args else func(self)
            if op:
                pre_tfs[name] = op
        pre_tfs["ToBatch"] = paddlex.inference.models.common.ToBatch()
        post_op = self.build_postprocess(**self.config["PostProcess"])
        return pre_tfs, None, post_op

    def process(
        self,
        batch_data: list[str | np.ndarray],
        limit_side_len: int | None = None,
        limit_type: str | None = None,
        thresh: float | None = None,
        box_thresh: float | None = None,
        unclip_ratio: float | None = None,
        max_side_limit: int | None = None,
    ) -> dict[str, object]:
        batch_raw_imgs = self.pre_tfs["Read"](imgs=batch_data.instances)
        batch_imgs, batch_shapes = self.pre_tfs["Resize"](
            imgs=batch_raw_imgs,
            limit_side_len=limit_side_len or self.limit_side_len,
            limit_type=limit_type or self.limit_type,
            max_side_limit=(
                max_side_limit if max_side_limit is not None else self.max_side_limit
            ),
        )
        batch_imgs = self.pre_tfs["Normalize"](imgs=batch_imgs)
        batch_imgs = self.pre_tfs["ToCHW"](imgs=batch_imgs)
        x = self.pre_tfs["ToBatch"](imgs=batch_imgs)

        batch_preds = self.om_sess.infer(
            x, mode="dymshape", custom_sizes=args_ppocr.custom_size
        )

        polys, scores = self.post_op(
            batch_preds,
            batch_shapes,
            thresh=thresh or self.thresh,
            box_thresh=box_thresh or self.box_thresh,
            unclip_ratio=unclip_ratio or self.unclip_ratio,
        )
        return {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": batch_raw_imgs,
            "dt_polys": polys,
            "dt_scores": scores,
        }


### rewrite functions of the TextRecPredictor in paddlex.inference.models.text_recognition
class AscendTextRecPredictor:
    def __init__(self, *args, input_shape=None, return_word_box=False, **kwargs):
        BasePredictor.__init__(self, *args, **kwargs)
        self.input_shape = input_shape
        self.return_word_box = return_word_box
        self.vis_font = self.get_vis_font()
        self.pre_tfs, self.infer, self.post_op = self._build()

        om_path = os.path.join(
            self.model_dir, f"{self.MODEL_FILE_PREFIX}_linux_aarch64.om"
        )
        if not os.path.exists(om_path):
            raise FileNotFoundError(f"Model file {om_path} not found")
        self.om_sess = InferSession(args_ppocr.device_id, om_path)

    def _build(self):
        pre_tfs = {"Read": ReadImage(format="RGB")}
        for cfg in self.config["PreProcess"]["transform_ops"]:
            tf_key = list(cfg.keys())[0]
            if tf_key not in self._FUNC_MAP:
                raise KeyError(
                    f"Unknown transform op '{tf_key}' in PreProcess.transform_ops. "
                    f"Supported ops: {list(self._FUNC_MAP.keys())}"
                )
            func = self._FUNC_MAP[tf_key]
            args = cfg.get(tf_key, {})
            name, op = func(self, **args) if args else func(self)
            if op:
                pre_tfs[name] = op
        pre_tfs["ToBatch"] = (
            paddlex.inference.models.text_recognition.processors.ToBatch()
        )
        post_op = self.build_postprocess(**self.config["PostProcess"])
        return pre_tfs, None, post_op

    def process(self, batch_data, return_word_box=False) -> dict[str, object]:
        batch_raw_imgs = self.pre_tfs["Read"](imgs=batch_data.instances)
        width_list = []
        for img in batch_raw_imgs:
            width_list.append(img.shape[1] / float(img.shape[0]))
        indices = np.argsort(np.array(width_list))
        batch_imgs = self.pre_tfs["ReisizeNorm"](imgs=batch_raw_imgs)
        x = self.pre_tfs["ToBatch"](imgs=batch_imgs)

        batch_preds = self.om_sess.infer(
            x, mode="dymshape", custom_sizes=args_ppocr.custom_size
        )

        batch_num = self.batch_sampler.batch_size
        img_num = len(batch_raw_imgs)
        rec_image_shape = next(
            op["RecResizeImg"]["image_shape"]
            for op in self.config["PreProcess"]["transform_ops"]
            if "RecResizeImg" in op
        )
        imgC, imgH, imgW = rec_image_shape[:3]
        max_wh_ratio = imgW / imgH
        end_img_no = min(img_num, batch_num)
        wh_ratio_list = []
        for ino in range(0, end_img_no):
            h, w = batch_raw_imgs[indices[ino]].shape[0:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
            wh_ratio_list.append(wh_ratio)
        texts, scores = self.post_op(
            batch_preds,
            return_word_box=return_word_box or self.return_word_box,
            wh_ratio_list=wh_ratio_list,
            max_wh_ratio=max_wh_ratio,
        )
        if self.model_name in (
            "arabic_PP-OCRv3_mobile_rec",
            "arabic_PP-OCRv5_mobile_rec",
        ):
            texts = [get_display(s) for s in texts]
        return {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": batch_raw_imgs,
            "rec_text": texts,
            "rec_score": scores,
            "vis_font": [self.vis_font] * len(batch_raw_imgs),
        }


def patch_paddlex():
    ## for TextDetPredictor
    TextDetPredictor.__init__ = AscendTextDetPredictor.__init__
    TextDetPredictor._build = AscendTextDetPredictor._build
    TextDetPredictor.process = AscendTextDetPredictor.process

    ## for TextDetPredictor
    TextRecPredictor.__init__ = AscendTextRecPredictor.__init__
    TextRecPredictor._build = AscendTextRecPredictor._build
    TextRecPredictor.process = AscendTextRecPredictor.process


if __name__ == "__main__":
    args_ppocr = parse_args()
    patch_paddlex()

    ocr = PaddleOCR(
        text_detection_model_name=args_ppocr.det_model_name,
        text_detection_model_dir=args_ppocr.det_model_dir,
        text_recognition_model_name=args_ppocr.rec_model_name,
        text_recognition_model_dir=args_ppocr.rec_model_dir,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    save_results = []
    for res in ocr.predict_iter(args_ppocr.image_dir):
        res.print()
        res.save_to_img("output")
        res.save_to_json("output")
