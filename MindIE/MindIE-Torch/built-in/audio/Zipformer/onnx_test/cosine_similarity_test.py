# Copyright(C) 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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

from scipy.spatial.distance import cosine

def read_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read().split()
        numbers = [float(num) for num in data]
    return numbers

def cosine_similarity(list1, list2):
    return 1 - cosine(list1, list2)

src_enc_path = "./PT/encoder.txt"
src_dec_path = "./PT/decoder.txt"
src_join_path = "./PT/joiner.txt"

tgt_enc_path = "./ONNX/encoder.txt"
tgt_dec_path = "./ONNX/decoder.txt"
tgt_join_path = "./ONNX/joiner.txt"

src_enc = read_file(src_enc_path)
src_dec = read_file(src_dec_path)
src_join = read_file(src_join_path)

tgt_enc = read_file(tgt_enc_path)
tgt_dec = read_file(tgt_dec_path)
tgt_join = read_file(tgt_join_path)

similarity = cosine_similarity(src_enc, tgt_enc)
print(f"Cosine Similarity ENCODER: {similarity}")
similarity = cosine_similarity(src_dec, tgt_dec)
print(f"Cosine Similarity DECODER: {similarity}")
similarity = cosine_similarity(src_join, tgt_join)
print(f"Cosine Similarity JOINER: {similarity}")
