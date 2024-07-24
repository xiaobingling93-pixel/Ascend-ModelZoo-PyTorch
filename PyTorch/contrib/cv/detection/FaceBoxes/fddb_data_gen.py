# Copyright 2024 Huawei Technologies Co., Ltd
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
import random

image_file_list = []


def walk_path(cur_path: str, ground_truth_info: dict):
    """
    循环遍历文件夹，直到最低层，扫描出所有jpg文件
    :param cur_path:
    """
    child_dirs = os.listdir(cur_path)
    for child_dir in child_dirs:
        abs_path = os.path.join(cur_path, child_dir)
        if os.path.isdir(abs_path):
            walk_path(abs_path, ground_truth_info)
        else:
            compare_path = abs_path[7:]
            for key in ground_truth_info.keys():
                if compare_path in ground_truth_info[key]:
                    global image_file_list
                    image_file_list.append(abs_path)


def write_txt(file_name: str, file_number: int):
    """
    将文件列表写入txt中
    :param file_name:
    """
    with open(file_name, 'w') as f_write:
        random.shuffle(image_file_list)
        for image_file in image_file_list[:file_number]:
            image_key = image_file[:-4]
            f_write.write(image_key[7:])
            f_write.write("\n")
        f_write.flush()


def read_txt_file(file_name: str) -> list:
    file_name_list = []
    with open(file_name, 'r') as file_fin:
        file_contents = file_fin.readlines()
        for file_con in file_contents:
            file_con = file_con.strip() + ".jpg"
            file_name_list.append(file_con)
    return file_name_list


def read_ground_truth(ground_truth_dir: str) -> dict:
    """
    获取ground truth dict
    """
    ground_truth_info = {}
    for file_index in range(1, 11):
        file_key = "FDDB-fold-%02d.txt" % file_index
        file_name = os.path.join(ground_truth_dir, file_key)
        file_name_list = read_txt_file(file_name)
        ground_truth_info[file_index] = file_name_list
    return ground_truth_info


if __name__ == "__main__":
    """
    FDDB数据集处理
    test_img_number 测试数据集数量    
    """
    ground_truth = read_ground_truth("ground_truth")
    test_img_number = 2000
    walk_path("images", ground_truth)
    write_txt("img_list.txt", test_img_number)
