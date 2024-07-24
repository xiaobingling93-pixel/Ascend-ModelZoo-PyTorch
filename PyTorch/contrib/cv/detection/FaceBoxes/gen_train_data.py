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
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def write_txt(file_name: str, verify_info_list: list):
    """
    :param file_name:
    :param verify_info_list: 验证过的有效xml文件
    输出文件key和xml组成的一行
    """
    with open(file_name, "w") as f_write:
        for data_line, verify_info in enumerate(verify_info_list):
            image_info = os.path.join(verify_info["dir_name"], verify_info["image_name"])
            f_write.write(image_info)
            f_write.write(" ")
            f_write.write(verify_info["xml_file"])
            if data_line < len(verify_info_list) - 1:
                f_write.write("\n")
        f_write.flush()


def verify_voc(xml_path: str):
    """
    :param xml_path:
    检查xml_path中内容是否为空
    """
    xml_content = ET.parse(xml_path)
    if xml_content.find("object"):
        return True
    print("{} has no object".format(xml_path))
    return False


def run(image_path: str, anno_path: str):
    """
    :param image_path:
    :param anno_path:
    启动生成img_list.txt
    """
    image_dir_list = os.listdir(image_path)
    image_info_data = {}
    verify_info_list = []
    for image_dir in image_dir_list:
        image_child_dir = os.path.join(image_path, image_dir)
        all_image_path = os.listdir(image_child_dir)
        for all_image in all_image_path:
            file_key = all_image[:-4]
            image_info = {
                "dir_name": image_dir,
                "image_name": all_image,
                "file_key": file_key,
                "xml_file": ""
            }
            image_info_data[file_key] = image_info
    xml_file_list = os.listdir(anno_path)
    for xml_file in xml_file_list:
        file_key = xml_file[:-4]
        xml_real_path = os.path.join(anno_path, xml_file)
        if file_key in image_info_data.keys() and verify_voc(xml_real_path):
            image_info_data[file_key]["xml_file"] = xml_file
            verify_info_list.append(image_info_data[file_key])
    write_txt("img_list.txt", verify_info_list)


if __name__ == "__main__":
    run("images", "annotations")
