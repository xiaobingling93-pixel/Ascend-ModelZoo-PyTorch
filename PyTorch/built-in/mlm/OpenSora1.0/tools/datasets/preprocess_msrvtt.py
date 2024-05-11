# Copyright 2024 Huawei Technologies Co., Ltd
# coding=utf-8
import csv
import json
import argparse


def trans(path):
    jsonfile = path
    csvfile = path.replace(".json", ".csv")
    with open(jsonfile, "r") as f:
        json_obj = json.loads(f.read())

    with open(csvfile, "w") as f:
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_ALL)
        for row in json_obj:
            writer.writerow(list(row.values()))


def parse_args():
    parser = argparse.ArgumentParser()

    # model config
    parser.add_argument("--data_path", help="data annotation file path")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trans(args.data_path)