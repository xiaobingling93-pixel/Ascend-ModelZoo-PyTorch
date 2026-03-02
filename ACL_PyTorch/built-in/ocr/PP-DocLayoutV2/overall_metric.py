import os
import json
import argparse

import pandas as pd
import numpy as np


def deep_get(data, keys, default=np.nan):
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def load_metrics(result_path: str):
    with open(result_path, "r") as file_handle:
        result = json.load(file_handle)

    metric_specs = [
        ("text_block", "Edit_dist", "all", "ALL_page_avg", np.nan, lambda value: float(value)),
        ("display_formula", "CDM", "page", "ALL", 0.0, lambda value: float(value) * 100),
        ("table", "TEDS", "page", "ALL", 0.0, lambda value: float(value) * 100),
        ("table", "TEDS_structure_only", "page", "ALL", 0.0, lambda value: float(value) * 100),
        ("reading_order", "Edit_dist", "all", "ALL_page_avg", np.nan, lambda value: float(value)),
    ]

    row = {}
    for category_type, metric_name, level_name, leaf_key, default_value, post_process in metric_specs:
        if level_name == "page":
            value = deep_get(result, [category_type, "page", metric_name, leaf_key], default=default_value)
        else:
            value = deep_get(result, [category_type, "all", metric_name, leaf_key], default=default_value)

        if value is None:
            value = default_value

        row[f"{category_type}_{metric_name}"] = post_process(value)

    return row


def main():
    parser = argparse.ArgumentParser(description="result path")
    parser.add_argument("--result", type=str, default="OmniDocBench/result")
    parser.add_argument("--match_name", type=str, default="quick_match")
    args = parser.parse_args()

    ocr_types = ["end2end"]  # Add more types here if needed

    rows = []
    index_names = []

    for ocr_type in ocr_types:
        result_path = os.path.join(args.result, f"{ocr_type}_{args.match_name}_metric_result.json")
        if not os.path.exists(result_path):
            print(f"[WARN] missing: {result_path}")
            continue

        rows.append(load_metrics(result_path))
        index_names.append(ocr_type)

    dataframe = pd.DataFrame(rows, index=index_names).round(3)

    dataframe["overall"] = (
        (1 - dataframe["text_block_Edit_dist"]) * 100
        + dataframe["display_formula_CDM"]
        + dataframe["table_TEDS"]
    ) / 3

    print(dataframe)


if __name__ == "__main__":
    main()
