import os, json, re
import pandas as pd

from tqdm import tqdm
from collections import defaultdict

from utils import load_row_ann






def load_page(path):

    tag_data, rel_data = load_row_ann(path)
    ret_data = defaultdict(list)
    ret_adv = []
    tail_tag = ["tech_effect", "user_effect", "effect"]

    for key, value in rel_data.items():
        ret_data[value["type"]].append({
            "arg1_data": tag_data[value["arg1"]]["content"],
            "arg2_data": tag_data[value["arg2"]]["content"],
            "arg1_type": tag_data[value["arg1"]]["type"],
            "arg2_type": tag_data[value["arg2"]]["type"]
        })

    dep_data = {}
    for key, value in rel_data.items():
        if value["type"] == "DEP":
            dep_data[value["arg2"]] = value["arg1"]
    for key, value in rel_data.items():
        if value["type"] == "ADV" and tag_data[value["arg2"]]["type"] in tail_tag:
            head = tag_data[dep_data[value["arg1"]]]["content"]+"は、"+tag_data[value["arg1"]]["content"] if value["arg1"] in dep_data else tag_data[value["arg1"]]["content"]
            tail = tag_data[dep_data[value["arg2"]]]["content"]+"は、"+tag_data[value["arg2"]]["content"] if value["arg2"] in dep_data else tag_data[value["arg2"]]["content"]
            ret_adv.append((head, tail))

    return ret_data, ret_adv


def main(config):

    for data_type in ["情報系"]:
        raw_data_path = os.path.join(config["directories"]["patent"]["raw_data"], data_type)
        ann_files = [os.path.join(raw_data_path, f) for f in os.listdir(raw_data_path) if f.endswith(".ann")]

        compiled_data = defaultdict(list)
        adv_data = []
        for ann_file in tqdm(ann_files, desc="compiling raw data..."):
            if os.path.getsize(ann_file) == 0:
                continue
        
            _page_data, _page_adv = load_page(ann_file)
            for key, value in _page_data.items():
                compiled_data[key].extend(value)
            adv_data.extend(_page_adv)
                        
        os.makedirs(config["directories"]["patent"]["gold_data"], exist_ok=True)
        with open(os.path.join(config["directories"]["patent"]["gold_data"], f"{data_type}.json"), "w") as f:
            f.write(json.dumps(compiled_data, indent=4, ensure_ascii=False))
        with open(os.path.join(config["directories"]["patent"]["gold_data"], f"{data_type}_adv.json"), "w") as f:
            f.write(json.dumps(adv_data, indent=4, ensure_ascii=False))



if __name__ == "__main__":

    with open("./settings.json", "r") as f:
        config = json.load(f)

    main(config)