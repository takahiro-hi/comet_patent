import json, os, sys

sys.path.append("./scripts/utils")
from load_data import load_silver


"""
各種データセットのサマリー情報をマークダウン形式で出力
"""



def summarize_patent():

    with open(os.path.join(settings["dir_path"]["patent"]["gold"], "情報系_ADV.json"), "r") as f:
        gold_data_info = json.load(f)
    head_info, tail_info = [d["head"]["content"] for d in gold_data_info], [d["tail"]["content"] for d in gold_data_info]
    head_type_info, tail_type_info = [d["head"]["type"] for d in gold_data_info], [d["tail"]["type"] for d in gold_data_info]

    with open(os.path.join(settings["dir_path"]["patent"]["gold"], "化学系_ADV.json"), "r") as f:
        gold_data_chem = json.load(f)
    head_chem, tail_chem = [d["head"]["content"] for d in gold_data_chem], [d["tail"]["content"] for d in gold_data_chem]
    head_type_chem, tail_type_chem = [d["head"]["type"] for d in gold_data_chem], [d["tail"]["type"] for d in gold_data_chem]

    outputs = f"""## Patent

### GOLD data

#### Data Size:

| Dataset | Total Size | Unique Head | Unique Tail |
|---------|-----------|-------------|-------------|
| 情報系   | {len(gold_data_info):^10} | {len(set(head_info)):^11} | {len(set(tail_info)):^11} |
| 化学系   | {len(gold_data_chem):^10} | {len(set(head_chem)):^11} | {len(set(tail_chem)):^11} |

### Tag Type:

|       |      | Method | Effect | user Effect | tech Effect |
|-------|------|--------|--------|-------------|-------------|
| 情報系  | Head | {head_type_info.count("method"):^6} | {head_type_info.count("effect"):^6} | {head_type_info.count("user_effect"):^11} | {head_type_info.count("tech_effect"):^11} |
|       | Tail | {tail_type_info.count("method"):^6} | {tail_type_info.count("effect"):^6} | {tail_type_info.count("user_effect"):^11} | {tail_type_info.count("tech_effect"):^11} |
| 化学系  | Head | {head_type_chem.count("method"):^6} | {head_type_chem.count("effect"):^6} | {head_type_chem.count("user_effect"):^11} | {head_type_chem.count("tech_effect"):^11} |
|       | Tail | {tail_type_chem.count("method"):^6} | {tail_type_chem.count("effect"):^6} | {tail_type_chem.count("user_effect"):^11} | {tail_type_chem.count("tech_effect"):^11} |
"""

    silver = load_silver(settings, "patent", "情報系")
    outputs += f"""### SILVER data

#### Data Size:

| Dataset | Total Size | Unique Head | Unique Tail |
|---------|-----------|-------------|-------------|
| 情報系   | {len(silver):^10} | {len(set([d[0] for d in silver])):^11} | {len(set([d[1] for d in silver])):^11} |
"""

    return outputs


def main(settings):

    outputs = "# summary\n\n"
    outputs += summarize_patent()

    with open(os.path.join(settings["dir_path"]["data"], "summary.md"), "w") as f:
        f.write(outputs)



if __name__ == '__main__':

    with open("./settings_data.json", "r") as f:
        settings = json.load(f)
    
    main(settings)