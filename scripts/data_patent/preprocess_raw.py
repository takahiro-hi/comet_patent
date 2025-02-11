import os, json
from collections import defaultdict


"""
アノテーションした特許データを前処理する
"""
TAG_TYPE = ["method", "purpose", "problem", "effect", "tech_effect", "user_effect", "bg", "others"]
REL_TYPE = ["ADV", "CON", "EQU", "CHAIN", "PENDING", "DEP"]

RAW_DATA_PATH_DIC = {
   "情報系": "~/patent/work_data/松井さん/情報系_1",
   "化学系": "~/patent/work_data/松井さん/化学系",
}



def validate_ann_data(tag_data, rel_data, ann_file_path):
    """
    読み込んだ.annファイルについて, 異常がないかをチェックする関数
    """
    tag_id_list = list(tag_data.keys())
    rel_id_list = list(rel_data.keys())

    # 指定外のtag名が含まれていないか確認
    for tag_id, tag_info in tag_data.items():
        assert tag_info["type"] in TAG_TYPE, f"unexpected tag type: {tag_info['type']} {ann_file_path}"

    # 指定外のrel名が含まれていないか確認
    for rel_id, rel_info in rel_data.items():
        assert rel_info["type"] in REL_TYPE, f"unexpected rel type: {rel_info['type']} {ann_file_path}"

    # IDの重複確認
    assert len(tag_id_list) == len(set(tag_id_list)), f"tag ID is duplicated {ann_file_path}"
    assert len(rel_id_list) == len(set(rel_id_list)), f"rel ID is duplicated {ann_file_path}"

    # relのarg1, arg2が存在するtag IDに含まれているか確認
    for rel_id, rel_info in rel_data.items():
        assert rel_info["arg1"] in tag_id_list, f"rel arg1 is not in tag ID: {rel_info['arg1']} {ann_file_path}"
        assert rel_info["arg2"] in tag_id_list, f"rel arg2 is not in tag ID: {rel_info['arg2']} {ann_file_path}"


def extract_target_data(tag_data, rel_data):
    """
    抽出されたアノテーションデータから、対象データを取り出す
    """
    dep_dict = {}
    for _, rel_info in rel_data.items():
        if rel_info["type"] == "DEP":
            dep_dict[rel_info["arg2"]] = rel_info["arg1"]

    ret_list_adv = []
    for _, rel_info in rel_data.items():
        head_id, tail_id = rel_info["arg1"], rel_info["arg2"]

        if tag_data[head_id]["type"]=="method" and rel_info["type"]=="ADV" and tag_data[tail_id]["type"] in ["tech_effect", "user_effect", "effect"]:
            head, tail = tag_data[head_id]["content"], tag_data[tail_id]["content"]
            if head_id in dep_dict:
                if tag_data[dep_dict[head_id]]["content"][-2:] == "は、":
                    head = tag_data[dep_dict[head_id]]["content"] + head
                elif tag_data[dep_dict[head_id]]["content"][-1] == "は":
                    head = tag_data[dep_dict[head_id]]["content"] + "、" + head
                else:
                    head = tag_data[dep_dict[head_id]]["content"] + "は、" + head
            if tail_id in dep_dict:
                if tag_data[dep_dict[tail_id]]["content"][-2:] == "は、":
                    tail = tag_data[dep_dict[tail_id]]["content"] + tail
                elif tag_data[dep_dict[tail_id]]["content"][-1] == "は":
                    tail = tag_data[dep_dict[tail_id]]["content"] + "、" + tail
                else:
                    tail = tag_data[dep_dict[tail_id]]["content"] + "は、" + tail

            ret_list_adv.append({
                "head": {
                    "type": tag_data[head_id]["type"],
                    "content": head
                },
                "tail": {
                    "type": tag_data[tail_id]["type"],
                    "content": tail
                }
            })

    ret_list_all = [
        {
            "relation": rel_info["type"],
            "head": {
                "type": tag_data[rel_info["arg1"]]["type"],
                "content": tag_data[rel_info["arg1"]]["content"]
            },
            "tail": {
                "type": tag_data[rel_info["arg2"]]["type"],
                "content": tag_data[rel_info["arg2"]]["content"]
            }
        } for _, rel_info in rel_data.items()
    ]

    return ret_list_adv, ret_list_all


def load_row_ann(ann_file_path, allow_space=True):
    """
    .annファイルを読み込む

    args:
        ann_file_path: str, .annファイルのパス
        allow_space: bool, データに空白を許容するかどうか

    returns:
        ret_dic: list
            .annファイルのデータを格納したリスト
    """
    tag_data, rel_data = defaultdict(dict), defaultdict(dict)

    for line in open(ann_file_path, "r"):
        
        line = line.strip()

        # load tag
        if line.startswith("T"):
            tag_id, tag_info, content = line.split("\t")
            tag_type = tag_info.split(" ")[0]

            if ";" in tag_info:     # contain line break
                if not allow_space:
                    assert tag_info.count(";") == content.count(" "), f"unexpected space in content \n{ann_file_path}\n{line}"
                content = content.replace(" ", "")
            
            tag_data[tag_id] = {
                "type": tag_type,
                "loc": tag_info.split(" ")[1:],
                "content": content
            }

        # load relation
        elif line.startswith("R"):
            rel_id, rel_info = line.split("\t")
            rel_type, arg1, arg2 = rel_info.split(" ")
            rel_data[rel_id] = {
                "type": rel_type,
                "arg1": arg1.replace("Arg1:", "", 1),
                "arg2": arg2.replace("Arg2:", "", 1)
            }
        
        elif line.startswith("#"):
            pass

        else:
            raise ValueError(f"unexpected line \n{ann_file_path}\n{line}")
    
    validate_ann_data(tag_data, rel_data, ann_file_path)

    return extract_target_data(tag_data, rel_data)


def load_all_ann(dir_path, allow_space=True):
    """
    指定ディレクトリ内の全ての.annファイルを読み込む

    args:
        dir_path: str, ディレクトリのパス
        allow_space: bool, データに空白を許容するかどうか
    
    returns:
        all_ann_data: list
            全ての.annファイルのデータを格納したリスト
    """
    ann_adv_list, ann_all_list = [], []

    files = os.listdir(dir_path)
    ann_file = [f for f in files if f.endswith(".ann")]

    for f in ann_file:
        _ann_adv, _ann_all = load_row_ann(os.path.join(dir_path, f), allow_space)
        ann_adv_list.extend(_ann_adv)
        ann_all_list.extend(_ann_all)
    
    return ann_adv_list, ann_all_list



def main(settings):

    for data_type in ["情報系", "化学系"]:
        dir_path = os.path.expanduser(RAW_DATA_PATH_DIC[data_type])
        ann_data_adv, ann_data_all = load_all_ann(dir_path, allow_space=True)

        with open(os.path.join(settings["dir_path"]["patent"]["gold"], f"{data_type}_ADV.json"), "w") as f:
            json.dump(ann_data_adv, f, indent=4, ensure_ascii=False)
        with open(os.path.join(settings["dir_path"]["patent"]["gold"], f"{data_type}_ALL.json"), "w") as f:
            json.dump(ann_data_all, f, indent=4, ensure_ascii=False)



if __name__ == "__main__":

    with open("./settings_data.json", "r") as f:
        settings = json.load(f)
    
    os.makedirs(settings["dir_path"]["patent"]["gold"])

    main(settings)