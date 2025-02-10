import os, json, re

from tqdm import tqdm

from compile import load_row_ann





def check_each_row(text):

    pattern = r"[#TR]\d+\s"
    match = re.match(pattern, text)

    if match:
        return True
    else:
        return False


def check_ann_file(tag_data, rel_data, txt_file_path, ann_file_path):

    TAGs = ["method", "purpose", "problem", "effect", "tech_effect", "user_effect", "bg", "others"]
    RELs = ["ADV", "CON", "EQU", "CHAIN", "PENDING", "DEP"]

    # where each raw of ann is valied
    for line in open(ann_file_path, "r"):
        assert check_each_row(line), f"invalid row: {line}"

    # whether tag / rel type is supported
    for key, value in tag_data.items():
        assert value["type"] in TAGs, f"invalid tag type: {value['type']}, ({ann_file_path})"
    for key, value in rel_data.items():
        assert value["type"] in RELs, f"invalid relation type: {value['type']}"
    
    # whether arg1 and arg2 are in tag_data
    tag_ids = set(tag_data.keys())
    for key, value in rel_data.items():
        assert value["arg1"] in tag_ids and value["arg2"] in tag_ids, f"arg1 or arg2 is not in tag_data"
    
    # whether id is unique
    assert len(tag_data.keys()) == len(set(tag_data.keys())), "tag id is not unique"
    assert len(rel_data.keys()) == len(set(rel_data.keys())), "rel id is not unique"

    # whether arg1 and arg2 are different
    for key, value in rel_data.items():
        assert value["arg1"] != value["arg2"], "arg1 and arg2 are same"
    
    # whether loc is correct
    txt_data = open(txt_file_path, "r").read()
    for key, value in tag_data.items():
        s1, e1 = value["loc1"]
        s2, e2 = value["loc2"]
        if s2 == -1 and e2 == -1:
            _text = txt_data[s1:e1]
            assert _text == value["content"], f"content is not matched with loc: {_text} vs {value['content']}, ({txt_file_path})"
        else:
            _text = txt_data[s1:e1] + txt_data[s2:e2]
            assert _text == value["content"], f"content is not matched with loc: {_text} vs {value['content']}, ({txt_file_path})"



def main(config):

    for data_type in ["情報系", "化学系"]:
        dir_path = os.path.join(config["directories"]["patent"]["raw_data"], data_type)
        txt_list = [f for f in os.listdir(dir_path) if f.endswith(".txt")]
        ann_list = [f for f in os.listdir(dir_path) if f.endswith(".ann")]

        txt_list.sort()
        ann_list.sort()

        assert len(txt_list) == len(ann_list), "txt and ann file is not matched"

        for txt, ann in zip(tqdm(txt_list), ann_list):
            
            if os.path.getsize(os.path.join(dir_path, ann)) == 0:
                continue
            assert os.path.splitext(txt)[0] == os.path.splitext(ann)[0], "txt and ann file is not matched"

            tag_data, rel_data = load_row_ann(os.path.join(dir_path, ann))

            with open("./_tag.json", "w") as f:
                f.write(json.dumps(tag_data, indent=4, ensure_ascii=False))
            with open("./_rel.json", "w") as f:
                f.write(json.dumps(rel_data, indent=4, ensure_ascii=False))

            check_ann_file(tag_data, rel_data, os.path.join(dir_path, txt), os.path.join(dir_path, ann))
            
            
            




if __name__ == "__main__":

    with open("./settings.json", "r") as f:
        config = json.load(f)

    main(config)