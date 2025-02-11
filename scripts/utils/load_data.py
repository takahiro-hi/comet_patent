import os, json



def to_input_format(head, tail):

    assert len(head) != 0 or len(tail) != 0, "Either head or tail must be non-empty."

    head = head[:-1] if head[-1] == "。" else head
    tail = tail[:-1] if tail[-1] == "。" else tail

    return [head, tail]


def load_gold(settings, data_type, patent_domain):

    assert data_type in ["patent", "atomic"], "data_type must be either 'patent' or 'atomic'."
    assert patent_domain in ["情報系", "化学系"], "patent_domain must be either '情報系' or '化学系'."

    if data_type == "patent":
        data_path = os.path.join(settings["dir_path"][data_type]["gold"], f"{patent_domain}_ADV.json")
    elif data_type == "atomic":
        data_path = os.path.join(settings["dir_path"][data_type]["gold"], f"{patent_domain}_ADV.json")
    
    with open(data_path, "r") as f:
        data = json.load(f)
    
    ret_data = [to_input_format(d["head"]["content"], d["tail"]["content"]) for d in data]

    return ret_data



if __name__ == "__main__":

    with open("./settings_data.json", "r") as f:
        settings = json.load(f)

    data = load_gold(settings, "patent", "情報系")
    print(data[:10])
