import os, json



def to_input_format(head, tail):

    assert len(head) != 0 or len(tail) != 0, "Either head or tail must be non-empty."

    head = head[:-1] if head[-1] == "。" else head
    tail = tail[:-1] if tail[-1] == "。" else tail

    return [head, tail]


def load_gold(settings_data, data_type, patent_domain=None):

    if data_type == "patent":
        assert patent_domain in ["情報系", "化学系"], "patent_domain must be either '情報系' or '化学系'."

        with open(os.path.join(settings_data["dir_path"][data_type]["gold"], f"{patent_domain}_ADV.json"), "r") as f:
            data = json.load(f)
        ret_data = [to_input_format(d["head"]["content"], d["tail"]["content"]) for d in data]

    elif data_type == "atomic":
        #data_path = os.path.join(settings_data["dir_path"][data_type]["gold"], f"{patent_domain}_ADV.json")
        assert False, "Not implemented yet."
    
    else:
        assert False, "Invalid data_type."

    return ret_data


def load_silver(settings_data, temperature, data_type, patent_domain=None):

    if data_type == "patent":
        assert patent_domain in ["情報系", "化学系"], "patent_domain must be either '情報系' or '化学系'."

        with open(os.path.join(settings_data["dir_path"][data_type]["silver"], f"{patent_domain}_triple_{temperature}.json"), "r") as f:
            data = json.load(f)
        ret_data = [to_input_format(d["head"], tail) for d in data for tail in d["tail"] if len(tail) > 0]
    
    elif data_type == "atomic":
        assert False, "Not implemented yet."

    else:
        assert False, "Invalid data_type."
    
    return ret_data




if __name__ == "__main__":

    with open("./settings_data.json", "r") as f:
        settings = json.load(f)

    data = load_silver(settings, 1.0, "patent", "情報系")
    print(data[:10])
