import os, json, random



def to_input_format(head, tail):

    assert len(head) != 0 or len(tail) != 0, "Either head or tail must be non-empty."

    head = head[:-1] if head[-1] == "。" else head
    tail = tail[:-1] if tail[-1] == "。" else tail

    return (head, tail)


def load_gold(settings_data, data_type, patent_domain, shuffle):

    if data_type == "patent":
        assert patent_domain in ["情報系", "化学系"], "patent_domain must be either '情報系' or '化学系'."

        with open(os.path.join(settings_data["dir_path"][data_type]["gold"], f"{patent_domain}_ADV.json"), "r") as f:
            data = json.load(f)
        ret_data = [to_input_format(d["head"]["content"], d["tail"]["content"]) for d in data]

    elif data_type == "atomic":
        with open(os.path.join(settings_data["dir_path"][data_type]["gold"], "xeffect.json"), "r") as f:
            data = json.load(f)
        ret_data = [to_input_format(d[0], d[1]) for d in data]
    
    else:
        assert False, "Invalid data_type."
    
    if shuffle:
        random.shuffle(ret_data)

    return ret_data


def load_silver(settings_data, data_type, temperature, patent_domain, shuffle):

    if data_type == "patent":
        assert patent_domain in ["情報系", "化学系"], "patent_domain must be either '情報系' or '化学系'."

        with open(os.path.join(settings_data["dir_path"]["patent"]["silver"], f"{patent_domain}_triple_{temperature}.json"), "r") as f:
            data = json.load(f)
        ret_data = [to_input_format(d["head"], tail) for d in data for tail in d["tail"] if len(tail) > 0]
    
    elif data_type == "atomic":
        with open(os.path.join(settings_data["dir_path"]["atomic"]["silver"], f"triple.json"), "r") as f:
            data = json.load(f)
        ret_data = [to_input_format(d["head"], tail) for d in data for tail in d["tail"] if len(tail) > 0]

    if shuffle:
        random.shuffle(ret_data)

    return ret_data



def get_settings(filter_type):

    with open("./settings_data.json", "r") as f:
        settings_data = json.load(f)
    with open("./settings_model.json", "r") as f:
        settings_model = json.load(f)
    
    assert filter_type in ["base", "adv", "comet"], "invalid filter_type"
    assert settings_model["params_base"]["model_name"] == settings_model["params_adv"]["model_name"], "model_name is not matched"
    
    settings = {
        "data": settings_data,
        "result_path": settings_model["result_path"],
        "tr_params": settings_model[f"params_{filter_type}"]
    }

    return settings