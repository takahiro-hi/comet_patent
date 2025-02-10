import os, json, random, random


def to_input_format(head, tail):

    if len(head) != 0:
        head = head[:-1] if head[-1] == "。" else head
        
    if len(tail) != 0:
        tail = tail[:-1] if tail[-1] == "。" else tail

    return [head, tail]


def load_gold(config, gold_type, patent_type=None):

    if gold_type == "atomic":
        path = os.path.join(config["directories"][gold_type]["gold_data"], "data_after.json")
    elif gold_type == "patent":
        path = os.path.join(config["directories"][gold_type]["gold_data"], f"{patent_type}_adv.json")
    else:
        assert False, f"invalid gold_type: {gold_type}"
    with open(path, "r") as f:
        gold = json.load(f)
    
    ret_data_set = set()
    ret_data = []
    for d in gold:
        _data = to_input_format(d[0], d[1])
        _data_tuple = tuple(_data)
        if not _data_tuple in ret_data_set:
            ret_data.append(_data)
            ret_data_set.add(_data_tuple)

    random.shuffle(ret_data)    
    return ret_data


def load_silver(config, gold_type, patent_type=None):

    if gold_type == "atomic":
        path = os.path.join(config["directories"][gold_type]["silver_data"], "triple.json")
    elif gold_type == "patent":
        path = os.path.join(config["directories"][gold_type]["silver_data"], f"{patent_type}_tail.json")
    else:
        assert False, f"invalid gold_type: {gold_type}"
    with open(path, "r") as f:
        silver = json.load(f)
    
    ret_data_set = set()
    ret_data = []
    for dict in silver:
        tail_list = list(set(dict["tail"]))
        for tail in tail_list:
            _data = to_input_format(dict["head"], tail)
            _data_tuple = tuple(_data)
            if not _data_tuple in ret_data_set:
                ret_data.append(_data)
                ret_data_set.add(_data_tuple)
    
    random.shuffle(ret_data)
    return ret_data


def load_train_data_for_step1(config, gold_data, patent_type=None):

    pos = load_gold(config, gold_data, patent_type)
    neg = load_silver(config, gold_data, patent_type)

    print(f"pos: {len(pos)}, neg: {len(neg)}")

    min_size = min(len(pos), len(neg))
    pos = pos[:min_size]
    neg = neg[:min_size]
    
    print(f"pos: {len(pos)}, neg: {len(neg)}")
    
    data = pos + neg
    label = [1.] * len(pos) + [0.] * len(neg)

    combined = list(zip(data, label))
    random.shuffle(combined)
    data[:], label[:] = zip(*combined)

    return data, label
