import os, json, sys, random
import pandas as pd


def to_input_format(head, tail):

    if len(head) != 0:
        head = head[:-1] if head[-1] == "。" else head
        
    if len(tail) != 0:
        tail = tail[:-1] if tail[-1] == "。" else tail

    return [head, tail]


def load_eval_data(settings, temperature, seen_unseen, data_type, flag_concat=False):

    data = pd.read_excel(os.path.join(settings["directories"]["patent"]["eval"], f"{data_type}_eval_sheets.xlsx"), sheet_name=None)
    data_df = data[f"{seen_unseen}_{temperature}"]

    ret_data, ret_label = [], []

    for row in data_df.itertuples():
        if type(row.head)==str and type(row.tail)==str:
            if row.Label in ["positive", "negative"]:
                ret_data.append(to_input_format(row.head, row.tail))
                ret_label.append(1 if row.Label=="positive" else 0)
            elif row.Label=="N / A" or type(row.Label)==float:
                continue
            else:
                assert False, f"invalid label: {row.Label}"
    
    if flag_concat:
        ret_data = [d[0] + "; " + d[1] for d in ret_data]
    
    return ret_data, ret_label


def load_atomic(config, relation="after"):

    assert relation=="after", f"invalid relation: {relation}"

    with open(os.path.join(config["directories"]["atomic"]["gold_data"], f"data_{relation}.json"), "r") as f:
        gold = json.load(f)
    
    with open(os.path.join(config["directories"]["atomic"]["silver_data"], f"triple.json"), "r") as f:
        silver = json.load(f)
    
    return [to_input_format(d[0], d[1]) for d in gold], [to_input_format(d["head"], tail) for d in silver for tail in d["tail"]]


def load_gold(config, gold_type, flag_concat, patent_type=None):

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

    if flag_concat:
        ret_data = [d[0] + "; " + d[1] for d in ret_data]
    
    return ret_data