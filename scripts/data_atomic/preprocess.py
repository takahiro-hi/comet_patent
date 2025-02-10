import json, os

def reshape(data):
    return data.replace("X", "人物X")


with open("./settings.json", "r") as f:
    config = json.load(f)

path = os.path.join(config["directories"]["atomic"]["gold_data"], "data.jsonl")
with open(path) as f:
    raw_data = [json.loads(line) for line in f.readlines()]

selected_data = []
for data in raw_data:
    _data_list = [(data["event"], tail) for tail in data["inference"]["event"]["after"]]
    _data_list = [(reshape(head), reshape(tail)) for head, tail in _data_list]
    _data_list = list(set(_data_list))
    selected_data += _data_list

save_path = os.path.join(config["directories"]["atomic"]["gold_data"], "data_after.json")
with open(save_path, "w") as f:
    f.write(json.dumps(selected_data, indent=4, ensure_ascii=False))