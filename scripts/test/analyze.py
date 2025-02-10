import json


with open("./1126.json", "r") as f:
    data = json.load(f)

save_result_1 = {
    "000": 0,
    "001": 0,
    "010": 0,
    "011": 0,
    "100": 0,
    "101": 0,
    "110": 0,
    "111": 0
}

save_result_2 = {
    "00": 0,
    "01": 0,
    "10": 0,
    "11": 0
}

for d in data:
    three = 1 if d["base"] else 0
    two = 1 if d["switch"] else 0
    one = 1 if d["no_switch"] else 0
    save_result_1[f"{three}{two}{one}"] += 1
    save_result_2[f"{two}{one}"] += 1

print(save_result_1)
print(save_result_2)