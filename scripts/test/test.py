import sys, json, os, torch

from transformers import BertJapaneseTokenizer
from sklearn.metrics import accuracy_score

sys.path.append("./scripts/train")
from utils_load_data import load_silver
from classifier import Classifier
from base import tokenize_data


def pred(inps, model_type):

    if model_type == "switch":
        model_path = os.path.join(config["directories"]["step_2"], f"adv/情報系/switch/4/selector.pth")
    elif model_type == "no_switch":
        model_path = os.path.join(config["directories"]["step_2"], f"adv/情報系/no_switch/4/selector.pth")
    elif model_type == "base":
        model_path = os.path.join(config["directories"]["step_1"], f"patent/情報系/es/selector.pth")

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    selector = Classifier(config["adv_train"]["model_name"]).to(device)
    selector.load_state_dict(state_dict)

    with torch.no_grad():
        pre = selector(**inps).squeeze(-1)
    prob = torch.sigmoid(pre)
    pred_label = prob > 0.5

    return prob, pred_label


with open("./settings.json") as f:
    config = json.load(f)
device = "cuda:3"

tokenizer = BertJapaneseTokenizer.from_pretrained(config["adv_train"]["model_name"])

data_patent = load_silver(config, "patent", "情報系")
data_patent = data_patent[:1000]
inps = tokenize_data(data_patent, tokenizer, device)

with open("./selected_silver.json", "w") as f:
    json.dump(data_patent, f, indent=4, ensure_ascii=False)

prob_base, pred_base = pred(inps, "base")
prob_switch, pred_switch = pred(inps, "switch")
prob_no_switch, pred_no_switch = pred(inps, "no_switch")

save_data = []
for idx, data in enumerate(data_patent):
    save_data.append({
        "data": data,
        "base": pred_base[idx].item(),
        "switch": pred_switch[idx].item(),
        "no_switch": pred_no_switch[idx].item()
    })

with open("./1126.json", "w") as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)