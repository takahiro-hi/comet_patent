import json, sys, torch, os, argparse
from transformers import BertJapaneseTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict

from util_load_data import load_eval_data
from utils_plot import plot_metrics

sys.path.append("./scripts/train")
from classifier import Classifier
from base import tokenize_data



def measure_metrics(label, pred_label):
    
    acc = accuracy_score(label, pred_label)
    precision = precision_score(label, pred_label, zero_division=0)
    recall = recall_score(label, pred_label, zero_division=0)
    f1 = f1_score(label, pred_label, zero_division=0)

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def eval(model, device, tokenizer):

    ret_dict = defaultdict(dict)

    for seen_unseen in ["seen", "unseen"]:
        for temperature_s, temperature_f in zip(["08", "10", "13"], ["0.8", "1.0", "1.3"]):
            data, label = load_eval_data(settings, temperature_f, seen_unseen, data_type, False)

            inps = tokenize_data(data, tokenizer, device)
            with torch.no_grad():
                pred_logits = model(**inps).squeeze(-1)
            pred_prob = torch.sigmoid(pred_logits)
            pred_label = [1 if p >= 0.5 else 0 for p in pred_prob]
            ret_dict[f"{seen_unseen}_{temperature_s}"] = measure_metrics(label, pred_label)
    
    return ret_dict


def main(settings, data_type, device, no_list, flag_predict=False):

    if flag_predict:
        os.makedirs(settings["directories"]["eval"]["metrics"], exist_ok=True)
        result_metrics = defaultdict(dict)

        tokenizer = BertJapaneseTokenizer.from_pretrained(settings["base_train"]["model_name"])
        
        model_path = os.path.join(settings["directories"]["step_1"], f"patent/{data_type}/es/selector.pth")
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model = Classifier(settings["base_train"]["model_name"]).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        result_metrics["base"] = eval(model, device, tokenizer)

        # Adversarial models
        for model_type in ["switch", "no_switch"]:
            for no in no_list:
                model_path = os.path.join(settings["directories"]["step_2"], f"adv/{data_type}/{model_type}/{no}/selector.pth")
                state_dict = torch.load(model_path, map_location=device, weights_only=True)
                model = Classifier(settings["base_train"]["model_name"]).to(device)
                model.load_state_dict(state_dict)
                model.eval()

                result_metrics[f"{model_type}_{no}"] = eval(model, device, tokenizer)
                
        with open(os.path.join(settings["directories"]["eval"]["metrics"], "metrics.json"), "w") as f:
            json.dump(result_metrics, f, indent=4, ensure_ascii=False)

    
    else:
        with open(os.path.join(settings["directories"]["eval"]["metrics"], "metrics.json"), "r") as f:
            result_metrics = json.load(f)

    for data_type in ["seen", "unseen"]:
        for temperature_f, temperature in zip(["0.8", "1.0", "1.3"], ["08", "10", "13"]):
            plot_metrics_data = {}
            for metric in ["accuracy", "precision", "recall", "f1"]:
                plot_metrics_data[metric] = {
                    "Base": [result_metrics["base"][f"{data_type}_{temperature}"][metric]],
                    "Adv (patent)": [result_metrics[f"no_switch_{no}"][f"{data_type}_{temperature}"][metric] for no in no_list],
                    "Adv (atomic_patent)": [result_metrics[f"switch_{no}"][f"{data_type}_{temperature}"][metric] for no in no_list]
                }
            plot_metrics(plot_metrics_data, f"data type: {data_type}\ntemperature: {temperature_f}", os.path.join(settings["directories"]["eval"]["metrics"],f"{data_type}_{temperature}.pdf"))


if __name__ == "__main__":

    data_type = "情報系"
    no_list = ["1", "2", "3", "4", "5"]
    device = "cuda:0"

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--predict", action="store_true")
    args = argparser.parse_args()

    with open("./settings.json", "r") as f:
        settings = json.load(f)
    

    main(settings, data_type, device, no_list, args.predict)
