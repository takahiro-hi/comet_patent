import json, os, sys, torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image

sys.path.append("./scripts/utils")
from load_data import to_input_format



def tokenize_data(data, tokenizer, device="cuda"):

    return tokenizer(data, return_tensors='pt', padding="longest").to(device)



def plot_x_iters(values, save_file_path, save_value_flag, title, font_size_dict=None):

    if save_value_flag:
        with open(save_file_path+".json", "w") as f:
            f.write(json.dumps(values, indent=4, ensure_ascii=False))
    
    if font_size_dict is None:
        font_size_dict = {
            "title": 15,
            "label": {"x": 13, "y": 13},
            "ticks": {"x": 11, "y": 11},
            "legend": 13
        }

    x_label = "iteration"
    keys = list(values[list(values.keys())[0]].keys())
    assert len(keys) == 2 and x_label in keys, "invalid data format"
    y_label = list(set(keys) - set([x_label]))[0]

    plt.figure(figsize=(10, 6))

    for key, data_dict in values.items():
        plt.plot(data_dict[x_label], data_dict[y_label], label=key)
    
    plt.title(title, fontsize=font_size_dict["title"])
    plt.xlabel(x_label, fontsize=font_size_dict["label"]["x"])
    plt.ylabel(y_label, fontsize=font_size_dict["label"]["y"])
    plt.grid(True)
    plt.xticks(fontsize=font_size_dict["ticks"]["x"])
    plt.yticks(fontsize=font_size_dict["ticks"]["y"])
    plt.legend(fontsize=font_size_dict["legend"])
    plt.tight_layout()
    plt.savefig(save_file_path+".png")
    plt.close()



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



def plot_patent_metrics(val_patent, model_type, save_file_path):

    
    plt.figure(figsize=(10, 6))
    
    for metric_name, values in val_patent[model_type].items():
        plt.plot(val_patent["iteration"], values, label=f"{model_type} {metric_name}")
    
    plt.xlabel("Iteration")
    plt.ylabel("Metric Value")
    plt.title(f"Metrics for {model_type}")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_file_path)



def return_eval_data(temperature, settings_data, patent_domain):

    data = pd.read_excel(os.path.join(settings_data["dir_path"]["patent"]["eval"], f"{patent_domain}_eval_sheets.xlsx"), sheet_name=None)

    ret_data = {"data": [], "label": []}
    
    for data_type in ["seen", "unseen"]:
        _data_df = data[f"{data_type}_{temperature}"]
        _data_df = _data_df[["head", "tail", "Label"]]
        _data_df = _data_df.dropna()

        for row in _data_df.itertuples():
            if row.Label in ["positive", "negative"]:
                ret_data["data"].append(to_input_format(row.head, row.tail))
                ret_data["label"].append(1. if row.Label == "positive" else 0.)
            elif row.Label == "N / A":
                pass
            else:
                raise ValueError(f"{row.Label} is invalid label")

    return ret_data



def _pred(model, tokenizer, data, label):

    model.eval()
    
    inps = tokenize_data(data, tokenizer)
    with torch.no_grad():
        pre_logits = model(**inps).squeeze(-1)
        pre_prob = torch.sigmoid(pre_logits)
        pre_label = (pre_prob >= 0.5).long()
    
    acc = accuracy_score(label, pre_label.cpu().numpy())
    precision = precision_score(label, pre_label.cpu().numpy(), zero_division=0)
    recall = recall_score(label, pre_label.cpu().numpy(), zero_division=0)
    f1 = f1_score(label, pre_label.cpu().numpy(), zero_division=0)

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}



def test_adv_base(base_model, adv_model, tokenizer, settings_data, patent_domain, result_path):

    os.makedirs(os.path.join(result_path, "temp"), exist_ok=True)

    file_name_list = []
    for temperature in ["0.8", "1.0", "1.3"]:

        test_data = return_eval_data(temperature, settings_data, patent_domain)

        # base_model
        base_metrics = _pred(base_model, tokenizer, test_data["data"], test_data["label"])
        # adv_model
        adv_metrics = _pred(adv_model, tokenizer, test_data["data"], test_data["label"])

        file_name = os.path.join(result_path, "temp", f"{temperature}.png")
        file_name_list.append(file_name)
        title = f"temperautre {temperature}"
        _plot_metrics(base_metrics, adv_metrics, title, file_name)
    
    _rearrange_image(file_name_list, os.path.join(result_path, "metrics.png"))



def _plot_metrics(base_metrics, adv_metrics, title, save_file_path):

    metrics = ["accuracy", "precision", "recall", "f1"]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, [base_metrics[metric] for metric in metrics], width, label="base")
    rects2 = ax.bar(x + width/2, [adv_metrics[metric] for metric in metrics], width, label="adv")

    ax.set_ylabel("Scores", fontsize=13)
    ax.set_title("Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=13)
    ax.grid(axis="y")
    ax.legend(fontsize=13)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    plt.savefig(save_file_path)
    plt.close()



def _rearrange_image(image_list, save_file_path):
    """
    画像を横並びにした画像を作成する
    """
    images = [Image.open(path) for path in image_list]

    widths, heights = images[0].size

    new_width = widths * len(images)
    new_height = heights

    new_image = Image.new("RGB", (new_width, new_height), color="white")
    for i, img in enumerate(images):
        new_image.paste(img, (i*widths, 0))
    
    new_image.save(save_file_path)
