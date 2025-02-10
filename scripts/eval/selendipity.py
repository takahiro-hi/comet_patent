import torch, configparser, os, json, sys, tqdm, transformers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import BertJapaneseTokenizer
from janome.tokenizer import Tokenizer
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import defaultdict
from tqdm import tqdm

from utils_plot import plot_heat_map
from util_load_data import load_eval_data, load_gold

sys.path.append("./scripts/train")
from classifier import Classifier
from base import tokenize_data




def measure_bleu_sim(settings, gold_data, silver_data):

    tokenizer = Tokenizer()
    smooth_func = SmoothingFunction().method1
    bleu_scores_matrix = np.zeros((len(silver_data), len(gold_data)))
    for i in tqdm(range(len(silver_data))):
        candidate = [token.surface for token in tokenizer.tokenize(silver_data[i])]
        for j in range(len(gold_data)):
            reference = [[token.surface for token in tokenizer.tokenize(gold_data[j])]]
            bleu_scores_matrix[i, j] = sentence_bleu(reference, candidate, smoothing_function=smooth_func)

    #plot_heat_map(bleu_scores_matrix, save_file)
    results = {}
    for threshold in settings["eval"]["selendipity"]["threshold"]:
        counts_per_silver = np.sum(bleu_scores_matrix >= threshold, axis=1) / len(gold_data)
        average_ratio = np.mean(counts_per_silver)
        results[threshold] = average_ratio
    return results
    

def measure_cos_sim(settings, gold_data, silver_data, device, batch_size=1000):
    
    transformers.BertTokenizer = transformers.BertJapaneseTokenizer
    transformer = models.Transformer('cl-tohoku/bert-base-japanese-whole-word-masking')
    pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode_mean_tokens=True, pooling_mode_cls_token=False, pooling_mode_max_tokens=False)
    model = SentenceTransformer(modules=[transformer, pooling]).to(device)

    embedding_list_silver = model.encode(silver_data, convert_to_tensor=True, device=device)
    embedding_list_gold = model.encode(gold_data, convert_to_tensor=True, device=device)

    cosine_sim_matrix = np.zeros((len(silver_data), len(gold_data)))
    for i in range(0, len(silver_data), batch_size):
        embeddings_s = embedding_list_silver[i:i+batch_size]
        for j in range(0, len(gold_data), batch_size):
            embeddings_g = embedding_list_gold[j:j+batch_size]
            sim_scores = torch.nn.functional.cosine_similarity(embeddings_s.unsqueeze(1), embeddings_g.unsqueeze(0), dim=2)
            cosine_sim_matrix[i:i+batch_size, j:j+batch_size] = sim_scores.cpu().numpy()
    
    #plot_heat_map(cosine_sim_matrix, save_file)
    results = {}
    for threshold in settings["eval"]["selendipity"]["threshold"]:
        counts_per_silver = np.sum(cosine_sim_matrix >= threshold, axis=1) / len(gold_data)
        average_ratio = np.mean(counts_per_silver)
        results[threshold] = average_ratio
    return results


def _predict_given_model(settings, model, tokenizer):

    ret_data = defaultdict(dict)

    for temper_f, temper_s in tqdm(zip(["0.8", "1.0", "1.3"], ["08", "10", "13"]), total=3, desc="Predicting silver data"):
        for seen_unseen in ["seen", "unseen"]:

            eval_silver_data, va_labels = load_eval_data(settings, temper_f, seen_unseen, data_type, flag_concat=True)
            inps = tokenize_data(eval_silver_data, tokenizer, device)

            with torch.no_grad():
                pred_logits = model(**inps).squeeze(-1)
            pred_prob = torch.sigmoid(pred_logits)

            ret_data[f"{seen_unseen}_{temper_s}"] = [1 if p >= 0.5 else 0 for p in pred_prob]
    
    return ret_data


def predict_silver(settings, tokenizer, data_type, device):

    save_data = defaultdict(dict)

    # Base model
    model_path = os.path.join(settings["directories"]["step_1"], f"patent/{data_type}/es/selector.pth")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model = Classifier(settings["base_train"]["model_name"]).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    save_data["base"] = _predict_given_model(settings, model, tokenizer)
    
    # Adversarial models
    for model_type in ["no_switch", "switch"]:
        for no in no_list:
            model_path = os.path.join(settings["directories"]["step_2"], f"adv/{data_type}/{model_type}/{no}/selector.pth")
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model = Classifier(settings["base_train"]["model_name"]).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            save_data[f"{model_type}_{no}"] = _predict_given_model(settings, model, tokenizer)
    
    save_dir = os.path.join(settings["directories"]["eval"]["selendipity"], "predicted_label.json")
    os.makedirs(settings["directories"]["eval"]["selendipity"], exist_ok=True)
    with open(save_dir, "w") as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)


def measure_similarity(settings, data_type, device, eval_silver_data, predicted_label, save_file):

    ret_dict_cos = defaultdict(list)
    ret_dict_bleu = defaultdict(list)

    gold_data = load_gold(settings, "patent", True, data_type)
    all_silver = eval_silver_data

    ret_dict_cos["all_silver"] = measure_cos_sim(settings, gold_data, all_silver, device)
    ret_dict_bleu["all_silver"] = measure_bleu_sim(settings, gold_data, all_silver)

    for model in predicted_label.keys():
        selected_silver = [data for data, label in zip(eval_silver_data, predicted_label[model]) if label==1]

        if model == "base":
            save_key = "base"
        elif model[:len("no_switch")] == "no_switch":
            save_key = "patent"
        elif model[:len("switch")] == "switch":
            save_key = "atomic_patent"
        else:
            assert False, f"invalid model: {model}"
        ret_dict_cos[save_key] = measure_cos_sim(settings, gold_data, selected_silver, device)
        ret_dict_bleu[save_key] = measure_bleu_sim(settings, gold_data, selected_silver)

    save_dir = os.path.join(settings["directories"]["eval"]["selendipity"], save_file)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "cosine_similarity.json"), "w") as f:
        json.dump(ret_dict_cos, f, indent=4, ensure_ascii=False)
    with open(os.path.join(save_dir, "bleu_similarity.json"), "w") as f:
        json.dump(ret_dict_bleu, f, indent=4, ensure_ascii=False)


def plot_selendipity(settings):

    for temper_f, temper_s in tqdm(zip([0.8, 1.0, 1.3], ["08", "10", "13"]), total=3, desc="Measuring similarity"):
        for seen_unseen in ["seen", "unseen"]:
            with open(os.path.join(settings["directories"]["eval"]["selendipity"], f"{data_type}/{seen_unseen}_{temper_s}/cosine_similarity.json"), "r") as f:
                cosine_sim = json.load(f)
            with open(os.path.join(settings["directories"]["eval"]["selendipity"], f"{data_type}/{seen_unseen}_{temper_s}/bleu_similarity.json"), "r") as f:
                bleu_sim = json.load(f)
            _plot(cosine_sim, os.path.join(settings["directories"]["eval"]["selendipity"], f"{data_type}/{seen_unseen}_{temper_s}/cosine_similarity.png"))
            _plot(bleu_sim, os.path.join(settings["directories"]["eval"]["selendipity"], f"{data_type}/{seen_unseen}_{temper_s}/bleu_similarity.png"))


def _plot(data, save_file):

    plt.figure(figsize=(10, 6))

    for key, values in data.items():
        x = [float(k) for k in values.keys()]
        y = list(values.values())
        plt.plot(x, y, label=key, marker='o')

    plt.title("Comparison of Cosine Similarity Ratios", fontsize=14)
    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("Ratio", fontsize=12)
    plt.xticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    plt.legend(title="Keys", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig(save_file)
    plt.close()



def main(settings, data_type, device, no_list):

    tokenizer = BertJapaneseTokenizer.from_pretrained(settings["base_train"]["model_name"])

    # predict silver data for each model
    #predict_silver(settings, tokenizer, data_type, device)
    
    # measure similarity for each model
    with open(os.path.join(settings["directories"]["eval"]["selendipity"], "predicted_label.json"), "r") as f:
        predicted_label = json.load(f)
    for temper_f, temper_s in tqdm(zip([0.8, 1.0, 1.3], ["08", "10", "13"]), total=3, desc="Measuring similarity"):
        for seen_unseen in ["seen", "unseen"]:
            eval_silver_data, va_labels = load_eval_data(settings, temper_f, seen_unseen, data_type, flag_concat=True)
            _pred_label = {key: value[f"{seen_unseen}_{temper_s}"] for key, value in predicted_label.items()}
            for k, v in _pred_label.items():
                assert len(v) == len(eval_silver_data), f"invalid length: {len(v)} != {len(eval_silver_data)}"
            measure_similarity(settings, data_type, device, eval_silver_data, _pred_label, f"{data_type}/{seen_unseen}_{temper_s}")
            
    plot_selendipity(settings)



if __name__ == "__main__":

    data_type = "情報系"
    device = "cuda:0"
    no_list = ["1", "2", "3", "4", "5"]

    with open("./settings.json", "r") as f:
        settings = json.load(f)
    
    main(settings, data_type, device, no_list)
