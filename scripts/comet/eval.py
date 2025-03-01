import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bert_score import score

sys.path.append("./scripts/utils")
from load_data import load_gold


ADV_TOKEN = "xEffect"



def eval(silver_type, settings_data):

    model_path = f"./result_情報系/comet/{silver_type}/model"

    model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("nlp-waseda/comet-v2-gpt2-small-japanese", padding_side="left")

    gold_data = load_gold(settings_data, "patent", "情報系", False)

    test_dataset = {
        "inputs": [d[0] + ADV_TOKEN for d in gold_data],
        "outputs": [d[1] + tokenizer.eos_token for d in gold_data]
    }

    assert len(tokenizer.encode(ADV_TOKEN, add_special_tokens=False)) == 1, "ADV_TOKEN should be one token"
    special_ids = set(tokenizer.all_special_ids)
    adv_id = tokenizer.convert_tokens_to_ids(ADV_TOKEN)

    input_ids = tokenizer(test_dataset["inputs"], truncation=True, max_length=256, return_tensors="pt", padding=True)
    with torch.no_grad():
        output_ids = model.generate(**input_ids.to("cuda"))
    filtered_output_ids = [
        [token_id for token_id in seq if not (token_id in special_ids and token_id != adv_id)]
        for seq in output_ids.tolist()
    ]
    outputs = [tokenizer.decode(seq, skip_special_tokens=False) for seq in filtered_output_ids]

    generated_data = []
    for o in outputs:
        head, tail = o.split(ADV_TOKEN)
        generated_data.append(tail)
    
    # gpuメモリを解放
    del model
    del input_ids
    torch.cuda.empty_cache()
    
    return generated_data, test_dataset["inputs"], test_dataset["outputs"]



def calc_metrics(gold, pred):

    # BERTスコア
    P, R, F1 = score(pred, gold, lang="ja", verbose=False)

    return {
        "Precision": P.mean().item(),
        "Recall": R.mean().item(),
        "F1": F1.mean().item()
    }


if __name__ == "__main__":
    
    with open("./settings_data.json", "r") as f:
        settings_data = json.load(f)

    gened_all, gold_all, ans = eval("all", settings_data)
    gened_base, test_base, _ = eval("base_model", settings_data)
    gened_adv, test_adv, _ = eval("adv_model", settings_data)

    for t_all, t_base, t_adv in zip(gold_all, test_base, test_adv):
        assert t_all == t_base == t_adv
    
    save_data = []
    for g_all, g_base, g_adv, head, a in zip(gened_all, gened_base, gened_adv, gold_all, ans):
        save_data.append({
            "input_head": head,
            "ground_truth": a,
            "predication": {
                "all": g_all,
                "base": g_base,
                "adv": g_adv
            } 
        })

    with open("./result_情報系/comet/eval_data.json", "w") as f:
        f.write(json.dumps(save_data, indent=4, ensure_ascii=False))
    
    metrics_all = calc_metrics(ans, gened_all)
    metrics_base = calc_metrics(ans, gened_base)
    metrics_adv = calc_metrics(ans, gened_adv)
    save_metrics = {
        "Precision": {
            "all": metrics_all["Precision"],
            "base": metrics_base["Precision"],
            "adv": metrics_adv["Precision"]
        },
        "Recall": {
            "all": metrics_all["Recall"],
            "base": metrics_base["Recall"],
            "adv": metrics_adv["Recall"]
        },
        "F1": {
            "all": metrics_all["F1"],
            "base": metrics_base["F1"],
            "adv": metrics_adv["F1"]
        }
    }
    with open("./result_情報系/comet/eval_metrics.json", "w") as f:
        f.write(json.dumps(save_metrics, indent=4, ensure_ascii=False))