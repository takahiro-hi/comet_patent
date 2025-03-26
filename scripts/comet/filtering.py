import json, sys, argparse, os, torch

from transformers import BertJapaneseTokenizer
from tqdm import tqdm

sys.path.append("./scripts/utils")
from load_data import load_silver

sys.path.append("./scripts/filter")
from model import Classifier
from utils import tokenize_data



def load_model(model_type, settings_model, args):

    assert settings_model["params_base"]["model_name"] == settings_model["params_adv"]["model_name"], "model_name is not matched"

    if model_type == "base":
        model_path = os.path.join(settings_model["result_path"][args.patent_domain], f"filter_base_t_{args.temperature_tail}/augmentation_es/filter.pth")
    
    elif model_type == "adv":
        model_path = os.path.join(settings_model["result_path"][args.patent_domain], f"filter_adv_t_{args.temperature_tail}/augmentation/{args.adv_model}/selector.pth")
    
    state_dict = torch.load(model_path, map_location="cuda", weights_only=True)
    filter = Classifier(settings_model["params_adv"]["model_name"]).to("cuda")
    filter.load_state_dict(state_dict)

    tokenizer = BertJapaneseTokenizer.from_pretrained(settings_model["params_adv"]["model_name"])
    
    return filter, tokenizer



def _pred(model, tokenizer, data, batch_size=508):

    pred_label = []
    for i in tqdm(range(0, len(data), batch_size)):
        inps = tokenize_data(data[i:i+batch_size], tokenizer)
        with torch.no_grad():
            pred = model(**inps).squeeze(-1)
        prob = torch.sigmoid(pred).cpu().numpy()
        label = (prob >= 0.5).astype(int)
        pred_label.extend(label)

    assert len(data) == len(pred_label), "something wrong with the data"
    selected_data = [d for d, l in zip(data, pred_label) if l == 1]
    return selected_data



def main(settings_data, settings_model, args):

    silver_data = load_silver(settings_data, "patent", settings_model["params_comet"]["tr_silver_temperature"], args.patent_domain, False)
    print(f"silver data: {len(silver_data)}")
    with open(os.path.join(settings_data["dir_path"]["patent"]["train"], f"{args.patent_domain}_triple_{settings_model["params_comet"]["tr_silver_temperature"]}.json"), "w") as f:
        json.dump(silver_data, f, indent=4, ensure_ascii=False)

    base_model, tokenizer = load_model("base", settings_model, args)
    selected_by_base = _pred(base_model, tokenizer, silver_data)
    print(f"selected by base: {len(selected_by_base)}")
    with open(os.path.join(settings_data["dir_path"]["patent"]["train"], f"{args.patent_domain}_triple_{settings_model["params_comet"]["tr_silver_temperature"]}_base.json"), "w") as f:
        json.dump(selected_by_base, f, indent=4, ensure_ascii=False)

    adv_model, tokenizer = load_model("adv", settings_model, args)
    selected_by_adv = _pred(adv_model, tokenizer, silver_data)
    print(f"selected by adv: {len(selected_by_adv)}")
    with open(os.path.join(settings_data["dir_path"]["patent"]["train"], f"{args.patent_domain}_triple_{settings_model["params_comet"]["tr_silver_temperature"]}_adv_{args.adv_model}.json"), "w") as f:
        json.dump(selected_by_adv, f, indent=4, ensure_ascii=False)




if __name__ == "__main__":
    """
    python scripts/comet/filtering.py --patent_domain "情報系" --temperature_tail 1.3 --adv_model init_0_no4
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--patent_domain", type=str, default="情報系")
    parser.add_argument("--temperature_tail", type=str, default="1.3")
    parser.add_argument("--adv_model", type=str)
    args = parser.parse_args()

    with open("./settings_data.json", "r") as f:
        settings_data = json.load(f)
    
    with open("./settings_model.json", "r") as f:
        settings_model = json.load(f)
    
    main(settings_data, settings_model, args)