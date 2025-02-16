import json, os, random, torch, argparse, sys

from sklearn.metrics import accuracy_score
from transformers import BertJapaneseTokenizer
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from model import Classifier
from utils import tokenize_data, plot_x_iters, get_settings

sys.path.append("./scripts/utils")
from load_data import load_gold, load_silver



def get_train_data(args, settings):

    pos = load_gold(settings["data"], args.gold_type, args.patent_domain)
    neg = load_silver(settings["data"], args.temperature_tail, args.gold_type, args.patent_domain)

    print(f"gold type: {args.gold_type}, patent_domain: {args.patent_domain}")
    print(f"pos: {len(pos)}, neg: {len(neg)}")

    assert len(pos) < len(neg), "somethig wrong with the data"

    neg = random.sample(neg, len(pos))
    data = pos + neg
    label = [1.] * len(pos) + [0.] * len(neg)

    combined = list(zip(data, label))
    random.shuffle(combined)
    data[:], label[:] = zip(*combined)

    return data, label



def main(result_path, settings, args):

    tr_parms = settings["tr_params"]

    # load data
    _data, _label = get_train_data(args, settings)
    split_idx = int(len(_data) * 0.8)
    train_data, train_label = _data[:split_idx], _label[:split_idx]
    val_data, val_label = _data[split_idx:], _label[split_idx:]
    
    # settings
    tokenizer = BertJapaneseTokenizer.from_pretrained(tr_parms["model_name"])
    filter = Classifier(tr_parms["model_name"]).to("cuda")
    filter = torch.nn.DataParallel(filter)
    optimizer = optim.AdamW(filter.parameters(), lr=tr_parms["learning_rate"], weight_decay=tr_parms["weight_decay"])
    loss_func = nn.BCEWithLogitsLoss()

    patience_count = 0
    best_loss = float("inf")
    best_model_state = None

    # training and evaluation
    loss_train, loss_val = {"iteration":[], "loss":[]}, {"iteration":[], "loss":[]}
    for epoch in tqdm(range(tr_parms["epochs"]), desc="training"):
        for idx, step in enumerate(tqdm(range(0, len(train_label), tr_parms["batch_size"]), leave=False, desc="steps")):
            total_step = idx + epoch * len(train_label) // tr_parms["batch_size"]

            # training
            filter.train()
            _tr_data = train_data[step:step+tr_parms["batch_size"]]
            _tr_label = train_label[step:step+tr_parms["batch_size"]]
            _tr_inps = tokenize_data(_tr_data, tokenizer)

            filter.zero_grad()
            tr_pred = filter(**_tr_inps).squeeze(-1)
            loss = loss_func(tr_pred, torch.tensor(_tr_label).to("cuda"))
            loss.backward()
            optimizer.step()
            loss_train["iteration"].append(total_step)
            loss_train["loss"].append(loss.item())

        # evaluation
        if epoch % tr_parms["eval_steps"] == 0 or epoch == tr_parms["epochs"] - 1:                
            filter.eval()
            _val_inps = tokenize_data(val_data, tokenizer)
            with torch.no_grad():
                _val_pred = filter(**_val_inps).squeeze(-1)
            _loss = loss_func(_val_pred, torch.tensor(val_label).to("cuda")).item()
            loss_val["iteration"].append(total_step)
            loss_val["loss"].append(_loss)

            _val_prob = torch.sigmoid(_val_pred).cpu().numpy()
            acc = accuracy_score(val_label, (_val_prob >= 0.5).astype(int))
            save_data = [{"data": data, "label": label, "prob": prob.item()} for data, label, prob in zip(val_data, val_label, _val_prob)]
            save_data = sorted(save_data, key=lambda x: x["prob"], reverse=True)
            save_txt = f"\n===={idx + epoch * len(train_label)}_acc:{acc}_loss:{_loss}====\n"
            for d in save_data:
                save_txt += f"{int(d['label'])}\t{round(d['prob'], 3)}\t{d['data']}\n"
            with open(os.path.join(result_path, "pred_val.txt"), "a") as f:
                f.write(save_txt)
        
            # judge early stopping
            if tr_parms["early_stopping"]["flag"]:
                if _loss < best_loss:
                    best_loss = _loss
                    best_model_state = filter.module.state_dict()
                    patience_count = 0
                else:
                    patience_count += 1
                
                if patience_count >= tr_parms["early_stopping"]["patience"]:
                    print("early stopping")
                    break

    # save selector
    if best_model_state is None:
        torch.save(filter.module.state_dict(), os.path.join(result_path, "filter.pth"))
    else:
        torch.save(best_model_state, os.path.join(result_path, "filter.pth"))

    plot_x_iters(
        values = {"train": loss_train, "validation": loss_val},
        save_file_path = os.path.join(result_path, "loss"),
        save_value_flag = True,
        title = "loss",
    )



if __name__=="__main__":

    """
    nohup python scripts/filter/base.py --device_ids "3" --gold_type patent --patent_domain "情報系" --temperature_tail 1.3 > nohup1.out &
    """

    random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_ids", type=str)
    parser.add_argument("--gold_type", type=str)
    parser.add_argument("--patent_domain", type=str, default=None)
    parser.add_argument("--temperature_tail", type=float, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids

    settings = get_settings("base")

    _temp_dir = f"es" if settings["tr_params"]["early_stopping"]["flag"] else f"epoch_{settings['tr_params']['epochs']}"
    if args.gold_type == "patent":
        result_path = os.path.join(settings["result_path"][args.patent_domain]["base"], f"{args.temperature_tail}_{_temp_dir}")
    elif args.gold_type == "atomic":
        result_path = os.path.join(settings["result_path"][args.gold_type], _temp_dir)

    os.makedirs(result_path)

    with open(os.path.join(result_path, "settings.json"), "w") as f:
        f.write(json.dumps(settings, indent=4, ensure_ascii=False))

    main(result_path, settings, args)