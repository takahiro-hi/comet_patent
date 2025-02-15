import json, os, random, torch, argparse, sys

from sklearn.metrics import accuracy_score
from transformers import BertJapaneseTokenizer
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim

from model import Classifier
from utils import tokenize_data

sys.path.append("./scripts/utils")
from load_data import load_gold, load_silver



def construct_train_data(args):

    pos = load_gold(settings["data"], args.gold_type, args.patent_domain)
    neg = load_silver(settings["data"], args.gold_type, args.patent_domain)

    print(f"gold: {args.gold_type}, patent_domain: {args.patent_domain}")
    print(f"pos: {len(pos)}, neg: {len(neg)}")

    assert len(pos) > len(neg), "somethig wrong with the data"

    pos = random.sample(pos, len(neg))
    data = pos + neg
    label = [1] * len(pos) + [0] * len(neg)

    combined = list(zip(data, label))
    random.shuffle(combined)
    data[:], label[:] = zip(*combined)

    return data, label



def main(result_path, config, args):

    # load data
    _data, _label = load_train_data_for_step1(config, args.gold_type, args.data_type)
    split_idx = int(len(_data) * 0.8)
    train_data, train_label = _data[:split_idx], _label[:split_idx]
    val_data, val_label = _data[split_idx:], _label[split_idx:]
    
    # settings
    torch.cuda.set_device(2)
    device = "cuda"
    ids = [2, 3]
    model_name = config["base_train"]["model_name"]
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    selector = Classifier(model_name).to(device)
    selector = torch.nn.DataParallel(selector, device_ids=ids)
    optimizer = optim.AdamW([
        {"params": selector.module.model.parameters(), "lr": config["base_train"]["lr_bert"], "weight_decay": config["base_train"]["weight_decay"]},
        {"params": selector.module.dense.parameters(), "lr": config["base_train"]["lr_head"], "weight_decay": config["base_train"]["weight_decay"]},
        {"params": selector.module.out_proj.parameters(), "lr": config["base_train"]["lr_head"], "weight_decay": config["base_train"]["weight_decay"]}
    ])
    loss_func = nn.BCEWithLogitsLoss()

    patience_count = 0
    flag_stop = False
    # training and evaluation
    loss_train, loss_val = [], []
    for _ in tqdm(range(config["base_train"]["epochs"]), desc="training"):
        if flag_stop:
            break
        for idx, step in enumerate(tqdm(range(0, len(train_label), config["base_train"]["batch_size"]), leave=False, desc="steps")):
            # training
            selector.train()
            _tr_data = train_data[step:step+config["base_train"]["batch_size"]]
            _tr_label = train_label[step:step+config["base_train"]["batch_size"]]
            _tr_inps = tokenize_data(_tr_data, tokenizer, device)

            selector.zero_grad()
            tr_pred = selector(**_tr_inps).squeeze(-1)
            loss = loss_func(tr_pred, torch.tensor(_tr_label).to(device))
            loss.backward()
            optimizer.step()
            loss_train.append(loss.item())

            # validation
            selector.eval()
            _val_inps = tokenize_data(val_data, tokenizer, device)
            with torch.no_grad():
                _val_pred = selector(**_val_inps).squeeze(-1)
            loss_val.append(loss_func(_val_pred, torch.tensor(val_label).to(device)).item())
            
            # judge early stopping
            if config["base_train"]["early_stopping"]["flag"] and idx % config["base_train"]["early_stopping"]["steps"] == 0:
                if len(loss_val) > 1 and loss_val[-1] > loss_val[-2]:
                    patience_count += 1
                else:
                    patience_count = 0
                if patience_count >= config["base_train"]["early_stopping"]["patience"]:
                    flag_stop = True
                    print("early stopping")
                    break
                
            if idx % 100 == 0:
                _val_prob = torch.sigmoid(_val_pred).cpu().numpy()
                acc = accuracy_score(val_label, (_val_prob >= 0.5).astype(int))
                _loss = loss_func(_val_pred, torch.tensor(val_label).to(device)).item()
                save_data = [{"data": data, "label": label, "prob": prob.item()} for data, label, prob in zip(val_data, val_label, _val_prob)]
                save_data = sorted(save_data, key=lambda x: x["prob"], reverse=True)
                save_txt = f"\n===={step}_acc:{acc}_loss:{_loss}====\n"
                for d in save_data:
                    save_txt += f"{int(d['label'])}\t{round(d['prob'], 3)}\t{d['data']}\n"
                with open(os.path.join(result_path, "pred_val.txt"), "a") as f:
                    f.write(save_txt)

    # save selector
    torch.save(selector.module.state_dict(), os.path.join(result_path, "selector.pth"))
    plot_loss(loss_train, loss_val, result_path)


def plot_loss(loss_train, loss_val, result_path):

    plt.figure(figsize=(10, 6))
    plt.plot(loss_train, label='train')
    plt.plot(loss_val, label='val')
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(result_path, "loss.png"))
    plt.close()



if __name__=="__main__":

    """
    nohup python scripts/train/base.py --gold_type "patent" --data_type "情報系" &
    nohup python scripts/train/base.py --gold_type "atomic" --data_type "none" &
    """

    random.seed(42)
    
    with open("./settings_filter.json", "r") as f:
        settings_filter = json.load(f)

    with open("./settings_data.json", "r") as f:
        settings_data = json.load(f)

    settings = {"filter": settings_filter, "data": settings_data}

    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_type", type=str)
    parser.add_argument("--patetn_domain", type=str, default=None)
    args = parser.parse_args()

    if config["base_train"]["early_stopping"]["flag"]:
        save_dir = f"{args.gold_type}/{args.data_type}/es"
    else:
        save_dir = f"{args.gold_type}/{args.data_type}/no_es_epoch_{config['base_train']['epochs']}"

    result_path = os.path.join(config["directories"]["step_1"], save_dir)
    os.makedirs(result_path)

    with open(os.path.join(result_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    main(result_path, config, args)