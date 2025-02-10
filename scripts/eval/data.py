import sys
import json
from util_load_data import load_eval_data, load_atomic

sys.path.append("./scripts/train")
from utils_load_data import load_silver, load_gold


def main(config, patent_type="情報系"):

    md_txt = "# Dataset Summary"

    data_gold = load_gold(config, "patent", patent_type)
    data_g, head_g, tail_g = [f"{d[0]}; {d[1]}" for d in data_gold], [d[0] for d in data_gold], [d[1] for d in data_gold]

    atomic_gold, atomic_silver = load_atomic(config)
    atomic_g, atomic_g_head, atomic_tail = [f"{d[0]}; {d[1]}" for d in atomic_gold], [d[0] for d in atomic_gold], [d[1] for d in atomic_gold]
    atomic_s, atomic_s_head, atomic_s_tail = [f"{d[0]}; {d[1]}" for d in atomic_silver], [d[0] for d in atomic_silver], [d[1] for d in atomic_silver]

    data_silver = load_silver(config, "patent", patent_type)
    data_s, head_s, tail_s = [f"{d[0]}; {d[1]}" for d in data_silver], [d[0] for d in data_silver], [d[1] for d in data_silver]

    seen_08_data, seen_08_label = load_eval_data(config, "0.8", "seen", patent_type, False)
    seen_10_data, seen_10_label = load_eval_data(config, "1.0", "seen", patent_type, False)
    seen_13_data, seen_13_label = load_eval_data(config, "1.3", "seen", patent_type, False)
    unseen_08_data, unseen_08_label = load_eval_data(config, "0.8", "unseen", patent_type, False)
    unseen_10_data, unseen_10_label = load_eval_data(config, "1.0", "unseen", patent_type, False)
    unseen_13_data, unseen_13_label = load_eval_data(config, "1.3", "unseen", patent_type, False)
    seen_08_data, seen_08_head, seen_08_tail = [f"{d[0]}; {d[1]}" for d in seen_08_data], [d[0] for d in seen_08_data], [d[1] for d in seen_08_data]
    seen_10_data, seen_10_head, seen_10_tail = [f"{d[0]}; {d[1]}" for d in seen_10_data], [d[0] for d in seen_10_data], [d[1] for d in seen_10_data]
    seen_13_data, seen_13_head, seen_13_tail = [f"{d[0]}; {d[1]}" for d in seen_13_data], [d[0] for d in seen_13_data], [d[1] for d in seen_13_data]
    unseen_08_data, unseen_08_head, unseen_08_tail = [f"{d[0]}; {d[1]}" for d in unseen_08_data], [d[0] for d in unseen_08_data], [d[1] for d in unseen_08_data]
    unseen_10_data, unseen_10_head, unseen_10_tail = [f"{d[0]}; {d[1]}" for d in unseen_10_data], [d[0] for d in unseen_10_data], [d[1] for d in unseen_10_data]
    unseen_13_data, unseen_13_head, unseen_13_tail = [f"{d[0]}; {d[1]}" for d in unseen_13_data], [d[0] for d in unseen_13_data], [d[1] for d in unseen_13_data]

    md_txt += f"""

## Data Size

| Dataset               | Total Size | Unique Data | Unique Head | Unique Tail |
|:---------------------:|:----------:|:-----------:|:-----------:|:-----------:|
| GOLD DATA\t\t| {len(data_g)}\t| {len(set(data_g))}\t| {len(set(head_g))}\t| {len(set(tail_g))}\t|
| SILVER DATA\t\t| {len(data_s)}\t| {len(set(data_s))}\t| {len(set(head_s))}\t| {len(set(tail_s))}\t|
| ANN Seen 08\t\t| {len(seen_08_data)}\t| {len(set(seen_08_data))}\t| {len(set(seen_08_head))}\t| {len(set(seen_08_tail))}\t|
| ANN Seen 10\t\t| {len(seen_10_data)}\t| {len(set(seen_10_data))}\t| {len(set(seen_10_head))}\t| {len(set(seen_10_tail))}\t|
| ANN Seen 13\t\t| {len(seen_13_data)}\t| {len(set(seen_13_data))}\t| {len(set(seen_13_head))}\t| {len(set(seen_13_tail))}\t|
| ANN Unseen 08\t\t| {len(unseen_08_data)}\t| {len(set(unseen_08_data))}\t| {len(set(unseen_08_head))}\t| {len(set(unseen_08_tail))}\t|
| ANN Unseen 10\t\t| {len(unseen_10_data)}\t| {len(set(unseen_10_data))}\t| {len(set(unseen_10_head))}\t| {len(set(unseen_10_tail))}\t|
| ANN Unseen 13\t\t| {len(unseen_13_data)}\t| {len(set(unseen_13_data))}\t| {len(set(unseen_13_head))}\t| {len(set(unseen_13_tail))}\t|
| ATOMIC GOLD DATA\t\t| {len(atomic_g)}\t| {len(set(atomic_g))}\t| {len(set(atomic_g_head))}\t| {len(set(atomic_tail))}\t|
| ATOMIC SILVER DATA\t\t| {len(atomic_s)}\t| {len(set(atomic_s))}\t| {len(set(atomic_s_head))}\t| {len(set(atomic_s_tail))}\t|"""


    md_txt += f"""

## Duplicates

| Dataset Pair                | Duplicates of Data | Duplicates of Head | Duplicates of Tail |
|:---------------------------:|:------------------:|:------------------:|:------------------:|
| GOLD DATA & SILVER DATA\t\t| {len(set(data_g) & set(data_s))}\t| {len(set(head_g) & set(head_s))}\t| {len(set(tail_g) & set(tail_s))}\t|
| GOLD DATA & ANN Seen 08\t\t| {len(set(data_g) & set(seen_08_data))}\t| {len(set(head_g) & set(seen_08_head))}\t| {len(set(tail_g) & set(seen_08_tail))}\t|
| GOLD DATA & ANN Seen 10\t\t| {len(set(data_g) & set(seen_10_data))}\t| {len(set(head_g) & set(seen_10_head))}\t| {len(set(tail_g) & set(seen_10_tail))}\t|
| GOLD DATA & ANN Seen 13\t\t| {len(set(data_g) & set(seen_13_data))}\t| {len(set(head_g) & set(seen_13_head))}\t| {len(set(tail_g) & set(seen_13_tail))}\t|
| GOLD DATA & ANN Unseen 08\t\t| {len(set(data_g) & set(unseen_08_data))}\t| {len(set(head_g) & set(unseen_08_head))}\t| {len(set(tail_g) & set(unseen_08_tail))}\t|
| GOLD DATA & ANN Unseen 10\t\t| {len(set(data_g) & set(unseen_10_data))}\t| {len(set(head_g) & set(unseen_10_head))}\t| {len(set(tail_g) & set(unseen_10_tail))}\t|
| GOLD DATA & ANN Unseen 13\t\t| {len(set(data_g) & set(unseen_13_data))}\t| {len(set(head_g) & set(unseen_13_head))}\t| {len(set(tail_g) & set(unseen_13_tail))}\t|
| SILVER DATA & ANN Seen 08\t\t| {len(set(data_s) & set(seen_08_data))}\t| {len(set(head_s) & set(seen_08_head))}\t| {len(set(tail_s) & set(seen_08_tail))}\t|
| SILVER DATA & ANN Seen 10\t\t| {len(set(data_s) & set(seen_10_data))}\t| {len(set(head_s) & set(seen_10_head))}\t| {len(set(tail_s) & set(seen_10_tail))}\t|
| SILVER DATA & ANN Seen 13\t\t| {len(set(data_s) & set(seen_13_data))}\t| {len(set(head_s) & set(seen_13_head))}\t| {len(set(tail_s) & set(seen_13_tail))}\t|
| SILVER DATA & ANN Unseen 08\t\t| {len(set(data_s) & set(unseen_08_data))}\t| {len(set(head_s) & set(unseen_08_head))}\t| {len(set(tail_s) & set(unseen_08_tail))}\t|
| SILVER DATA & ANN Unseen 10\t\t| {len(set(data_s) & set(unseen_10_data))}\t| {len(set(head_s) & set(unseen_10_head))}\t| {len(set(tail_s) & set(unseen_10_tail))}\t|
| SILVER DATA & ANN Unseen 13\t\t| {len(set(data_s) & set(unseen_13_data))}\t| {len(set(head_s) & set(unseen_13_head))}\t| {len(set(tail_s) & set(unseen_13_tail))}\t|"""

    """for idx, data in enumerate(list(set(seen_13_data) - set(data_s))):
        print(idx, data)"""

    md_txt += f"""

## label distribution

| Dataset               | Accept | Reject | Accept Rate | Reject Rate |
|:---------------------:|:------:|:------:|:-----------:|:-----------:|
| ANN Seen 08\t\t| {seen_08_label.count(1)}\t| {seen_08_label.count(0)}\t| {round(seen_08_label.count(1)/len(seen_08_label), 3)}\t| {round(seen_08_label.count(0)/len(seen_08_label), 3)}\t|
| ANN Seen 10\t\t| {seen_10_label.count(1)}\t| {seen_10_label.count(0)}\t| {round(seen_10_label.count(1)/len(seen_10_label), 3)}\t| {round(seen_10_label.count(0)/len(seen_10_label), 3)}\t|
| ANN Seen 13\t\t| {seen_13_label.count(1)}\t| {seen_13_label.count(0)}\t| {round(seen_13_label.count(1)/len(seen_13_label), 3)}\t| {round(seen_13_label.count(0)/len(seen_13_label), 3)}\t|
| ANN Unseen 08\t\t| {unseen_08_label.count(1)}\t| {unseen_08_label.count(0)}\t| {round(unseen_08_label.count(1)/len(unseen_08_label), 3)}\t| {round(unseen_08_label.count(0)/len(unseen_08_label), 3)}\t|
| ANN Unseen 10\t\t| {unseen_10_label.count(1)}\t| {unseen_10_label.count(0)}\t| {round(unseen_10_label.count(1)/len(unseen_10_label), 3)}\t| {round(unseen_10_label.count(0)/len(unseen_10_label), 3)}\t|
| ANN Unseen 13\t\t| {unseen_13_label.count(1)}\t| {unseen_13_label.count(0)}\t| {round(unseen_13_label.count(1)/len(unseen_13_label), 3)}\t| {round(unseen_13_label.count(0)/len(unseen_13_label), 3)}\t|
"""

    print(md_txt)


if __name__=="__main__":

    with open("./settings.json", "r") as f:
        config = json.load(f)

    main(config)