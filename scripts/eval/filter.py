import json, os

from tqdm import tqdm

from util_load_data import load_eval_data




data_type = "情報系"
with open("./settings.json", "r") as f:
    settings = json.load(f)

with open(os.path.join(settings["directories"]["eval"]["selendipity"], "predicted_label.json"), "r") as f:
    predicted_label = json.load(f)

md_output = []
for temper_f, temper_s in zip([0.8, 1.0, 1.3], ["08", "10", "13"]):
    for seen_unseen in ["seen", "unseen"]:
        eval_silver_data, va_labels = load_eval_data(settings, temper_f, seen_unseen, data_type, flag_concat=True)
        _pred_label = {key: value[f"{seen_unseen}_{temper_s}"] for key, value in predicted_label.items()}
        for k, v in _pred_label.items():
            assert len(v) == len(eval_silver_data), f"invalid length: {len(v)} != {len(eval_silver_data)}"

        """print(f"temperature: {temper_s}, seen_unseen: {seen_unseen}")
        print(f"ratio of positive labels (ground truth): {sum(va_labels) / len(va_labels)} ({sum(va_labels)} / {len(va_labels)})")
        for k, v in _pred_label.items():
            print(f"ratio of positive labels ({k}): {sum(v) / len(v)} ({sum(v)} / {len(v)})")"""
        
        # Markdown Table Header
        md_output.append(f"## Temperature: {temper_s}, Seen/Unseen: {seen_unseen}")
        md_output.append("")
        md_output.append("| Model | Positive Ratio | Positive Count | Total Count |")
        md_output.append("|-------|----------------|----------------|-------------|")

        # Ground Truth
        ground_truth_ratio = sum(va_labels) / len(va_labels)
        ground_truth_positive = sum(va_labels)
        ground_truth_total = len(va_labels)
        md_output.append(f"| Ground Truth | {ground_truth_ratio:.4f} | {ground_truth_positive} | {ground_truth_total} |")

        # Predicted Labels
        for k, v in _pred_label.items():
            pred_ratio = sum(v) / len(v)
            pred_positive = sum(v)
            pred_total = len(v)
            md_output.append(f"| {k} | {pred_ratio:.4f} | {pred_positive} | {pred_total} |")

        md_output.append("")  # Add space between tables

    # Print Markdown Output
    print("\n".join(md_output))