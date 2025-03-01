import os, json, sys, argparse, torch
from tqdm import tqdm

from prompt_template import construct_prompt_for_head, construct_prompt_for_tail
from model import Model

sys.path.append("./scripts/utils")
from load_data import load_gold


"""
シルバーデータを生成する
"""



def generate_head(model, gold_data, settings, patent_domain):

    generated_head = []
    gold_head = [d[0] for d in gold_data]

    for _ in tqdm(range(settings["parameters_silver"]["head"]["try_num"]), desc="generating head..."):
        prompt = construct_prompt_for_head(gold_head, settings["parameters_silver"]["head"]["few_shot_exam"])
        generated_data = model(prompt, "head")
        generated_head.extend(generated_data)

    generated_head = list(set(generated_head))
    saved_data = list(set(generated_head) - set(gold_head))

    with open(os.path.join(settings["dir_path"]["patent"]["silver"], f"{patent_domain}_head.json"), "w") as f:
        f.write(json.dumps(saved_data, indent=4, ensure_ascii=False))

    return saved_data


def generate_tail(model, gold_data, save_name, settings, args, head_list):

    generated_tail = []

    for head in tqdm(head_list):
        prompt = construct_prompt_for_tail(gold_data, settings["parameters_silver"]["tail"]["few_shot_exam"], target_head=head)
        generated_data = model(prompt, "tail", temperature=args.temperature_tail)
        generated_data = list(set(generated_data))
        generated_tail.append({"head": head, "tail": generated_data})

    with open(os.path.join(settings["dir_path"]["patent"]["silver"], save_name), "w") as f:
        f.write(json.dumps(generated_tail, indent=4, ensure_ascii=False))


def load_head(settings, patent_domain):

    with open(os.path.join(settings["dir_path"]["patent"]["silver"], f"{patent_domain}_head.json"), "r") as f:
        head = json.load(f)

    return head


def main(settings, args):

    model = Model(settings)

    gold = load_gold(settings, args.gold_type, args.patent_domain, shuffle=False)

    #generated_head = generate_head(model, gold, settings, args.patent_domain)
    generated_head = load_head(settings, args.patent_domain)
    split_num = 1000
    print(args.temperature_tail)
    for idx, i in tqdm(enumerate(range(0, len(generated_head), split_num)), desc="generating tail...", total=len(generated_head)//split_num):
        generate_tail(model, gold, f"{args.patent_domain}_triple_{args.temperature_tail}_no{idx}.json", settings, args, generated_head[i:i+split_num])



if __name__ == "__main__":
    """
    nohup python scripts/data/generate_silver.py --device_id "2, 3" --temperature_tail 1.3 --gold_type patent --patent_domain 情報系 &
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_ids", type=str)
    parser.add_argument("--temperature_tail", type=float)
    parser.add_argument("--gold_type", type=str)
    parser.add_argument("--patent_domain", type=str, default="情報系")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids

    with open("./settings_data.json", "r") as f:
        settings = json.load(f)

    os.makedirs(settings["dir_path"][args.gold_type]["silver"], exist_ok=True)

    main(settings, args)