import json, os, random, sys

sys.path.append("./scripts/utils")
from load_data import load_gold


"""
シルバーデータ生成のためのプロンプトを返す関数
"""



def construct_prompt_for_head(head_list, num_few_shot):

    sampled_heads = random.sample(head_list, num_few_shot)

    prompt = ""
    for idx, head in enumerate(sampled_heads):
        prompt += f"{idx+1}: {head}\n\n"
    prompt += f"{idx+2}: "

    return prompt


def construct_prompt_for_tail(data_list, num_few_shot, target_head):

    prompt = "事態１により発生する効果である事態２を生成してください。\n\n\n"
    sampled_data = random.sample(data_list, num_few_shot)

    for idx, (head, tail) in enumerate(sampled_data):
        prompt += f"事態１：{head}\n"
        prompt += f"事態２：{tail}\n\n"

    prompt += f"事態１：" + target_head + "\n" + "事態２："

    return prompt



if __name__=="__main__":

    random.seed(1)

    with open("./settings_data.json", "r") as f:
        settings = json.load(f)
    
    gold = load_gold(settings, "patent", "情報系")

    prompt = construct_prompt_for_tail(gold, 15, target_head="ipadを使って検索をする。")
    print(f"##\n{prompt}##")
    print("\n^^^^^^^^^^^^\n")
    head_list = [d[0] for d in gold]
    prompt = construct_prompt_for_head(head_list, 10)
    print(f"{prompt}##")
