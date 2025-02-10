import json, os, random
import pandas as pd


def construct_prompt_for_head(data_list, num_few_shot):

    prompt = ""

    head_list = [d[0] for d in data_list]
    sampled_heads = random.sample(head_list, num_few_shot)
        
    for idx, head in enumerate(sampled_heads):
        prompt += f"{idx+1}: {head}\n\n"
    prompt += f"{idx+2}: "
    
    return prompt


def construct_prompt_for_tail(data_list, num_few_shot, target_head):

    prompt = "事態１の後に発生する事態２を生成してください。\n\n\n"
    sampled_data = random.sample(data_list, num_few_shot)

    for idx, (head, tail) in enumerate(sampled_data):
        prompt += f"事態１：{head}\n"
        prompt += f"事態２：{tail}\n\n"

    prompt += f"事態１：" + target_head + "\n" + "事態２："
  
    return prompt



if __name__=="__main__":

    random.seed(1)

    with open("./data/atomic/gold/data_after.json", "r") as f:
        data = json.load(f)
    
    prompt = construct_prompt_for_tail(data, 15, target_head="ipadを使って検索をする。")
    print(f"##\n{prompt}##")
    print("\n^^^^^^^^^^^^\n")
    prompt = construct_prompt_for_head(data, 10)
    print(f"##{prompt}##")