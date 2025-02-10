import os, json, random, sys, csv
import pandas as pd
from openpyxl import load_workbook
from openpyxl.worksheet.datavalidation import DataValidation

from tqdm import tqdm

from prompt_template import construct_prompt_for_tail
from model import Model

sys.path.append("./scripts/train")
from utils_load_data import to_input_format, load_silver


    

def generate_tail(model, data_list, config, head_list, data_type, temperature, file_name):

    generated_tail = []

    for head in tqdm(head_list, desc="generating tail..."):
        prompt = construct_prompt_for_tail(data_list, config["generate_silver"]["tail"]["few_shot_exam"], target_head=head)
        generated_data = model(prompt, "tail", temperature=temperature)
        selected_data = random.choice(generated_data)
        generated_tail.append(to_input_format(head, selected_data))
    
    with open(os.path.join(config["directories"]["patent"]["eval"], file_name), "w") as f:
        f.write(json.dumps(generated_tail, indent=4, ensure_ascii=False))



def main(config, data_type="情報系"):

    data_num = 300

    model = Model(config)

    with open(os.path.join(config["directories"]["patent"]["gold_data"], f"{data_type}_adv.json"), "r") as f:
        data = json.load(f)
    
    with open(os.path.join(config["directories"]["patent"]["silver_data"], f"{data_type}_head.json"), "r") as f:
        all_head = json.load(f)

    silver = load_silver(config, "patent", data_type)
    silver_head = [d[0] for d in silver]
    
    # select and generate seen data
    while True:
        seen_13 = random.sample(silver, data_num)
        seen_head = [d[0] for d in seen_13]
        if len(set(seen_head)) == len(seen_head):
            break
    with open(os.path.join(config["directories"]["patent"]["eval"], f"{data_type}_seen_13.json"), "w") as f:
        f.write(json.dumps(seen_13, indent=4, ensure_ascii=False))

    for t_str, temperature in zip(["10", "08"], [1.0, 0.8]):
        generate_tail(model, data, config, seen_head, data_type, temperature, f"{data_type}_seen_{t_str}.json")
    
    # generate unseen data
    unsed_head = list(set(all_head) - set(silver_head))
    while True:
        unseen_head = random.sample(unsed_head, data_num)
        if len(set(unseen_head)) == len(unseen_head):
            break
    print(f"seen_head: {len(seen_head)}")
    print(f"unseen_head: {len(unseen_head)}")
    print(f"silver_head: {len(silver_head)}")
    print(f"all_head: {len(all_head)}")
    print(f"seen & unseen: {len(set(seen_head) & set(unseen_head))}")
    print(f"seen & silver: {len(set(seen_head) & set(silver_head))}")
    print(f"seen & all: {len(set(seen_head) & set(all_head))}")
    print(f"unseen & silver: {len(set(unseen_head) & set(silver_head))}")
    print(f"unseen & all: {len(set(unseen_head) & set(all_head))}")
    print(f"silver & all: {len(set(silver_head) & set(all_head))}")
    for t_str, temperature in zip(["13", "10", "08"], [1.3, 1.0, 0.8]):
        generate_tail(model, data, config, unseen_head, data_type, temperature, f"{data_type}_unseen_{t_str}.json")

    
def make_sheet(config, tr_data_type="情報系"):
    
    for ev_data_type in ["seen", "unseen"]:
        for tempe_str, tempe_float in zip(["13", "10", "08"], [1.3, 1.0, 0.8]):

            with open(os.path.join(config["directories"]["patent"]["eval"], f"{tr_data_type}_{ev_data_type}_{tempe_str}.json"), "r") as f:
                _data = json.load(f)
            
            df = pd.DataFrame(_data, columns=["head", "tail"])
            df["Label"] = ""
            
            filename = os.path.join(config["directories"]["patent"]["eval"], f"{ev_data_type}_{tempe_str}.xlsx")
            df.to_excel(filename, index=False)
            
            wb = load_workbook(filename)
            ws = wb.active
            
            dv = DataValidation(type="list", formula1='"positive,negative,N / A"', allow_blank=True)
            dv.error = '入力値は "positive", "negative", "N / A" のいずれかを選択してください'
            dv.errorTitle = '入力エラー'
            
            ws.add_data_validation(dv)
            dv.add(f'C2:C{len(_data)+1}') 
            wb.save(filename)




if __name__ == "__main__":

    with open("./settings.json", "r") as f:
        config = json.load(f)

    random.seed(1)

    os.makedirs(config["directories"]["patent"]["eval"], exist_ok=True)

    #main(config)
    make_sheet(config)