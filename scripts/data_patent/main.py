import os, json, random
from tqdm import tqdm

from prompt_template import construct_prompt_for_head, construct_prompt_for_tail
from model import Model



def generate_head(model, data_list, config, data_type):

    generated_head = []

    for _ in tqdm(range(config["generate_silver"]["head"]["try_num"]), desc="generating head..."):
        prompt = construct_prompt_for_head(data_list, config["generate_silver"]["head"]["few_shot_exam"])
        generated_data = model(prompt, "head")
        generated_head.extend(generated_data)
    
    generated_head = list(set(generated_head))
    
    original_head = [d[0] for d in data_list]
    saved_data = list(set(generated_head) - set(original_head))

    with open(os.path.join(config["directories"]["patent"]["silver_data"], f"{data_type}_head.json"), "w") as f:
        f.write(json.dumps(saved_data, indent=4, ensure_ascii=False))
    
    return saved_data
    

def generate_tail(model, data_list, config, head_list, data_type):

    generated_tail = []

    for head in tqdm(head_list, desc="generating tail..."):
        prompt = construct_prompt_for_tail(data_list, config["generate_silver"]["tail"]["few_shot_exam"], target_head=head)
        generated_data = model(prompt, "tail")
        generated_data = list(set(generated_data))
        generated_tail.append({"head": head, "tail": generated_data})
    
    with open(os.path.join(config["directories"]["patent"]["silver_data"], f"{data_type}_tail.json"), "w") as f:
        f.write(json.dumps(generated_tail, indent=4, ensure_ascii=False))


def load_head(config, data_type):

    with open(os.path.join(config["directories"]["patent"]["silver_data"], f"{data_type}_head.json"), "r") as f:
        head = json.load(f)
    
    return head[:5000]


def main(config, data_type="情報系"):

    model = Model(config)

    with open(os.path.join(config["directories"]["patent"]["gold_data"], f"{data_type}_adv.json"), "r") as f:
        data = json.load(f)
    
    #generated_head = generate_head(model, data, config, data_type)
    generated_head = load_head(config, data_type)
    generate_tail(model, data, config, generated_head, data_type)



if __name__ == "__main__":

    with open("./settings.json", "r") as f:
        config = json.load(f)

    random.seed(1)

    os.makedirs(config["directories"]["patent"]["silver_data"], exist_ok=True)

    main(config)