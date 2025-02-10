import os, json, random
from tqdm import tqdm

from prompt_template import construct_prompt_for_head, construct_prompt_for_tail
from model import Model



def generate_head(model, data_list, config, try_num=500):

    generated_head = []

    for _ in tqdm(range(try_num), desc="generating head..."):
        prompt = construct_prompt_for_head(data_list, 10)
        generated_data = model(prompt, "head", temperature=config["generate_silver"]["head"]["temperature"])
        generated_head.extend(generated_data)
    
    generated_head = list(set(generated_head))
    
    original_head = [d[0] for d in data_list]
    saved_data = list(set(generated_head) - set(original_head))

    with open(os.path.join(config["directories"]["atomic"]["silver_data"], f"head.json"), "w") as f:
        f.write(json.dumps(saved_data, indent=4, ensure_ascii=False))
    
    return saved_data
    

def generate_tail(model, data_list, config, head_list):

    generated_tail = []

    for head in tqdm(head_list, desc="generating tail..."):
        prompt = construct_prompt_for_tail(data_list, config["generate_silver"]["tail"]["few_shot_exam"], target_head=head)
        generated_data = model(prompt, "tail", temperature=1.7)
        generated_data = list(set(generated_data))
        generated_tail.append({"head": head, "tail": generated_data})
    
    with open(os.path.join(config["directories"]["atomic"]["silver_data"], f"triple.json"), "w") as f:
        f.write(json.dumps(generated_tail, indent=4, ensure_ascii=False))


def main(config):

    model = Model(config)

    with open(f"./data/atomic/gold/data_after.json", "r") as f:
        data = json.load(f)
    
    generated_head = generate_head(model, data, config)
    generate_tail(model, data, config, generated_head)



if __name__ == "__main__":

    with open("./settings.json", "r") as f:
        config = json.load(f)

    random.seed(1)

    os.makedirs(config["directories"]["atomic"]["silver_data"], exist_ok=True)

    main(config)