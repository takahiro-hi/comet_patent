import torch, os, random, json
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

from prompt_template import construct_prompt_for_head, construct_prompt_for_tail


class Model(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.config = config
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["generate_silver"]["model_name"])    
        self.model = AutoModelForCausalLM.from_pretrained(self.config["generate_silver"]["model_name"], device_map="auto")
        self.model.eval()
    

    def forward(self, prompt, type, temperature=None, eos_word="\n"):

        assert type in ["head", "tail"], "type must be either 'head' or 'tail'."

        if temperature is None:
            temperature = self.config["generate_silver"][type]["temperature"]

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(input_ids,
                                    do_sample = True,
                                    output_scores = False,
                                    return_dict_in_generate = False,
                                    top_k = 0.,
                                    top_p = self.config["generate_silver"][type]["top_p"],
                                    temperature = temperature,
                                    num_return_sequences = self.config["generate_silver"][type]["return_num"],
                                    max_new_tokens = 50
                                    )
        
        generated_data = self._get_content(outputs, input_ids, eos_word)
        
        return generated_data
    

    def _get_content(self, output_seq, input_ids, eos_word):

        output_ids = [output_id[input_ids.size(1):] for output_id in output_seq]
        output_content = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        ret_content = [o[:o.index(eos_word)].strip() if eos_word in o else o.strip() for o in output_content]

        return ret_content
    


if __name__ == "__main__":

    with open("./settings.json", "r") as f:
        config = json.load(f)
    
    with open("./data/gold/情報系_adv.json", "r") as f:
        data = json.load(f)
    
    model = Model(config)

    random.seed(1)
    prompt = construct_prompt_for_tail(data, config["generate_silver"]["tail"]["few_shot_exam"], target_head="ipadを使って検索をする。")

    generated_data = model(prompt, config["generate_silver"]["tail"]["temperature"], "tail")
    for idx, data in enumerate(generated_data):
        print(f"{idx+1}: {data}")