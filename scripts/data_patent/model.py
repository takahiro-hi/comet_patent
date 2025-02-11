import torch, os, random, json, sys
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import pandas as pd

from prompt_template import construct_prompt_for_head, construct_prompt_for_tail

sys.path.append("./scripts/utils")
from load_data import load_gold



class Model(nn.Module):

    def __init__(self, settings):

        super().__init__()

        self.settings = settings["parameters_silver"]
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.settings["model"])    
        self.model = AutoModelForCausalLM.from_pretrained(self.settings["model"], device_map="auto")
        self.model.eval()


    def forward(self, prompt, type, temperature=None):

        assert type in ["head", "tail"], "type must be either 'head' or 'tail'."

        if temperature is None:
            temperature = self.settings[type]["temperature"]

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(input_ids,
                                    do_sample = True,
                                    output_scores = False,
                                    return_dict_in_generate = False,
                                    top_k = 0.,
                                    top_p = self.settings[type]["top_p"],
                                    temperature = temperature,
                                    num_return_sequences = self.settings[type]["return_num"],
                                    max_new_tokens = 50,
                                    )

        generated_data = self._get_content(outputs, input_ids)

        return generated_data


    def _get_content(self, output_seq, input_ids):

        output_ids = [output_id[input_ids.size(1):] for output_id in output_seq]
        output_content = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        output_content = [o[:o.index("\n")] if "\n" in o else o for o in output_content]
        output_content = [o[:o.index("。")] if "。" in o else o for o in output_content]

        return output_content




if __name__ == "__main__":

    with open("./settings_data.json", "r") as f:
        settings = json.load(f)
    
    data = load_gold(settings, "patent", "情報系")
    
    model = Model(settings)

    random.seed(1)
    prompt = construct_prompt_for_tail(data, settings["parameters_silver"]["tail"]["few_shot_exam"], target_head="ipadを使って検索をする。")

    #for temperature in [1.3, 1.0, 0.8]:
    for temperature in [0.8]:
        print(f"temperature: {temperature}")
        generated_data = model(prompt, "tail", temperature)
        for idx, data in enumerate(generated_data):
            print(f"{idx+1}: {data}")