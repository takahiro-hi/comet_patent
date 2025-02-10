import random, torch, os, japanize_matplotlib
import matplotlib.pyplot as plt

from transformers import BertJapaneseTokenizer

from utils_load_data import load_gold, load_silver, load_train_data_for_step1
from base import tokenize_data


class DataLoader:

    def __init__(self, device, config, args, result_path):
        
        self.device = device
        self.args = args
        self.config = config
        self.result_path = result_path

        self.tokenizer = BertJapaneseTokenizer.from_pretrained(config["adv_train"]["model_name"])

        # load dataset
        self.gold_data_dict = {
            "patent": load_gold(config, "patent", args.data_type),
            "atomic": load_gold(config, "atomic"),
            "mark_idx": {
                "patent": 0,
                "atomic": 0
            }
        }
        self.silver_data_dict = {
            "patent": load_silver(config, "patent", args.data_type),
            "atomic": load_silver(config, "atomic"),
            "mark_idx": {
                "patent": 0,
                "atomic": 0
            }
        }
        self.val_p_data, self.val_p_label = load_train_data_for_step1(config, "patent", args.data_type)
        self.val_p_inps = tokenize_data(self.val_p_data, self.tokenizer, self.device)

        # variables to manage sampling from data
        self.thresholds_list_s_to_d = []
        self.patent_data_ratio = self.get_patent_data_ratio()
    

    def get_patent_data_ratio(self):

        if self.config["adv_train"]["switch"]["flag"]:
            _patent_data_ratio = []
            for i in range(self.config["adv_train"]["iterations"]):
                if i < self.config["adv_train"]["switch"]["iters"]:
                    _patent_data_ratio.append(self.config["adv_train"]["switch"]["start_ratio"] + (1.0 - self.config["adv_train"]["switch"]["start_ratio"]) * i / self.config["adv_train"]["switch"]["iters"])
                else:
                    _patent_data_ratio.append(1.0)
        else:
            _patent_data_ratio = [1.0 for i in range(self.config["adv_train"]["iterations"])]

        return _patent_data_ratio
    
    
    def random_sampling_from_gold(self, num, gold_type, flag_tokenize=True):

        data_list = self.gold_data_dict[gold_type]
        selected_data = data_list[self.gold_data_dict["mark_idx"][gold_type]:self.gold_data_dict["mark_idx"][gold_type] + num]
        self.gold_data_dict["mark_idx"][gold_type] += num

        if self.gold_data_dict["mark_idx"][gold_type] >= len(data_list):
            self.gold_data_dict["mark_idx"][gold_type] = num - len(selected_data)
            random.shuffle(data_list)
            selected_data += data_list[:self.gold_data_dict["mark_idx"][gold_type]]

        if flag_tokenize:
            return selected_data, tokenize_data(selected_data, self.tokenizer, self.device)
        else:
            return selected_data
    

    def random_sampling_from_silver(self, num, silver_type, flag_permutation, flag_tokenize):

        data_list = self.silver_data_dict[silver_type]

        if flag_permutation:
            ret_data = data_list[self.silver_data_dict["mark_idx"][silver_type]:self.silver_data_dict["mark_idx"][silver_type] + num]
            self.silver_data_dict["mark_idx"][silver_type] += num

            if self.silver_data_dict["mark_idx"][silver_type] >= len(data_list):
                self.silver_data_dict["mark_idx"][silver_type] = num - len(ret_data)
                random.shuffle(data_list)
                ret_data += data_list[:self.silver_data_dict["mark_idx"][silver_type]]
        
        else:
            ret_data = random.sample(data_list, num)
        
        if flag_tokenize:
            return ret_data, tokenize_data(ret_data, self.tokenizer, self.device)
        else:
            return ret_data


    def get_mixed_data(self, gold_silver, iteration, num, flag_permutation):

        num_patent = int(num * self.patent_data_ratio[iteration-1])
        num_atomic = num - num_patent

        if gold_silver == "gold":
            _data_patent = self.random_sampling_from_gold(num_patent, "patent", False)
            _data_atomic = self.random_sampling_from_gold(num_atomic, "atomic", False)
            data = _data_patent + _data_atomic
            inps = tokenize_data(data, self.tokenizer, self.device)
        elif gold_silver == "silver":
            _data_patent = self.random_sampling_from_silver(num_patent, "patent", flag_permutation, False)
            _data_atomic = self.random_sampling_from_silver(num_atomic, "atomic", flag_permutation, False)
            data = _data_patent + _data_atomic
            inps = tokenize_data(data, self.tokenizer, self.device)
    
        return data, inps
        
    
    def sampling_to_train_selector(self, discriminator, iteration):

        max_attempt = 10

        ret_data, ret_label, ret_prob, ret_logits = [], [], [], []
        remaining_samples = {0.: self.config["adv_train"]["selector"]["batch_size"]//2, 1.: self.config["adv_train"]["selector"]["batch_size"]//2}

        for _ in range(max_attempt):
            data, inps = self.get_mixed_data("silver", iteration, self.config["adv_train"]["selector"]["select_num"], False)
            with torch.no_grad():
                ans_logit = discriminator(**inps.to(self.device)).squeeze(-1)
                ans_prob = torch.sigmoid(ans_logit)
                ans_label = (ans_prob >= 0.5).float()
            for label in [0., 1.]:
                if remaining_samples[label] > 0:
                    indices = torch.where(ans_label == label)[0]
                    sampled_indices = random.sample(indices.tolist(), min(len(indices), remaining_samples[label]))

                    if len(sampled_indices) > 0:
                        ret_logits.extend([ans_logit[i] for i in sampled_indices])
                        ret_data.extend([data[i] for i in sampled_indices])
                        ret_label.extend([label for _ in range(len(sampled_indices))])
                        ret_prob.extend([ans_prob[i] for i in sampled_indices])
                        remaining_samples[label] -= len(sampled_indices)
            
            if all(v <= 0 for v in remaining_samples.values()):
                return ret_data, tokenize_data(ret_data, self.tokenizer, self.device), ret_label, torch.stack(ret_prob), torch.stack(ret_logits)

        return data, None, ans_label, ans_prob, ans_logit

    
    def sampling_to_train_discriminator(self, selector, iteration, step):

        config_d = self.config["adv_train"]["discriminator"]

        data_gold, inps_gold = self.get_mixed_data("gold", iteration, config_d["batch_size"]//2, True)

        if iteration < config_d["sampling_of_silver"]["end_iters"]:
            lower = config_d["sampling_of_silver"]["start_threshold"] + iteration * (config_d["sampling_of_silver"]["end_threshold"] - config_d["sampling_of_silver"]["start_threshold"]) / config_d["sampling_of_silver"]["end_iters"]
        else:
            lower = config_d["sampling_of_silver"]["end_threshold"]
        
        if config_d["sampling_of_silver"]["flag_random"] and iteration % config_d["sampling_of_silver"]["steps_random"] == 0 and step == 1:
            data_silver, inps_silver = self.get_mixed_data("silver", iteration, config_d["batch_size"]//2, True)
            _temp_thread = lower

        else:
            data_silver_large, inp_silver = self.get_mixed_data("silver", iteration, config_d["select_num"], True)
            with torch.no_grad():
                pre_logits_s = selector(**inp_silver.to(self.device)).squeeze(-1)
                pre_prob_s = torch.sigmoid(pre_logits_s)
            
            _temp_thread = lower
            
            while True:
                valid_indices = (_temp_thread <= pre_prob_s) & (pre_prob_s <= 1.0)
                selected_indices = torch.where(valid_indices)[0]
                if len(selected_indices) >= config_d["batch_size"]//2:
                    sampled_index = random.sample(selected_indices.tolist(), config_d["batch_size"]//2)
                    break
                _temp_thread -= 0.05
            
            data_silver = [data_silver_large[i] for i in sampled_index]
            
        self.thresholds_list_s_to_d.append(_temp_thread)    

        ret_data = data_gold + data_silver
        return ret_data, tokenize_data(ret_data, self.tokenizer, self.device), [1.] * len(data_gold) + [0.] * len(data_silver)
    

    def plot_thresholds(self):

        avg_threshold = [self.thresholds_list_s_to_d[i] for i in range(0, len(self.thresholds_list_s_to_d), len(self.thresholds_list_s_to_d)//self.config["adv_train"]["iterations"])]

        plt.figure()
        plt.plot(avg_threshold)
        plt.xlabel("iteration", fontsize=13)
        plt.ylabel("threshold", fontsize=13)
        plt.grid(True)
        plt.savefig(os.path.join(self.result_path, "threshold_s_to_d.png"))
        plt.close()
    

    def plot_gold_ratio(self):

        plt.figure()
        plt.plot(self.patent_data_ratio)
        plt.xlabel("iteration", fontsize=13)
        plt.ylabel("patent data ratio", fontsize=13)
        plt.grid(True)
        plt.savefig(os.path.join(self.result_path, "patent_data_ratio.png"))
        plt.close()
        
