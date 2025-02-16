import random, torch, os, japanize_matplotlib, sys
import matplotlib.pyplot as plt

from transformers import BertJapaneseTokenizer

from utils import tokenize_data

sys.path.append("./scripts/utils")
from load_data import load_gold, load_silver
#from utils_load_data import load_gold, load_silver, load_train_data_for_step1



def get_val_patent(args, settings):

    pos = load_gold(settings["data"], "patent", args.patent_domain)
    neg = load_silver(settings["data"], args.temperature_tail, "patent", args.patent_domain)

    print(f"patent_domain: {args.patent_domain}")
    print(f"pos: {len(pos)}, neg: {len(neg)}")

    assert len(pos) < len(neg), "somethig wrong with the data"

    neg = random.sample(neg, len(pos))
    data = pos + neg
    label = [1.] * len(pos) + [0.] * len(neg)

    combined = list(zip(data, label))
    random.shuffle(combined)
    data[:], label[:] = zip(*combined)

    return data, label




class DataLoader:

    def __init__(self, settings, args):

        self.settings = settings
        self.tr_params = settings["tr_params"]
        self.args = args

        self.tokenizer = BertJapaneseTokenizer.from_pretrained(self.tr_params["model_name"])

        # load dataset
        self.gold_data_dict = {
            "patent": load_gold(self.settings["data"], "patent", args.patent_domain),
            "atomic": load_gold(self.settings["data"], "atomic"),
            "mark_idx": {
                "patent": 0,
                "atomic": 0
            }
        }
        self.silver_data_dict = {
            "patent": load_silver(self.settings["data"], self.args.temperature_tail, "patent", args.patent_domain),
            "atomic": load_silver(self.settings["data"], None, "atomic"),
            "mark_idx": {
                "patent": 0,
                "atomic": 0
            }
        }
        for _type in ["patent", "atomic"]:
            random.shuffle(self.gold_data_dict[_type])
            random.shuffle(self.silver_data_dict[_type])
        self.val_patent_data, self.val_patent_label = get_val_patent(args, settings)
        self.val_patent_inps = tokenize_data(self.val_patent_data, self.tokenizer)

        # variables to manage sampling from data
        self.thresholds_list_s_to_d = {"iteration": [], "threshold": []}
        self.patent_data_ratio = self.get_patent_data_ratio()
    

    def get_patent_data_ratio(self):

        if self.tr_params["switch"]["flag"]:
            ret_list = {"iteration": [], "ratio": []}
            for i in range(self.tr_params["iterations"]):
                ret_list["iteration"].append(i)
                if i < self.tr_params["switch"]["end_iters"]:
                    ret_list["ratio"].append(self.tr_params["switch"]["start_ratio"] + (1.0 - self.tr_params["switch"]["start_ratio"]) * i / self.tr_params["switch"]["end_iters"])
                else:
                    ret_list["ratio"].append(1.0)
        else:
            ret_list = {
                "iteration": [i for i in range(self.tr_params["iterations"])],
                "ratio": [1.0 for _ in range(self.tr_params["iterations"])]
            }

        return ret_list
    
    
    def random_sampling_from_gold(self, num, gold_type, flag_tokenize=True):

        data_list = self.gold_data_dict[gold_type]
        selected_data = data_list[self.gold_data_dict["mark_idx"][gold_type]:self.gold_data_dict["mark_idx"][gold_type] + num]
        self.gold_data_dict["mark_idx"][gold_type] += num

        if self.gold_data_dict["mark_idx"][gold_type] >= len(data_list):
            self.gold_data_dict["mark_idx"][gold_type] = num - len(selected_data)
            random.shuffle(data_list)
            selected_data += data_list[:self.gold_data_dict["mark_idx"][gold_type]]

        if flag_tokenize:
            return selected_data, tokenize_data(selected_data, self.tokenizer)
        else:
            return selected_data
    

    def random_sampling_from_silver(self, num, silver_type, flag_permutation, flag_tokenize):

        data_list = self.silver_data_dict[silver_type]

        if flag_permutation:    # 順番に取り出す
            ret_data = data_list[self.silver_data_dict["mark_idx"][silver_type]:self.silver_data_dict["mark_idx"][silver_type] + num]
            self.silver_data_dict["mark_idx"][silver_type] += num

            if self.silver_data_dict["mark_idx"][silver_type] >= len(data_list):
                self.silver_data_dict["mark_idx"][silver_type] = num - len(ret_data)
                random.shuffle(data_list)
                ret_data += data_list[:self.silver_data_dict["mark_idx"][silver_type]]
        
        else:   # 全体からランダムに取り出す
            ret_data = random.sample(data_list, num)
        
        if flag_tokenize:
            return ret_data, tokenize_data(ret_data, self.tokenizer)
        else:
            return ret_data


    def get_mixed_data(self, gold_silver, iteration, num, flag_permutation):

        num_patent = int(num * self.patent_data_ratio["ratio"][iteration-1])
        num_atomic = num - num_patent

        if gold_silver == "gold":
            _data_patent = self.random_sampling_from_gold(num_patent, "patent", False)
            _data_atomic = self.random_sampling_from_gold(num_atomic, "atomic", False)
            data = _data_patent + _data_atomic
            inps = tokenize_data(data, self.tokenizer)
        elif gold_silver == "silver":
            _data_patent = self.random_sampling_from_silver(num_patent, "patent", flag_permutation, False)
            _data_atomic = self.random_sampling_from_silver(num_atomic, "atomic", flag_permutation, False)
            data = _data_patent + _data_atomic
            inps = tokenize_data(data, self.tokenizer)
        else:
            raise ValueError("invalid gold_silver value")
    
        return data, inps
        
    
    def sampling_to_train_selector(self, discriminator, iteration):

        max_attempt = 10

        ret_data, ret_label, ret_prob, ret_logits = [], [], [], []
        remaining_samples = {0.: self.tr_params["selector"]["batch_size"]//2, 1.: self.tr_params["selector"]["batch_size"]//2}

        for _ in range(max_attempt):
            data, inps = self.get_mixed_data("silver", iteration, self.tr_params["selector"]["select_num"], False)
            with torch.no_grad():
                ans_logit = discriminator(**inps.to("cuda")).squeeze(-1)
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
                return ret_data, tokenize_data(ret_data, self.tokenizer), ret_label, torch.stack(ret_prob)

        return data, None, ans_label, ans_prob

    
    def sampling_to_train_discriminator(self, selector, iteration, step):

        params_d = self.tr_params["discriminator"]
        data_gold, inps_gold = self.get_mixed_data("gold", iteration, params_d["batch_size"]//2, True)

        if iteration < params_d["sample_from_silver"]["end_iters"]:
            lower = params_d["sample_from_silver"]["start_thresh"] + iteration * (params_d["sample_from_silver"]["end_thresh"] - params_d["sample_from_silver"]["start_thresh"]) / params_d["sample_from_silver"]["end_iters"]
        else:
            lower = params_d["sample_from_silver"]["end_thresh"]
        
        if iteration % params_d["sample_from_silver"]["steps_random"] == 0 and step == 1:
            data_silver, inps_silver = self.get_mixed_data("silver", iteration, params_d["batch_size"]//2, True)
            _temp_thread = lower

        else:
            data_silver_large, inp_silver = self.get_mixed_data("silver", iteration, params_d["select_num"], True)
            with torch.no_grad():
                pre_logits_s = selector(**inp_silver.to("cuda")).squeeze(-1)
                pre_prob_s = torch.sigmoid(pre_logits_s)
            
            _temp_thread = lower
            while True:
                valid_indices = (_temp_thread <= pre_prob_s) & (pre_prob_s <= 1.0)
                selected_indices = torch.where(valid_indices)[0]
                if len(selected_indices) >= params_d["batch_size"]//2:
                    sampled_index = random.sample(selected_indices.tolist(), params_d["batch_size"]//2)
                    break
                _temp_thread -= 0.05
            
            data_silver = [data_silver_large[i] for i in sampled_index]

        if step == 1:
            self.thresholds_list_s_to_d["iteration"].append(iteration)
            self.thresholds_list_s_to_d["threshold"].append(_temp_thread)

        ret_data = data_gold + data_silver
        return ret_data, tokenize_data(ret_data, self.tokenizer), [1.] * len(data_gold) + [0.] * len(data_silver)


    def plot_gold_ratio(self):

        plt.figure()
        plt.plot(self.patent_data_ratio)
        plt.xlabel("iteration", fontsize=13)
        plt.ylabel("patent data ratio", fontsize=13)
        plt.grid(True)
        plt.savefig(os.path.join(self.result_path, "patent_data_ratio.png"))
        plt.close()
        