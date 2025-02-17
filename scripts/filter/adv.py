import torch, os, json, random, argparse
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils import get_settings, plot_patent_metrics, plot_x_iters, test_adv_base

from loss_func import LossFuncForDiscriminator
from dataloader import DataLoader
from model import Classifier




class SymbolicKDUsingAdversarialNet():

    def __init__(self, result_path, settings, args):

        self.result_path = result_path
        self.settings = settings
        self.args = args
        self.tr_params = self.settings["tr_params"]
        
        # settings for training
        self.selector = self.initialize_model()
        self.discriminator = self.initialize_model()
        self.optim_s = optim.AdamW(self.selector.parameters(), lr=self.tr_params["selector"]["learning_rate"], weight_decay=self.tr_params["selector"]["weight_decay"])
        self.optim_d = optim.AdamW(self.discriminator.parameters(), lr=self.tr_params["discriminator"]["learning_rate"], weight_decay=self.tr_params["discriminator"]["weight_decay"])
        self.loss_func_d = LossFuncForDiscriminator(self.result_path, self.tr_params)
        self.loss_func_s = nn.BCEWithLogitsLoss()    

        # dataset
        self.dataloader = DataLoader(self.settings, self.args)

        # log files
        self.log_warning = os.path.join(self.result_path, "log_warning.txt")
        self.log_train_s = os.path.join(self.result_path, "log_selector.txt")
        self.log_train_d = os.path.join(self.result_path, "log_discriminator.txt")
        self.log_data_w, self.log_data_s, self.log_data_d  = "", "", ""

        # result list
        self.loss_d_tr, self.loss_s_tr = {"iteration":[], "loss":[]}, {"iteration":[], "loss":[]}
        self.val_patent = {
            "iteration":[], 
            "selector": { "acc":[], "precision":[], "recall":[], "f1":[]},
            "discriminator": {"acc":[], "precision":[], "recall":[], "f1":[]}
        }


    def initialize_model(self):

        if self.args.init_filter == "0":
            filter = Classifier(self.tr_params["model_name"]).to("cuda")
        
        elif self.args.init_filter == "1":
            if self.tr_params["augmentation"]["flag"]:
                model_path = os.path.join(self.settings["result_path"].format(self.args.patent_domain, self.args.temperature_tail), f"filter_base/augmentation_es/filter.pth")
            else:
                model_path = os.path.join(self.settings["result_path"].format(self.args.patent_domain, self.args.temperature_tail), "filter_base/no_augmentation_es/filter.pth")
            state_dict = torch.load(model_path, map_location="cuda", weights_only=True)
            filter = Classifier(self.tr_params["model_name"]).to("cuda")
            filter.load_state_dict(state_dict)
        
        return torch.nn.DataParallel(filter)


    def train_gan(self):
        
        for iteration in tqdm(range(1, 1+self.tr_params["iterations"]), desc="training"):                

            _temp_loss_s, _temp_loss_d = [], []
            
            # validate patent
            if iteration%10==1:
                self.validate_patent(iteration)

            # train selector
            for step in range(1, self.tr_params["selector"]["steps"]+1):
                loss_s = self.train_selector(iteration, step)
                if loss_s is not None:
                    _temp_loss_s.append(loss_s.item())
            if len(_temp_loss_s) != 0:
                self.loss_s_tr["iteration"].append(iteration)
                self.loss_s_tr["loss"].append(sum(_temp_loss_s)/len(_temp_loss_s))

            # train discriminator
            for step in range(1, self.tr_params["discriminator"]["steps"]+1):
                loss_d = self.train_discriminator(iteration, step)
                _temp_loss_d.append(loss_d.item())
            self.loss_d_tr["iteration"].append(iteration)
            self.loss_d_tr["loss"].append(sum(_temp_loss_d)/len(_temp_loss_d))

            # print logs
            for txt, file in zip([self.log_data_s, self.log_data_d, self.log_data_w], [self.log_train_s, self.log_train_d, self.log_warning]):
                with open(file, mode="a") as f:
                    f.write(txt)
            self.log_data_s = self.log_data_d = self.log_data_w = ""
        
        # save results
        torch.save(self.selector.module.state_dict(), os.path.join(self.result_path, "selector.pth"))
        torch.save(self.discriminator.module.state_dict(), os.path.join(self.result_path, "discriminator.pth"))

        self.plot_result()


    def train_selector(self, iteration, step):

        self.discriminator.eval()
        self.selector.train()

        data, inp, label, ans_prob = self.dataloader.sampling_to_train_selector(self.discriminator, iteration)

        if inp is None:
            self.log_data_w += f"\n\ncannot keep balance (pos:{torch.sum(label==1).item()}, neg:{torch.sum(label==0).item()})\n"
            self.log_data_w += f"iteration-step : {iteration}-{step}\n"
            for d, p in zip(data[:50], ans_prob[:50]):
                self.log_data_w += f"ans : {int(p.item()>=0.5)} ({p.item()}), data : {d}\n"
            return None

        pre_s = self.selector(**inp.to("cuda")).squeeze(-1)
        loss_s = self.loss_func_s(pre_s, torch.tensor(label).to("cuda"))

        self.optim_s.zero_grad()
        loss_s.backward()
        self.optim_s.step()

        self.take_log_selector(iteration, step, data, label, ans_prob, pre_s, loss_s.item())

        return loss_s.cpu()
    

    def train_discriminator(self, iteration, step):

        self.discriminator.train()
        self.selector.eval()

        tr_data, tr_inp, tr_label = self.dataloader.sampling_to_train_discriminator(self.selector, iteration, step)

        pred_d = self.discriminator(**tr_inp.to("cuda")).squeeze(-1)
        loss_d = self.loss_func_d(pred_d, torch.tensor(tr_label).to("cuda"), iteration)
        self.optim_d.zero_grad()
        loss_d.backward()
        self.optim_d.step()

        self.take_log_discriminator(iteration, step, tr_data, tr_label, torch.sigmoid(pred_d), loss_d.item())

        return loss_d.cpu()


    def take_log_selector(self,iteration, step, data, label, ans_prob, pre_s, loss):

        self.log_data_s += f"\n\n#################### {iteration}-{step} (loss:{loss}) ####################\n\n"        
        for d, l, a_prob, pre_prob in zip(data, label, ans_prob, torch.sigmoid(pre_s)):
            self.log_data_s += f"ans:{int(l)}({a_prob}), pre:{round(pre_prob.item(), 3)}, sentence:{d}\n"


    def take_log_discriminator(self, iteration, step, data, label, prob, loss):

        self.log_data_d += f"\n\n#################### {iteration}-{step} (loss:{loss}) ####################\n\n"
        for d, l, p in zip(data, label, prob):
            self.log_data_d += f"ans:{int(l)}, pre:{round(p.item(), 3)}, sentence:{d}\n"


    def validate_patent(self, iteration):

        self.selector.eval()
        self.discriminator.eval()

        self.val_patent["iteration"].append(iteration)
        label, inps = self.dataloader.val_patent_label, self.dataloader.val_patent_inps

        for model, model_name in zip([self.selector, self.discriminator], ["selector", "discriminator"]):
            with torch.no_grad():
                pre = model(**inps.to("cuda")).squeeze(-1)
            prob = torch.sigmoid(pre)
            pred_label = prob.cpu().numpy() > 0.5
            loss = self.loss_func_s(pre, torch.tensor(label).to("cuda"))

            self.val_patent[model_name]["acc"].append(accuracy_score(label, pred_label))
            self.val_patent[model_name]["precision"].append(precision_score(label, pred_label, zero_division=0))
            self.val_patent[model_name]["recall"].append(recall_score(label, pred_label, zero_division=0))
            self.val_patent[model_name]["f1"].append(f1_score(label, pred_label, zero_division=0))


    def plot_result(self):

        title = f"adv_{self.args.patent_domain}_{self.args.temperature_tail}_{self.args.init_filter}"

        plot_x_iters(
            values = {"selector": self.loss_s_tr, "discriminator": self.loss_d_tr},
            save_file_path = os.path.join(result_path, "loss"),
            save_value_flag = True,
            title = title
        )

        plot_x_iters(
            values = {"threshold": self.dataloader.thresholds_list_s_to_d},
            save_file_path = os.path.join(result_path, "threshold"),
            save_value_flag = True,
            title = title
        )

        plot_x_iters(
            values = {"ratio": self.dataloader.patent_data_ratio},
            save_file_path = os.path.join(result_path, "patent_data_ratio"),
            save_value_flag = True,
            title = title
        )

        plot_patent_metrics(self.val_patent, "selector", os.path.join(self.result_path, "patent_metrics_selector.png"))
        plot_patent_metrics(self.val_patent, "discriminator", os.path.join(self.result_path, "patent_metrics_discriminator.png"))


    def test(self):

        model_path = os.path.join(self.settings["result_path"].format(self.args.patent_domain, self.args.temperature_tail), f"filter_base/no_augmentation_es/filter.pth")
        state_dict = torch.load(model_path, map_location="cuda", weights_only=True)
        base_model = Classifier(self.tr_params["model_name"]).to("cuda")
        base_model.load_state_dict(state_dict)

        test_adv_base(base_model, self.selector, self.dataloader.tokenizer, self.settings["data"], self.args.patent_domain, self.result_path)




def main(result_path, settings, args):

    gan = SymbolicKDUsingAdversarialNet(result_path, settings, args)
    gan.train_gan()
    gan.test()





if __name__=="__main__":

    """
    python scripts/filter/adv.py --no "no1" --device_ids "3" --init_filter "1" --patent_domain "情報系" --temperature_tail 1.3
    """

    random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--no", type=str)
    parser.add_argument("--device_ids", type=str)
    parser.add_argument("--init_filter", type=str)    # 0: use pretrained model, 1: use base model
    parser.add_argument("--patent_domain", type=str)
    parser.add_argument("--temperature_tail", type=float, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids

    settings = get_settings("adv")

    if settings["tr_params"]["augmentation"]["flag"]:
        result_path = os.path.join(settings["result_path"].format(args.patent_domain, args.temperature_tail), f"filter_adv/augmentation/init_{args.init_filter}_{args.no}")
    else:
        result_path = os.path.join(settings["result_path"].format(args.patent_domain, args.temperature_tail), f"filter_adv/no_augmentation/init_{args.init_filter}_{args.no}")
    os.makedirs(result_path)
    
    with open(os.path.join(result_path, "settings.json"), "w") as f:
        f.write(json.dumps(settings, indent=4, ensure_ascii=False))
    
    main(result_path, settings, args)