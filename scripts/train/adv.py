import torch, configparser, os, json, random, argparse
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from loss_func import LossFuncForDiscriminator
from dataloader import DataLoader
from classifier import Classifier
from utils_plot import plot_loss, log_loss, plot_metrics_val, log_metric_val, plot_patent_metrics



class SymbolicKDUsingAdversarialNet():

    def __init__(self, result_path, config, args):

        self.result_path = result_path
        self.config_adv = config["adv_train"]
        self.config = config
        self.args = args

        # seggings for device
        torch.cuda.set_device(2)
        self.device = "cuda"
        self.cuda_ids = [2]
        
        # settings for training
        self.selector, self.discriminator = self.get_pretrained_classifier()
        self.optim_s = optim.AdamW([
            {"params": self.selector.module.model.parameters(), "lr": self.config_adv["selector"]["lr_bert"], "weight_decay": self.config_adv["selector"]["weight_decay"]},
            {"params": self.selector.module.dense.parameters(), "lr": self.config_adv["selector"]["lr_head"], "weight_decay": self.config_adv["selector"]["weight_decay"]},
            {"params": self.selector.module.out_proj.parameters(), "lr": self.config_adv["selector"]["lr_head"], "weight_decay": self.config_adv["selector"]["weight_decay"]}
        ])
        self.optim_d = optim.AdamW([
            {"params": self.discriminator.module.model.parameters(), "lr": self.config_adv["discriminator"]["lr_bert"], "weight_decay": self.config_adv["discriminator"]["weight_decay"]},
            {"params": self.discriminator.module.dense.parameters(), "lr": self.config_adv["discriminator"]["lr_head"], "weight_decay": self.config_adv["discriminator"]["weight_decay"]},
            {"params": self.discriminator.module.out_proj.parameters(), "lr": self.config_adv["discriminator"]["lr_head"], "weight_decay": self.config_adv["discriminator"]["weight_decay"]}
        ])

        self.loss_func_d = LossFuncForDiscriminator(self.result_path, self.config)
        self.loss_func_ce = nn.BCEWithLogitsLoss()
        self.loss_func_s = nn.BCEWithLogitsLoss()    

        # dataset
        self.dataloader = DataLoader(self.device, self.config, self.args, self.result_path)

        # log files
        self.log_warning = os.path.join(self.result_path, "log_warning.txt")
        self.log_train_s = os.path.join(self.result_path, "log_selector.txt")
        self.log_train_d = os.path.join(self.result_path, "log_discriminator.txt")
        self.log_data_w, self.log_data_s, self.log_data_d  = "", "", ""

        # result list
        self.loss_d_list, self.loss_s_train_list = [], []
        self.val_patent = {
            "iteration":[], 
            "selector": {"loss":[], "acc":[], "precision":[], "recall":[], "f1":[]},
            "discriminator": {"loss":[], "acc":[], "precision":[], "recall":[], "f1":[]}
        }


    def get_pretrained_classifier(self):

        if self.config_adv["switch"]["flag"]:
            #model_path = os.path.join(self.config["directories"]["step_1"], f"atomic/none/es/selector.pth")
            selector = Classifier(self.config_adv["model_name"]).to(self.device)
            discriminator = Classifier(self.config_adv["model_name"]).to(self.device)
        else:
            model_path = os.path.join(self.config["directories"]["step_1"], f"patent/{self.args.data_type}/es/selector.pth")
            
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)

            selector = Classifier(self.config_adv["model_name"]).to(self.device)
            selector.load_state_dict(state_dict)

            discriminator = Classifier(self.config_adv["model_name"]).to(self.device)
            discriminator.load_state_dict(state_dict)
        
        return torch.nn.DataParallel(selector, device_ids=self.cuda_ids), torch.nn.DataParallel(discriminator, device_ids=self.cuda_ids)
  

    def train_gan(self):
        
        for iteration in tqdm(range(1, 1+self.config_adv["iterations"]), desc="training"):

            if self.config_adv["switch"]["flag"] and iteration == self.config_adv["switch"]["iters"]:
                plot_patent_metrics(self.val_patent, "selector", os.path.join(self.result_path, "patent_metrics_selector.png"))
                plot_patent_metrics(self.val_patent, "discriminator", os.path.join(self.result_path, "patent_metrics_discriminator.png"))

            _temp_loss_s, _temp_loss_d = [], []
            
            # validate patent
            if iteration%10==1:
                self.validate_patent(iteration)

            # train selector
            for step in range(1, self.config_adv["selector"]["steps"]+1):
                loss_s = self.train_selector(iteration, step)
                if loss_s is not None:
                    _temp_loss_s.append(loss_s.item())

            # train discriminator
            for step in range(1, self.config_adv["discriminator"]["steps"]+1):
                loss_d = self.train_discriminator(iteration, step)
                _temp_loss_d.append(loss_d.item())

            # save loss
            if len(_temp_loss_s) == 0:
                if len(self.loss_s_train_list) == 0:
                    self.loss_s_train_list.append(0.)
                else:
                    self.loss_s_train_list.append(self.loss_s_train_list[-1])
            else:
                self.loss_s_train_list.append(sum(_temp_loss_s)/len(_temp_loss_s))
            self.loss_d_list.append(sum(_temp_loss_d)/len(_temp_loss_d))

            # print logs
            for txt, file in zip([self.log_data_s, self.log_data_d, self.log_data_w], [self.log_train_s, self.log_train_d, self.log_warning]):
                with open(file, mode="a") as f:
                    f.write(txt)
            self.log_data_s = self.log_data_d = self.log_data_w = ""

            torch.cuda.empty_cache()
        
        # save results
        self.plot_result()

        torch.save(self.selector.module.state_dict(), os.path.join(self.result_path, "selector.pth"))
        torch.save(self.discriminator.module.state_dict(), os.path.join(self.result_path, "discriminator.pth"))


    def train_selector(self, iteration, step):

        self.discriminator.eval()
        self.selector.train()

        data, inp, label, ans_prob, ans_logits = self.dataloader.sampling_to_train_selector(self.discriminator, iteration)

        assert torch.equal(ans_prob, torch.sigmoid(ans_logits)), "ans_prob and ans_logits are not matched"

        if inp is None:
            self.log_data_w += f"\n\ncannot keep balance (pos:{torch.sum(label==1).item()}, neg:{torch.sum(label==0).item()})\n"
            self.log_data_w += f"iteration-step : {iteration}-{step}\n"
            for d, p in zip(data[:50], ans_prob[:50]):
                self.log_data_w += f"ans : {int(p.item()>=0.5)} ({p.item()}), data : {d}\n"
            return None

        pre_s = self.selector(**inp.to(self.device)).squeeze(-1)
        loss_s = self.loss_func_s(pre_s, torch.tensor(label).to(self.device))

        self.optim_s.zero_grad()
        loss_s.backward()
        self.optim_s.step()

        self.take_log_selector(iteration, step, data, label, ans_prob, pre_s, loss_s.item())

        return loss_s.cpu()
    

    def train_discriminator(self, iteration, step):

        self.discriminator.train()
        self.selector.eval()

        tr_data, tr_inp, tr_label = self.dataloader.sampling_to_train_discriminator(self.selector, iteration, step)

        pred_d = self.discriminator(**tr_inp.to(self.device)).squeeze(-1)
        loss_d = self.loss_func_d(pred_d, torch.tensor(tr_label).to(self.device), iteration)
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
        label, inps = self.dataloader.val_p_label, self.dataloader.val_p_inps

        for model, model_name in zip([self.selector, self.discriminator], ["selector", "discriminator"]):
            with torch.no_grad():
                pre = model(**inps.to(self.device)).squeeze(-1)
            prob = torch.sigmoid(pre)
            pred_label = prob.cpu().numpy() > 0.5
            loss = self.loss_func_ce(pre, torch.tensor(label).to(self.device))

            self.val_patent[model_name]["loss"].append(loss.item())
            self.val_patent[model_name]["acc"].append(accuracy_score(label, pred_label))
            self.val_patent[model_name]["precision"].append(precision_score(label, pred_label, zero_division=0))
            self.val_patent[model_name]["recall"].append(recall_score(label, pred_label, zero_division=0))
            self.val_patent[model_name]["f1"].append(f1_score(label, pred_label, zero_division=0))
            

    def plot_result(self):

        title = f"adv_{self.args.data_type}"

        plot_loss(self.loss_d_list, self.loss_s_train_list, os.path.join(self.result_path, "loss.png"), title, use_moving_avg=True)
        log_loss(self.loss_d_list, self.loss_s_train_list, os.path.join(self.result_path, "loss.csv"))

        self.dataloader.plot_thresholds()
        self.dataloader.plot_gold_ratio()


def main(config):

    gan = SymbolicKDUsingAdversarialNet(result_path, config, args)
    gan.train_gan()



if __name__=="__main__":

    """
    python scripts/train/adv.py --data_type "情報系" --no "1"
    """

    random.seed(42)

    with open("./settings.json", "r") as f:
        config = json.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--no", type=str)
    parser.add_argument("--data_type", type=str)
    args = parser.parse_args()

    if config["adv_train"]["switch"]["flag"]:
        result_path = os.path.join(config["directories"]["step_2"], f"adv/{args.data_type}/switch/{args.no}")
    else:
        result_path = os.path.join(config["directories"]["step_2"], f"adv/{args.data_type}/no_switch/{args.no}")
    os.makedirs(result_path)
    
    with open(os.path.join(result_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    main(config)