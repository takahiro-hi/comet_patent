import configparser, torch, os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


class LossFuncForDiscriminator(torch.nn.Module):

    def __init__(self, result_path, config, flag_linear=False):

        super(LossFuncForDiscriminator, self).__init__()

        self.result_path = result_path
        self.iterations = config["adv_train"]["iterations"]
        self.config_d_hinge= config["adv_train"]["discriminator"]["hinge_weight"]
        self.flag_linear = flag_linear

        self.max_hinge_weight = self.config_d_hinge["max_weight"]

        self.cross_entropy = nn.BCEWithLogitsLoss()

        self.lambda_ce = 1
        self.lambda_hinge_list = self.get_lambda_hinge_list()

        self.plot_threshold()
        self.plot_loss()


    def forward(self, logit, ans, iteration):

        lambda_hinge = self.lambda_hinge_list[iteration-1]
        lambda_ce = self.lambda_ce if self.flag_linear else 1 - lambda_hinge

        return lambda_ce * self.ce_loss(logit, ans) + lambda_hinge * self.hinge_loss(logit, ans)
        

    def get_lambda_hinge_list(self):

        start_iteration = self.config_d_hinge["start_iters"]
        end_iteration = self.config_d_hinge["end_iters"]

        ret_list = []
        for i in range(self.iterations):
            if i < start_iteration:
                ret_list.append(0)
            elif start_iteration <= i < end_iteration:
                ret_list.append(self.max_hinge_weight * (i - start_iteration) / (end_iteration - start_iteration))
            else:
                ret_list.append(self.max_hinge_weight)
                
        return ret_list


    def ce_loss(self, logit, ans):

        return self.cross_entropy(logit, ans)


    def hinge_loss(self, logit, ans):
        
        pos_mask = ans == 1
        hinge_loss_pos = torch.where(pos_mask, torch.max(torch.zeros_like(logit), 1 - logit), torch.zeros_like(logit))
        hinge_loss_pos = hinge_loss_pos.sum() / pos_mask.sum() if pos_mask.sum() > 0 else 0
        
        neg_mask = ans == 0
        adjusted_y_pred = torch.min(logit, torch.full_like(logit, 0))
        hinge_loss_neg = torch.where(neg_mask, torch.max(torch.zeros_like(logit), 1 + adjusted_y_pred), torch.zeros_like(logit))
        hinge_loss_neg = hinge_loss_neg.sum() / neg_mask.sum() if neg_mask.sum() > 0 else 0
        
        return hinge_loss_pos + hinge_loss_neg
    

    def plot_threshold(self):

        if self.flag_linear:
            plt.plot([i for i in range(len(self.lambda_hinge_list))], self.lambda_hinge_list, label="$\lambda_{hinge}$")
            plt.plot([i for i in range(len(self.lambda_hinge_list))], [1 - i for i in self.lambda_hinge_list], label="$\lambda_{ce}$")
            plt.xlabel("Iteration")
            plt.ylabel("Weight")
            plt.title("Weight of the hinge loss and cross entropy")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(self.result_path, "lambda_weight.png"))
            plt.close()

        else:
            plt.plot([i for i in range(len(self.lambda_hinge_list))], self.lambda_hinge_list)
            plt.xlabel("Iteration")
            plt.ylabel("$\lambda_{hinge}$")
            plt.title("Weight of the hinge loss")
            plt.grid()
            plt.savefig(os.path.join(self.result_path, "lambda_hinge.png"))
            plt.close()


    def plot_loss(self):

        logit_list = [i*0.01 for i in range(-299, 299, 1)]
        logit_list = torch.tensor(logit_list, requires_grad=False).unsqueeze(-1)

        loss_list_total, loss_list_ce, loss_list_hinge = [], [], []

        for ans in [1., 0.]:
            _temp_total, _temp_ce, _temp_hinge = [], [], []
            for logit in logit_list:
                _temp_total.append(self.forward(logit.unsqueeze(-1), torch.tensor([ans]).unsqueeze(-1), self.iterations))
                _temp_ce.append(self.ce_loss(logit.unsqueeze(-1), torch.tensor([ans]).unsqueeze(-1)))
                _temp_hinge.append(self.hinge_loss(logit.unsqueeze(-1), torch.tensor([ans]).unsqueeze(-1)))

            loss_list_total.append(_temp_total)
            loss_list_ce.append(_temp_ce)
            loss_list_hinge.append(_temp_hinge)

        color_list = ["red", "blue"]

        for idx, ans in enumerate([1., 0.]):
            plt.plot(logit_list, loss_list_total[idx], label=f"(label={int(ans)}) total", linestyle="solid", color=color_list[idx])
            plt.plot(logit_list, loss_list_ce[idx], label=f"(label={int(ans)}) cross entropy", linestyle="dashed", color=color_list[idx])
            plt.plot(logit_list, loss_list_hinge[idx], label=f"(label={int(ans)}) hinge", linestyle="dotted", color=color_list[idx])

        plt.xlabel("Prediction (logit)")
        plt.ylabel("Loss")
        plt.title("$loss_{total}$ = " + f"{self.lambda_ce} * " + "$loss_{ce}$ + " + f"{str(self.max_hinge_weight)} * " + "$loss_{hinge}$")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.result_path, "loss_func.png"))
        plt.close()

    

