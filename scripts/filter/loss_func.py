import torch, os
import matplotlib.pyplot as plt
import torch.nn as nn



class LossFuncForDiscriminator(torch.nn.Module):

    def __init__(self, result_path, tr_params):

        super(LossFuncForDiscriminator, self).__init__()

        self.cross_entropy = nn.BCEWithLogitsLoss()
        self.lambda_hinge_list = self.get_lambda_hinge_list(tr_params)

        self.plot_hinge_weight(result_path)
        self.plot_loss(result_path, tr_params)


    def forward(self, logit, ans, iteration):

        lambda_hinge = self.lambda_hinge_list[iteration-1]

        return self.cross_entropy(logit, ans) + lambda_hinge * self.hinge_loss(logit, ans)
        

    def get_lambda_hinge_list(self, tr_params):

        start_iteration = tr_params["discriminator"]["hinge_loss"]["start_iters"]
        end_iteration = tr_params["discriminator"]["hinge_loss"]["end_iters"]
        max_hinge_weight = tr_params["discriminator"]["hinge_loss"]["max_weight"]

        ret_list = []
        for i in range(tr_params["iterations"]):
            if i < start_iteration:
                ret_list.append(0)
            elif start_iteration <= i < end_iteration:
                ret_list.append(max_hinge_weight * (i - start_iteration) / (end_iteration - start_iteration))
            else:
                ret_list.append(max_hinge_weight)
                
        return ret_list


    def hinge_loss(self, logit, ans):
        
        pos_mask = ans == 1
        hinge_loss_pos = torch.where(pos_mask, torch.max(torch.zeros_like(logit), 1 - logit), torch.zeros_like(logit))
        hinge_loss_pos = hinge_loss_pos.sum() / pos_mask.sum() if pos_mask.sum() > 0 else 0
        
        neg_mask = ans == 0
        adjusted_y_pred = torch.min(logit, torch.full_like(logit, 0))
        hinge_loss_neg = torch.where(neg_mask, torch.max(torch.zeros_like(logit), 1 + adjusted_y_pred), torch.zeros_like(logit))
        hinge_loss_neg = hinge_loss_neg.sum() / neg_mask.sum() if neg_mask.sum() > 0 else 0
        
        return hinge_loss_pos + hinge_loss_neg
    

    def plot_hinge_weight(self, result_path):

        plt.plot([i for i in range(len(self.lambda_hinge_list))], self.lambda_hinge_list)
        plt.xlabel("Iteration")
        plt.ylabel("$\lambda_{hinge}$")
        plt.title("Weight of the hinge loss")
        plt.grid()
        plt.savefig(os.path.join(result_path, "lambda_hinge.png"))
        plt.close()


    def plot_loss(self, result_path, tr_params):

        logit_list = [i*0.01 for i in range(-299, 299, 1)]
        logit_list = torch.tensor(logit_list, requires_grad=False).unsqueeze(-1)

        loss_list_total, loss_list_ce, loss_list_hinge = [], [], []

        for ans in [1., 0.]:
            _temp_total, _temp_ce, _temp_hinge = [], [], []
            for logit in logit_list:
                _temp_total.append(self.forward(logit.unsqueeze(-1), torch.tensor([ans]).unsqueeze(-1), tr_params["iterations"]))
                _temp_ce.append(self.cross_entropy(logit.unsqueeze(-1), torch.tensor([ans]).unsqueeze(-1)))
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
        plt.title("$loss_{total}$ = " + f"1.0 * " + "$loss_{ce}$ + " + f"{str(tr_params["discriminator"]["hinge_loss"]["max_weight"])} * " + "$loss_{hinge}$")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(result_path, "loss_func.png"))
        plt.close()

    