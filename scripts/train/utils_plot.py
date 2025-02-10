import japanize_matplotlib, os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import roc_curve, roc_auc_score



def plot_patent_metrics(val_patent, model_type, save_file_path):

    iterations = val_patent["iteration"]
    metrics = val_patent[model_type]
    
    plt.figure(figsize=(10, 6))
    
    for metric_name, values in metrics.items():
        plt.plot(iterations, values, label=f"{model_type} {metric_name}")
    
    plt.xlabel("Iteration")
    plt.ylabel("Metric Value")
    plt.title(f"Metrics for {model_type}")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_file_path)


def plot_roc_curve(prob_list, ans_list, base_prob, file_path):

    fpr, tpr, thresholds = roc_curve(ans_list, prob_list)
    roc_auc = roc_auc_score(ans_list, prob_list)

    fpr_base, tpr_base, thresholds_base = roc_curve(ans_list, base_prob)
    roc_auc_base = roc_auc_score(ans_list, base_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve of Adv (auc={roc_auc:.3f})")
    plt.plot(fpr_base, tpr_base, label=f"ROC curve of Base (auc={roc_auc_base:.3f})", linestyle="--")
    plt.title(f"ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.legend()
    plt.savefig(file_path)
    plt.close()



def plot_loss(loss_d, loss_s_train, file_path, title, use_moving_avg=False, window_size=3):

    assert len(loss_d) == len(loss_s_train)

    fig_loss = plt.figure()
    left = np.array([i for i in range(len(loss_d))])
    plt.xlabel("iteration", fontsize=13)
    plt.ylabel("loss", fontsize=13)

    if use_moving_avg:
        title = "Loss\n" + title + f"\n(window size: {window_size})"
    else:
        title = "Loss\n" + title
    plt.title(title, fontsize=10)

    if use_moving_avg:
        loss_d = np.convolve(loss_d, np.ones(window_size)/window_size, mode='valid')
        loss_s_train = np.convolve(loss_s_train, np.ones(window_size)/window_size, mode='valid')
        left = left[:len(loss_d)]

    p2 = plt.plot(left, np.array(loss_d))
    p1 = plt.plot(left, np.array(loss_s_train))
    plt.legend((p1[0],p2[0]), ("selector_train", "discriminator"), loc="best")
    plt.grid(True)
    plt.savefig(file_path)
    plt.close(fig_loss)


def log_loss(loss_d, loss_s_train, file_path):

    with open(file_path, "w") as f:
        f.write("iteration\tdiscriminator\tselector_tr\n")
        for idx, loss in enumerate(zip(loss_d, loss_s_train)):
            f.write(f"{idx}\t{loss[0]}\t{loss[1]}a\n")


def plot_metrics_val(results_eval, metrics, file_path_plot, file_name, title, use_moving_avg=False, window_size=3):

    result = {t: pd.DataFrame(data) for t, data in results_eval.items()}
    for threshold, df in result.items():
        plt.figure()

        for metric in metrics:
            if use_moving_avg:
                smoothed_values = np.convolve(df[metric], np.ones(window_size)/window_size, mode='valid')
                plt.plot(smoothed_values, label=metric)
            else:
                plt.plot(df[metric], label=metric)

        if use_moving_avg:
            _title = f"Evaluation Metrics\n(Threshold: {threshold}, window size: {window_size})\n" + title
        else:
            _title = f"Evaluation Metrics\n(Threshold: {threshold})\n" + title

        plt.title(_title, fontsize=10)
        plt.xlabel("Iteration")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(file_path_plot, f"{file_name}_threshold_{threshold}.png"))
        plt.close()


def log_metric_val(results_eval, file_path_log, file_name):

    result = {t: pd.DataFrame(data) for t, data in results_eval.items()}
    for threshold, df in result.items():
        df.to_csv(os.path.join(file_path_log, f"{file_name}_threshold_{threshold}.csv"), index=False)








"""def plot_thresholds(thresholds_list, result_path):

    plt.figure()
    plt.plot(thresholds_list)
    plt.xlabel("iteration", fontsize=13)
    plt.ylabel("threshold", fontsize=13)
    plt.grid(True)
    plt.savefig(result_path)
    plt.close()"""
    

"""def plot_similarity(matrix, save_path, title):

    # plot heatmap
    plt.figure()
    plt.imshow(matrix, cmap="hot")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Gold Data")
    plt.ylabel("Silver Data")
    plt.savefig(save_path+"_heatmap.png")
    plt.close()

    # plot histogram
    max_similarities = np.max(matrix, axis=1)
    plt.hist(max_similarities, bins=20)
    plt.title(title)
    plt.xlabel("Maximum Similarity")
    plt.ylabel("Frequency")
    plt.savefig(save_path+"_hist.png")
    plt.close()"""



if __name__=="__main__":

    pass
