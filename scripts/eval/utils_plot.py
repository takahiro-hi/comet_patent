import os
import matplotlib.pyplot as plt
import numpy as np


def plot_heat_map(matric, save_file):

    plt.figure(figsize=(12, 6))
    plt.imshow(matric, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title("bleu similarity")
    plt.xlabel("Gold Data")
    plt.ylabel("Silver Data")
    plt.savefig(save_file)
    plt.close()


def plot_metrics(data, title, file_path):

    metrics = list(data.keys())
    models = ['Base', 'Adv (patent)', 'Adv (atomic_patent)']
    means = {metric: [np.mean(data[metric][model]) for model in models] for metric in metrics}
    stds = {metric: [np.std(data[metric][model]) for model in models] for metric in metrics}

    fig, ax = plt.subplots(figsize=(10, 8))

    x = np.arange(len(metrics))
    width = 0.2
    bar_positions = {model: x + i * width for i, model in enumerate(models)}

    for model in models:
        ax.bar(
            bar_positions[model],
            [means[metric][models.index(model)] for metric in metrics],
            width,
            label=model,
            yerr=[stds[metric][models.index(model)] for metric in metrics],
            capsize=5
        )

    ax.set_xlabel('Metric', fontsize=15)
    ax.set_ylabel('Score', fontsize=15)
    ax.set_title(f'Performance Comparison\n{title}', fontsize=15)
    ax.set_xticks(x + (width * (len(models) - 1)) / 2)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=15)
    ax.grid()   
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
