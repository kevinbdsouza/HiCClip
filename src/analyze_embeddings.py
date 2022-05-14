import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from utils import get_cumpos
from config import Config
import pandas as pd
import random

def reduce_pca(representations, colors, labels):
    pca_ob = PCA(n_components=3)
    pca_ob.fit(representations)
    pca_rep = pca_ob.transform(representations)
    plot3d(pca_rep, colors, labels)


def simple_plot(hic_win, mode):
    """
    simple_plot(hic_win, mode) -> No return object
    plots heatmaps of reds or differences.
    Args:
        hic_win (Array): Matrix of Hi-C values
        mode (string): one of reds or diff
    """

    if mode == "reds":
        plt.figure()
        sns.set_theme()
        ax = sns.heatmap(hic_win, cmap="Reds", vmin=0, vmax=1)
        ax.set_yticks([])
        ax.set_xticks([])
        # plt.savefig("/home/kevindsouza/Downloads/chr21.png")
        plt.show()

    if mode == "diff":
        plt.figure()
        sns.set_theme()
        rdgn = sns.diverging_palette(h_neg=220, h_pos=14, s=79, l=55, sep=3, as_cmap=True)
        sns.heatmap(hic_win, cmap=rdgn, center=0.00, cbar=True)
        plt.yticks([])
        plt.xticks([])
        # plt.savefig("/home/kevindsouza/Downloads/ctcf_ko.png")
        plt.show()


def plot_smoothness(representations):
    """
    plot_smoothness(representations) -> No return object
    Plot smoothness of representations.
    Args:
        representations (Array): representation matrix
    """

    window = 2000
    nrows = len(representations)
    diff_list = np.arange(-window, window + 1)
    diff_list = np.delete(diff_list, [window])
    diff_vals = np.zeros((nrows, 2 * window))
    for r in range(nrows):
        for i, d in enumerate(diff_list):
            if (r + d) >= 0 and (r + d) <= nrows - 1:
                diff_vals[r, i] = np.linalg.norm(representations[r, :] - representations[r + d, :], ord=1)
            else:
                continue

    diff_reduce = diff_vals.mean(axis=0)
    plt.title("Average L2 Norm of Embeddings with Distance")
    plt.xlabel("Distance in 10 Kbp", fontsize=14)
    plt.ylabel("Average L2 Norm", fontsize=14)
    plt.plot(diff_list, diff_reduce)
    plt.grid(b=None)
    plt.show()


def plot3d(representations, colors, labels):
    """
    plot3d(representations) -> No return object
    Plot first 3 dims of representations.
    Args:
        representations (Array): representation matrix
    """

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(representations[:, 0], representations[:, 1], representations[:, 2], color=colors, label=labels)
    plt.show()


def plot_euclid_heatmap(representations):
    """
    plot_euclid_heatmap(representations) -> No return object
    Plot heatmap of euclidean distance.
    Args:
        representations (Array): representation matrix
    """

    nr = len(representations)
    euclid_heatmap = np.zeros((nr, nr))

    for r1 in range(nr):
        for r2 in range(nr):
            euclid_heatmap[r1, r2] = np.linalg.norm(representations[r1, :] - representations[r2, :])

    plt.figure()
    sns.set_theme()
    ax = sns.heatmap(euclid_heatmap, cmap="Reds", vmin=0)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.show()


if __name__ == "__main__":
    cfg = Config()
    chr = 22
    cum_pos = get_cumpos(cfg, chr)

    embed_rows1 = np.load("/data2/hic_lstm/downstream/predictions/embeddings_temp.npy")
    embed_rows2 = np.load("/data2/hic_lstm/downstream/predictions/embeddings_GM12878.npy")
    main_data = pd.read_csv("/data2/hic_lstm/downstream/predictions/element_data_chr%s.csv" % (chr))
    main_data["pos"] = main_data["pos"] - (cum_pos + 1)
    main_data = main_data[cfg.class_columns + ["pos"]]

    embed_rows1 = embed_rows1[cum_pos + 1:, ]
    embed_rows2 = embed_rows2[cum_pos + 1:, ]
    embed_rows1 = embed_rows1[main_data["pos"]]
    embed_rows2 = embed_rows2[main_data["pos"]]

    main_data = main_data[cfg.class_columns]
    colors = []
    labels = []

    for i in range(len(main_data)):
        sub_df = main_data.loc[i]
        sub_df = sub_df[sub_df != 0]
        rand = int(random.choice(sub_df.index))
        colors.append(cfg.colors_list[rand])
        labels.append(cfg.class_elements_list[rand])

    reduce_pca(embed_rows1, colors, labels)
    reduce_pca(embed_rows2, colors, labels)

    #plot_smoothness(embed_rows1)
    #plot_smoothness(embed_rows2)

    #plot_euclid_heatmap(embed_rows1)
    #plot_euclid_heatmap(embed_rows2)
    print("done")
