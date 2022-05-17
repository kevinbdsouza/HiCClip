import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from utils import get_cumpos
from config import Config
import pandas as pd
import random
from matplotlib.colors import ListedColormap
import umap
from utils import simple_plot


def reduce_umap(representations, colors, cfg):
    n_components = 2
    umap_rep = umap.UMAP().fit_transform(representations)
    plot2d(umap_rep, colors, cfg)
    pass


def reduce_pca(representations, colors, cfg):
    n_components = 2
    pca_ob = PCA(n_components=n_components)
    pca_ob.fit(representations)
    pca_rep = pca_ob.transform(representations)
    if n_components == 3:
        plot3d(pca_rep, colors, cfg)
    else:
        plot2d(pca_rep, colors, cfg)


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


def plot2d(representations, color_index, cfg):
    """
    plot3d(representations) -> No return object
    Plot first 3 dims of representations.
    Args:
        representations (Array): representation matrix
    """

    plt.figure()
    color_map = ListedColormap(cfg.colors_list)

    scatter = plt.scatter(representations[:, 0], representations[:, 1], c=color_index, cmap=color_map)

    plt.legend(handles=scatter.legend_elements()[0], labels=cfg.class_elements_list)
    plt.tight_layout()
    plt.show()


def plot3d(representations, color_index, cfg):
    """
    plot3d(representations) -> No return object
    Plot first 3 dims of representations.
    Args:
        representations (Array): representation matrix
    """

    plt.figure()
    ax = plt.axes(projection='3d')
    color_map = ListedColormap(cfg.colors_list)

    scatter = ax.scatter3D(representations[:, 0], representations[:, 1], representations[:, 2], c=color_index,
                           cmap=color_map)

    plt.legend(handles=scatter.legend_elements()[0], labels=cfg.class_elements_list)
    plt.tight_layout()
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


def plot_embed_rows():
    embed_rows = np.load("/data2/hic_lstm/downstream/predictions/embeddings_temp.npy")
    main_data = pd.read_csv("/data2/hic_lstm/downstream/predictions/element_data_chr%s.csv" % (chr))
    main_data["pos"] = main_data["pos"] - (cum_pos + 1)
    main_data = main_data[cfg.class_columns + ["pos"]]

    embed_rows = embed_rows[cum_pos + 1:, ]
    embed_rows = embed_rows[main_data["pos"]]
    main_data = main_data[cfg.class_columns]

    colors = []

    for i in range(len(main_data)):
        sub_df = main_data.loc[i]
        sub_df = sub_df[sub_df != 0]
        rand = int(random.choice(sub_df.index))
        colors.append(rand)

    # reduce_pca(embed_rows, colors, cfg)
    reduce_umap(embed_rows, colors, cfg)
    # plot_smoothness(embed_rows1)
    # plot_euclid_heatmap(embed_rows2)


if __name__ == "__main__":
    cfg = Config()
    chr = 21
    cum_pos = get_cumpos(cfg, chr)

    plot_embed_rows()
    print("done")
