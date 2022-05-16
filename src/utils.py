import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_cumpos(cfg, chr_num):
    """
    get_cumpos(cfg, chr_num) -> int
    Returns cumulative index upto the end of the previous chromosome.
    Args:
        cfg (Config): the configuration to use for the experiment.
        chr_num (int): the current chromosome.
    """
    sizes = np.load(cfg.hic_path + cfg.sizes_file, allow_pickle=True).item()
    if chr_num == 1:
        cum_pos = 0
    else:
        key = "chr" + str(chr_num - 1)
        cum_pos = sizes[key]

    return cum_pos


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
