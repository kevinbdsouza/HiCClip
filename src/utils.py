import numpy as np


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
