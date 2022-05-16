import numpy as np
import pandas as pd
import torch
from torch import nn
from config import Config
from utils import get_cumpos


class BatchFasta():
    """
    Class to BatchFasta
    """

    def __init__(self, cfg, chr):
        self.chr = chr
        self.cfg = cfg

    def load_fasta(self):
        pass

    def batch_fasta(self):
        pass


class BatchHiCLSTMEmbeddings():
    """
    Class to Batch HiCLSTM Embeddings for Clip
    """

    def __init__(self, cfg, chr):
        self.chr = chr
        self.cfg = cfg
        self.cumpos = get_cumpos(cfg, chr)
        if chr == 22:
            self.cumpos_next = cfg.genome_len
        else:
            self.cumpos_next = get_cumpos(cfg, chr + 1)

    def load_embeddings(self):
        embed_rows = np.load(self.cfg.embeddings_path)
        return embed_rows

    def batch_embeddings(self, batch_size):
        embed_rows = self.load_embeddings()
        embed_rows = embed_rows[self.cumpos + 1:self.cumpos_next]

        seq_len = self.cfg.text_seq_len
        fill_length = seq_len - (len(embed_rows) % seq_len)
        fill = np.zeros((fill_length, 16))
        embed_rows = np.vstack((embed_rows, fill))
        num_seqs = int(len(embed_rows) / seq_len)

        embed_input = []
        batch_embed_input = []
        for r in range(num_seqs):
            for c in range(num_seqs):
                r_embeds = embed_rows[r * seq_len: (r + 1) * seq_len, :]
                c_embeds = embed_rows[c * seq_len: (c + 1) * seq_len, :]

                embeds = np.concatenate((r_embeds, c_embeds), axis=0)
                embed_input.append(embeds)

                if len(embed_input) == batch_size:
                    batch_embed_input.append(embed_input)
                    embed_input = []

        return batch_embed_input


if __name__ == "__main__":
    cfg = Config()
    chr = 21
    batch_embed_ob = BatchHiCLSTMEmbeddings(cfg, chr)
    batch_embed_ob.batch_embeddings(500)
