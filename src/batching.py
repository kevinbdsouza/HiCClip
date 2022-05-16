import numpy as np
import pandas as pd
import torch
from torch import nn
from config import Config


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

    def __init__(self, cfg):
        self.cfg = cfg

    def load_embeddings(self):
        embed_rows = np.load(self.cfg.embeddings_path)
        return embed_rows

    def batch_embeddings(self, batch_size):
        embed_rows = self.load_embeddings()
        seq_len = self.cfg.text_seq_len
        embed_input = []
        num_seqs = int(np.ceil(len(embed_rows) / self.cfg.text_seq_len))

        for r in range(num_seqs):
            for c in range(num_seqs):
                if r == num_seqs - 1:
                    r_embeds = embed_rows[r * seq_len:, :]
                else:
                    r_embeds = embed_rows[r * seq_len: (r + 1) * seq_len, :]
                if r == num_seqs - 1:
                    c_embeds = embed_rows[c * seq_len:, :]
                else:
                    c_embeds = embed_rows[c * seq_len: (c + 1) * seq_len, :]

                embeds = np.stack((r_embeds, c_embeds), axis=-1)
                embed_input.append(embeds)

        pass


if __name__ == "__main__":
    cfg = Config()
    batch_embed_ob = BatchHiCLSTMEmbeddings(cfg)
    batch_embed_ob.batch_embeddings(200)
