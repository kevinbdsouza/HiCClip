import numpy as np
import pandas as pd
import torch
from torch import nn
from config import Config
from utils import get_cumpos
from utils import simple_plot


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

                if (len(embed_input) == batch_size) or (r == num_seqs - 1 and c == num_seqs - 1):
                    batch_embed_input.append(embed_input)
                    embed_input = []

        return batch_embed_input


class BatchHiCMaps():
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
        self.fill_length = self.cfg.text_seq_len - ((self.cumpos_next - self.cumpos) % self.cfg.text_seq_len)
        self.hic_size = (self.cumpos_next - self.cumpos) + self.fill_length

    def contactProbabilities(self, values, smoothing=8, delta=1e-10):
        coeff = np.nan_to_num(1 / (values + delta))
        contact_prob = np.power(1 / np.exp(smoothing), coeff)
        return contact_prob

    def load_hic(self):
        try:
            data = pd.read_csv("%s%s/%s/hic_chr%s.txt" % (self.cfg.hic_path, self.cfg.cell, self.chr, self.chr),
                               sep="\t",
                               names=['i', 'j', 'v'])
            data = data.dropna()
            data[['i', 'j']] = data[['i', 'j']] / cfg.resolution
            data['v'] = self.contactProbabilities(data['v'], smoothing=cfg.hic_smoothing)
            rows = np.array(data["i"]).astype(int)
            cols = np.array(data["j"]).astype(int)

            hic_mat = np.zeros((self.hic_size, self.hic_size))
            hic_mat[rows, cols] = np.array(data["v"])
            hic_mat[cols, rows] = np.array(data["v"])

            return hic_mat
        except Exception as e:
            print("Hi-C txt file does not exist or error during Juicer extraction")

    def batch_hic_maps(self, batch_size):
        hic_mat = self.load_hic()
        seq_len = self.cfg.text_seq_len
        num_seqs = int(self.hic_size / seq_len)

        hic_input = []
        batched_hic = []
        for r in range(num_seqs):
            for c in range(num_seqs):
                hic_window = hic_mat[r * seq_len: (r + 1) * seq_len, c * seq_len: (c + 1) * seq_len]
                hic_input.append(hic_window)

                if (len(hic_input) == batch_size) or (r == num_seqs - 1 and c == num_seqs - 1):
                    batched_hic.append(hic_input)
                    hic_input = []

        return batched_hic


if __name__ == "__main__":
    cfg = Config()
    chr = 1
    batch_embed_ob = BatchHiCLSTMEmbeddings(cfg, chr)
    batch_embed_input = batch_embed_ob.batch_embeddings(200)

    batch_hic_ob = BatchHiCMaps(cfg, chr)
    batched_hic = batch_hic_ob.batch_hic_maps(200)
