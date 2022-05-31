import numpy as np
import pandas as pd
from config import Config
from utils import get_cumpos
from Bio import SeqIO
import os
import multiprocessing as mp

os.chdir("/home/kevindsouza/Documents/projects/PhD/HiCFold/src")


class BatchFasta():
    """
    Class to BatchFasta
    """

    def __init__(self, cfg, chr):
        self.chr = chr
        self.cfg = cfg
        self.fasta_full_path = cfg.fasta_path + cfg.fasta_file

    def load_fasta(self):
        for seq_record in SeqIO.parse(self.fasta_full_path, "fasta"):
            print(seq_record.id)
            print(repr(seq_record.seq))
            print(len(seq_record))

    def batch_fasta(self):
        self.load_fasta()
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
        embed_rows = embed_rows[self.cumpos:self.cumpos_next]

        seq_len = int(self.cfg.clip_config["text_seq_len"] / 2)
        fill_length = seq_len - (len(embed_rows) % seq_len)
        fill = np.zeros((fill_length, 16))
        embed_rows = np.vstack((embed_rows, fill))
        num_seqs = int(len(embed_rows) / seq_len)

        embed_input = []
        batched_embed = []
        for r in range(num_seqs):
            for c in range(num_seqs):
                r_embeds = embed_rows[r * seq_len: (r + 1) * seq_len, :]
                c_embeds = embed_rows[c * seq_len: (c + 1) * seq_len, :]

                embeds = np.concatenate((r_embeds, c_embeds), axis=0)
                embed_input.append(embeds)

                if (len(embed_input) == batch_size) or (r == num_seqs - 1 and c == num_seqs - 1):
                    batched_embed.append(embed_input)
                    embed_input = []

        return batched_embed


class BatchIndices():
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
        self.seq_len = int(self.cfg.clip_config["text_seq_len"] / 2)

    def modify_reps(self, chr_done, rep_chrs):
        if len(chr_done) >= 17:
            rep_chrs = 64
        elif len(chr_done) >= 16:
            rep_chrs = 32
        elif len(chr_done) >= 14:
            rep_chrs = 16
        elif len(chr_done) >= 10:
            rep_chrs = 8
        elif len(chr_done) >= 8:
            rep_chrs = 6
        elif len(chr_done) >= 2:
            rep_chrs = 4
        else:
            pass

        return rep_chrs

    def batch_chromo_init(self, chr):
        self.chr = chr
        self.cumpos = get_cumpos(self.cfg, chr)
        if chr == 22:
            self.cumpos_next = cfg.genome_len
        else:
            self.cumpos_next = get_cumpos(cfg, chr + 1)
        indices = np.arange(self.cumpos + 1, self.cumpos_next + 1)
        fill_length = self.seq_len - (len(indices) % self.seq_len)
        fill = np.zeros((fill_length,))
        indices = np.concatenate((indices, fill))
        return indices

    def get_partial_input(self, chr, indices_input, r_prev, c_prev, chr_done, rep_chrs, return_dict):
        indices = self.batch_chromo_init(chr)
        num_seqs = int(len(indices) / self.seq_len)

        for i in range(0, rep_chrs):
            if chr in chr_done:
                continue
            r_indices = indices[r_prev[chr - 1] * self.seq_len: (r_prev[chr - 1] + 1) * self.seq_len]
            c_indices = indices[c_prev[chr - 1] * self.seq_len: (c_prev[chr - 1] + 1) * self.seq_len]

            c_prev[chr - 1] += 1
            if c_prev[chr - 1] == num_seqs:
                c_prev[chr - 1] = 0
                r_prev[chr - 1] += 1

                if r_prev[chr - 1] == num_seqs:
                    chr_done.append(chr)
                    print("chr %s done" % chr)

            temp_indices = np.concatenate((r_indices, c_indices), axis=0)
            indices_input.append(temp_indices)

        return_dict["indices_input"] = indices_input
        return_dict["r_prev"] = r_prev
        return_dict["c_prev"] = c_prev
        return_dict["chr_done"] = chr_done

    def batch_chromosome_wise(self):
        num_chrs = 22
        rep_chrs = 4
        batch_num = 0

        chr_done = [21, 22, 19, 20, 18, 17]
        # r_prev = np.zeros((num_chrs)).astype(int)
        # c_prev = np.zeros(num_chrs).astype(int)
        r_prev = np.load(cfg.cross_chromosome_batches + "r_prev.npy")
        c_prev = np.load(cfg.cross_chromosome_batches + "c_prev.npy")

        while len(chr_done) < 18:
            batch_num += 1
            indices_input = []
            rep_chrs = self.modify_reps(chr_done, rep_chrs)
            if batch_num <= 2254:
                continue
            for chr in range(5, num_chrs + 1):
                if chr in chr_done:
                    continue

                manager = mp.Manager()
                return_dict = manager.dict()
                proc = mp.Process(target=self.get_partial_input,
                                  args=(chr, indices_input, r_prev, c_prev, chr_done, rep_chrs, return_dict))
                proc.start()
                proc.join()

                indices_input = return_dict["indices_input"]
                r_prev = return_dict["r_prev"]
                c_prev = return_dict["c_prev"]
                chr_done = return_dict["chr_done"]

            np.save(cfg.cross_chromosome_batches + "cross_chr_ind_%s.npy" % batch_num, indices_input)

    def batch_indices(self, batch_size):
        self.seq_len = int(self.cfg.clip_config["text_seq_len"] / 2)
        indices = np.arange(self.cumpos + 1, self.cumpos_next + 1)
        fill_length = self.seq_len - (len(indices) % self.seq_len)

        fill = np.zeros((fill_length,))
        indices = np.concatenate((indices, fill))
        num_seqs = int(len(indices) / self.seq_len)

        indices_input = []
        batched_indices = []
        for r in range(num_seqs):
            for c in range(num_seqs):
                r_indices = indices[r * self.seq_len: (r + 1) * self.seq_len]
                c_indices = indices[c * self.seq_len: (c + 1) * self.seq_len]

                temp_indices = np.concatenate((r_indices, c_indices), axis=0)
                indices_input.append(temp_indices)

                if (len(indices_input) == batch_size) or (r == num_seqs - 1 and c == num_seqs - 1):
                    batched_indices.append(indices_input)
                    indices_input = []

        return batched_indices


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
        self.seq_len = int(self.cfg.clip_config["text_seq_len"] / 2)
        self.fill_length = self.seq_len - ((self.cumpos_next - self.cumpos) % self.seq_len)
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
        num_seqs = int(self.hic_size / self.seq_len)

        hic_input = []
        batched_hic = []
        for r in range(num_seqs):
            for c in range(num_seqs):
                hic_window = hic_mat[r * self.seq_len: (r + 1) * self.seq_len, c * self.seq_len: (c + 1) * self.seq_len]
                hic_input.append(hic_window)

                if (len(hic_input) == batch_size) or (r == num_seqs - 1 and c == num_seqs - 1):
                    batched_hic.append(hic_input)
                    hic_input = []

        return batched_hic

    def batch_chromo_init(self, chr):
        self.chr = chr
        self.cumpos = get_cumpos(self.cfg, chr)
        if chr == 22:
            self.cumpos_next = cfg.genome_len
        else:
            self.cumpos_next = get_cumpos(cfg, chr + 1)
        self.fill_length = self.seq_len - ((self.cumpos_next - self.cumpos) % self.seq_len)
        self.hic_size = (self.cumpos_next - self.cumpos) + self.fill_length
        hic_mat = self.load_hic()
        return hic_mat

    def modify_reps(self, chr_done, rep_chrs):
        if len(chr_done) >= 17:
            rep_chrs = 64
        elif len(chr_done) >= 16:
            rep_chrs = 32
        elif len(chr_done) >= 14:
            rep_chrs = 16
        elif len(chr_done) >= 10:
            rep_chrs = 8
        elif len(chr_done) >= 8:
            rep_chrs = 6
        elif len(chr_done) >= 2:
            rep_chrs = 4
        else:
            pass

        return rep_chrs

    def get_partial_input(self, chr, hic_input, r_prev, c_prev, chr_done, rep_chrs, return_dict):
        hic_mat = self.batch_chromo_init(chr)
        num_seqs = int(self.hic_size / self.seq_len)

        for i in range(0, rep_chrs):
            if chr in chr_done:
                continue
            hic_window = hic_mat[r_prev[chr - 1] * self.seq_len: (r_prev[chr - 1] + 1) * self.seq_len,
                         c_prev[chr - 1] * self.seq_len: (c_prev[chr - 1] + 1) * self.seq_len]
            c_prev[chr - 1] += 1
            if c_prev[chr - 1] == num_seqs:
                c_prev[chr - 1] = 0
                r_prev[chr - 1] += 1

                if r_prev[chr - 1] == num_seqs:
                    chr_done.append(chr)
                    print("chr %s done" % chr)
            hic_input.append(hic_window)

        return_dict["hic_input"] = hic_input
        return_dict["r_prev"] = r_prev
        return_dict["c_prev"] = c_prev
        return_dict["chr_done"] = chr_done

    def batch_chromosome_wise(self):
        num_chrs = 22
        rep_chrs = 4
        batch_num = 0

        chr_done = [21, 22, 19, 20, 18, 17]
        #r_prev = np.zeros((num_chrs)).astype(int)
        #c_prev = np.zeros(num_chrs).astype(int)
        r_prev = np.load(cfg.cross_chromosome_batches + "r_prev.npy")
        c_prev = np.load(cfg.cross_chromosome_batches + "c_prev.npy")

        while len(chr_done) < 18:
            batch_num += 1
            hic_input = []
            rep_chrs = self.modify_reps(chr_done, rep_chrs)
            if batch_num <= 2254:
                continue
            for chr in range(5, num_chrs + 1):
                if chr in chr_done:
                    continue

                manager = mp.Manager()
                return_dict = manager.dict()
                proc = mp.Process(target=self.get_partial_input,
                                  args=(chr, hic_input, r_prev, c_prev, chr_done, rep_chrs, return_dict))
                proc.start()
                proc.join()

                hic_input = return_dict["hic_input"]
                r_prev = return_dict["r_prev"]
                c_prev = return_dict["c_prev"]
                chr_done = return_dict["chr_done"]

            np.save(cfg.cross_chromosome_batches + "cross_chr_hic_%s.npy" % batch_num, hic_input)


if __name__ == "__main__":
    cfg = Config()

    if cfg.batch_chromosome_wise:
        batch_hic_ob = BatchHiCMaps(cfg, 1)
        batch_hic_ob.batch_chromosome_wise()

        # batch_ind_ob = BatchIndices(cfg, 1)
        # batch_ind_ob.batch_chromosome_wise()
    else:
        for chr in cfg.chr_train_list:
            """
            batch_embed_ob = BatchHiCLSTMEmbeddings(cfg, chr)
            batched_embed = batch_embed_ob.batch_embeddings(cfg.clip_batch_size)
            np.save(cfg.batched_hic_path + "embed_%s.npy" % chr, batched_embed)
    
            batch_hic_ob = BatchHiCMaps(cfg, chr)
            batched_hic = batch_hic_ob.batch_hic_maps(cfg.clip_batch_size)
            np.save(cfg.batched_hic_path + "hic_%s.npy" % chr, batched_hic)
            """

            """
            batch_ind_ob = BatchIndices(cfg, chr)
            batched_indices = batch_ind_ob.batch_indices(cfg.clip_batch_size)
            np.save(cfg.batched_hic_path + "indices_%s.npy" % chr, batched_indices)
            """

            """
            batch_fasta_ob = BatchFasta(cfg, chr)
            batch_fasta_ob.batch_fasta()
            """

    print("done")
