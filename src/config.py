import os
import pathlib
import random


class Config:
    def __init__(self):
        """
        Includes Data Parameters, Model Parameters, Hyperparameters, Input Directories
        File Names, Model Names, Output Directories
        """

        "Data Parameters"
        self.num_chr = 23
        self.genome_len = 288091
        self.resolution = 10000
        self.cell = "GM12878"
        self.chr_train_list = list(range(1, 23))
        self.chr_test_list = list(range(22, 23))
        self.chr_train_list_shuff = list(range(4, 23))
        #random.shuffle(self.chr_train_list_shuff)
        self.save_processed_data = False

        "fasta"
        self.data_path = "/data2/hicfold/"
        self.downstream_dir = self.data_path + "downstream/"
        self.fasta_batch_size = 1000
        self.fasta_path = self.data_path + "fasta/ncbi-genomes-2022-05-13/"
        self.fasta_file = "GCF_000001405.40_GRCh38.p14_genomic.fna"
        self.batched_hic_path = self.data_path + "batched_hic_hg19/"
        self.cross_chromosome_batches = self.data_path + "cross_chromosome_batches/"
        self.batch_chromosome_wise = True

        "Model Paramters"
        self.pos_embed_size = 16
        self.input_size_lstm = 2 * self.pos_embed_size
        self.hidden_size_lstm = 8
        self.output_size_lstm = 1
        self.sequence_length = 150
        self.method = "hicfold"

        "Hyperparameters"
        self.learning_rate = 0.01
        self.num_epochs = 1
        self.batch_size = 210
        self.max_norm = 10
        self.hic_smoothing = 8

        "Input Directories and file names"
        self.hic_path = '/data2/hic_lstm/data/'
        self.sizes_file = 'chr_cum_sizes2.npy'
        self.start_end_file = 'starts.npy'
        self.model_name = "hicfold_" + self.cell

        "decoder parameters"

        "Output Directories"
        self.proj_dir = "/home/kevindsouza/Documents/projects/PhD/HiCFold/"
        self.model_dir = self.proj_dir + 'models/'
        self.output_directory = self.downstream_dir + "/predictions/"
        self.embeddings_path = "/data2/hic_lstm/downstream/predictions/embeddings_temp.npy"
        self.processed_data_dir = "/data2/hic_lstm/downstream/predictions/processed_data/" + self.cell + "/"

        "create directories if they don't exist"
        for file_path in [self.model_dir, self.output_directory, self.processed_data_dir]:
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                print("Creating directory %s" % file_path)
                pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
            else:
                print("Directory %s exists" % file_path)

        "classification"
        self.class_elements_list = ["Gene Expression", "Replication Timing", "Enhancers", "TSS",
                                    "PE-Interactions", "FIREs", "TADs", "Loop Domains",
                                    "TADBs", "Subcompartments"]
        self.class_columns = [str(i) for i in range(0, 10)]
        self.colors_list = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

        "clip parameters"
        self.clip_config = {"dim_text": 16,
                            "dim_image": 16,
                            "dim_latent": 32,
                            "num_text_tokens": 288091,
                            "text_enc_depth": 1,
                            "text_seq_len": 200,
                            "text_heads": 4,
                            "visual_enc_depth": 1,
                            "visual_image_size": 100,
                            "visual_patch_size": 10,
                            "visual_heads": 4,
                            "use_all_token_embeds": True,
                            "decoupled_contrastive_learning": True,
                            "extra_latent_projection": True,
                            "use_visual_ssl": False,
                            "visual_ssl_type": 'simclr',
                            "use_mlm": False,
                            "text_ssl_loss_weight": 0,
                            "image_ssl_loss_weight": 0}
        self.clip_batch_size = 50
        self.optim_config = {"learning_rate": 1.1e-2,
                                  "architecture": "clip",
                                  "dataset": "hic",
                                  "weight_decay": 6.02e-4,
                                  "max_gradient_clipping_norm": 10,
                                  "batch_size": self.clip_batch_size,
                                  "epochs": 10,
                                  "scheduler_lr": 1.1e-1}
        self.wandb_clip_entity = "clip_ob"
        self.wandb_clip_project = "clip"
        self.num_eval_batches = 5
        self.clip_model_name = "clip_try3.pth"
        self.pretrained_clip_model_path = "./clip_checkpoints/" + self.clip_model_name
        self.save_path_clip = "./clip_checkpoints/"
        self.exp_resume = False
        self.new_optim_config = True

        "diffusion parameters"
        self.dpn_depth = 6
        self.dpn_dim_head = 64
        self.dpn_heads = 8
        self.dp_normformer = False
        self.weight_decay = 6.02e-2
        self.max_grad_norm = 0.5
        self.dp_loss_type = "l2"
        self.clip = None
        self.dp_condition_on_text_encodings = False
        self.dp_timesteps = 100
        self.dp_cond_drop_prob = 0.1
        self.wandb_diffusion_config = {"learning_rate": 1.1e-4,
                                       "architecture": "DiffusionPrior",
                                       "dataset": "LAION-5B",
                                       "weight_decay": self.weight_decay,
                                       "max_gradient_clipping_norm": self.max_grad_norm,
                                       "batch_size": 10 ** 4,
                                       "epochs": 5,
                                       "diffusion_prior_network": {"depth": self.dpn_depth,
                                                                   "dim_head": self.dpn_dim_head,
                                                                   "heads": self.dpn_heads,
                                                                   "normformer": self.dp_normformer},
                                       "diffusion_prior": {
                                           "condition_on_text_encodings": self.dp_condition_on_text_encodings,
                                           "timesteps": self.dp_timesteps,
                                           "cond_drop_prob": self.dp_cond_drop_prob,
                                           "loss_type": self.dp_loss_type,
                                           "clip": self.clip}}
        self.wandb_diffusion_entity = "laion"
        self.wandb_diffusion_project = "diffusion-prior"
        self.image_embed_url = "https://mystic.the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/img_emb/"
        self.text_embed_url = "https://mystic.the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/text_emb/"
        self.dropout = 5e-2
        self.amp = False
        self.image_embed_dim = 768
        self.train_percent = 0.7
        self.val_percent = 0.2
        self.test_percent = 0.1
        self.save_interval = 30
        self.save_path_diffusion = "./diffusion_prior_checkpoints"
        self.pretrained_diffusion_model_path = None

        self.num_test_embeddings = 100
        self.report_metrics_every = 100
        self.ignore_batches_chr3 = [182, 154, 175, 136, 302, 4, 488, 774, 341, 40, 330, 298, 675, 582, 661, 359, 742,
                                    162, 733, 567, 327, 386, 770, 463, 50, 196, 273, 393, 499, 89, 305, 185,
                                    12, 629, 738, 169, 553, 697, 750, 133, 412, 369, 212, 413, 649, 250, 62,
                                    237, 122, 541, 116, 611, 659, 13, 264, 405, 101, 772, 594, 353, 203, 265,
                                    511, 721, 740, 255, 604, 754, 680, 584, 0, 215, 285, 83, 436, 588, 398,
                                    147, 615, 465, 646, 220, 295, 157, 34, 453, 688, 534, 585, 771, 340, 1,
                                    233, 423, 506, 312, 318, 428, 454, 45, 53, 194, 612, 336]
