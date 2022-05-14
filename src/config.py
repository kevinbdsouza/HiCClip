import os
import pathlib


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
        self.chr_train_list = list(range(22, 23))
        self.chr_test_list = list(range(22, 23))
        self.save_processed_data = False

        "fasta"
        self.data_path = "/data2/hicfold/"
        self.downstream_dir = self.data_path + "downstream/"
        self.fasta_batch_size = 1000
        self.fasta_path = self.data_path + "fasta/"

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

        "knockout"

        "duplication"
