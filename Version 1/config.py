import torch


class Config:
    def __init__(self):
        self.data_paths = {
            #'ciao': '/scratch/nh52/rx3582/tmp/AHNTE/data/ciao',
            'epinions': '/scratch/nh52/rx3582/tmp/AHNTE/data/epinions',
        }
        self.emb_dim = 256  # 用户的特征feature 维度  7317 × 64
        self.hidden_dim = 128
        self.out_features = 64
        self.num_layers = 3     # 网络层数
        self.epoch_max = 1000  # 训练轮次
        self.batch_size = 4096  # 训练大小
        self.num_workers = 8
        self.weight_decay = 1e-4
        self.lr = 0.001         # 学习率
        self.val_freq = 20
        self.seed = 2022
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.kHop = 2       # 用户多跳
        # 多跳用户卷积(encoder): hgnnp, uingcn, unigat, unisage, unigin
        self.conv = 'unigat'
        self.use_pagerank = True        # 是否使用pagerank
