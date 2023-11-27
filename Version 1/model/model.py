from typing import Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import dhg
from dhg.nn import HGNNPConv, UniGINConv, UniGATConv, UniGCNConv, UniSAGEConv
from dhg.structure.graphs import BiGraph


class Model(nn.Module):
    def __init__(self, num_users: int, num_items: int, emb_dim: int, hidden_feature: int, out_channels: int,
                 edge_features: int,
                 conv: str = 'unigin', num_layers: int = 3, use_pagerank: bool = False, bias: bool = True):
        super().__init__()
        # 用户商品网络层  学习用户 商品的高阶表示
        self.UILayer = UIModel(num_users, num_items, emb_dim)
        # 网络层数
        self.num_layers = num_layers

        self.hid_num = hidden_feature
        self.out_num = out_channels
        self.bias = bias
        self.edge_feafures = edge_features

        self.conv_name = conv
        self.conv_list = {
            'hgnnp': HGNNPConv,
            'uingcn': UniGCNConv,
            'unigat': UniGATConv,
            'unisage': UniSAGEConv,
            'unigin': UniGINConv
        }

        self.use_pagerank = use_pagerank
        self.pagerank_dim = 7
        if self.use_pagerank:
            # 用户超边 超图卷积
            self.Conv_Layer = self.conv_list[conv]((emb_dim + self.pagerank_dim), hidden_feature, bias=self.bias)
            # 学习 信任者的不同权重边特征学习
            self.Trustor_UniGAT_Layer = UniGATConv((emb_dim + self.pagerank_dim), self.hid_num, self.bias)
            # 学习 被信任者的不同权重边特征学习
            self.Trustee_UniGAT_Layer = UniGATConv((emb_dim + self.pagerank_dim), self.hid_num, self.bias)
        else:
            self.Conv_Layer = self.conv_list[conv](emb_dim, hidden_feature, bias=self.bias)
            self.Trustor_UniGAT_Layer = UniGATConv(emb_dim, self.hid_num, self.bias)
            self.Trustee_UniGAT_Layer = UniGATConv(emb_dim, self.hid_num, self.bias)
        # MLP
        self.MLP = MLPLayer(hidden_feature * 2, hidden_feature, self.out_num)
        self.act = nn.ReLU()
        self.theta = nn.Linear(hidden_feature, out_channels, bias=self.bias)

        # 超边组属性 学习特征
        self.hyper_edge_embedding = nn.Embedding(num_users, edge_features)
        self.edge_theta = nn.Linear(edge_features, emb_dim, bias=self.bias)

        # self.out_fc = nn.Linear(self.out_num*2, 1)
        # 初始化 超边组属性权重
        self.reset_parameters()

    def reset_parameters(self):
        r"""Initialize learnable parameters.
        """
        nn.init.normal_(self.hyper_edge_embedding.weight, 0, 0.1)

    def forward(self, ui_bigraph: BiGraph, social_hg: dhg.Hypergraph, or_hg: dhg.Hypergraph, ee_hg: dhg.Hypergraph,
                pagerank_weight: torch.Tensor, or_x: torch.Tensor, ee_x: torch.Tensor) -> Tuple[Any, Any, Tensor]:

        # get user embedding
        u_embs = self.UILayer(ui_bigraph)

        # hyperedge attribute
        edge_embs = self.hyper_edge_embedding.weight
        edge_x = social_hg.smoothing_with_HGNN(edge_embs)
        edge_layer = self.edge_theta(edge_x)

        u_embs = u_embs + edge_layer

        # use pagerank
        if self.use_pagerank:
            u_embs = torch.concat((u_embs, pagerank_weight), dim=-1)

        # generate the user hidden feature
        if self.conv_name in ['hgnnp', 'uingcn', 'unigin']:
            hyper_x_lats = []
            for _ in range(self.num_layers):
                _x = self.Conv_Layer(u_embs, social_hg)
                hyper_x_lats.append(_x)

            hyper_x_lats = sum(hyper_x_lats)
        else:
            hyper_x_lats = self.Conv_Layer(u_embs, social_hg)

        or_gat_x = self.Trustor_UniGAT_Layer(u_embs, or_hg)
        ee_gat_x = self.Trustee_UniGAT_Layer(u_embs, ee_hg)

        or_gat_x = torch.concat((or_gat_x, hyper_x_lats), dim=1)
        ee_gat_x = torch.concat((ee_gat_x, hyper_x_lats), dim=1)

        # MLP Layer
        trustor_X = self.MLP(or_gat_x)
        trustee_X = self.MLP(ee_gat_x)

        # add
        hyper_x = self.act(self.theta(hyper_x_lats))
        trustor_X = trustor_X + hyper_x
        trustee_X = trustee_X + hyper_x

        trustor_X = trustor_X[or_x]
        trustee_X = trustee_X[ee_x]

        # similarity = torch.cosine_similarity(trustor_X, trustee_X, dim=1)
        # loss_sim = self.act(similarity)

        # 方法1：各个向量积
        output = (trustor_X * trustee_X).sum(-1)
        # 方法2：MLP输出
        # output = self.out_fc(torch.concat([trustor_X, trustee_X], dim=-1)).squeeze()

        return trustor_X, trustee_X, output

    @staticmethod
    def loss(trustor_X, trustee_X, output, labels):
        labels = labels.float()
        cross_loss = F.binary_cross_entropy_with_logits(output, labels)

        # compute cosine similarities using PyTorch
        trustor_X_norm = trustor_X / trustor_X.norm(dim=1, keepdim=True)
        trustee_X_norm = trustee_X / trustee_X.norm(dim=1, keepdim=True)
        cosine_sim = torch.mm(trustor_X_norm, trustee_X_norm.t())

        # remove diagonal elements and apply temperature
        cosine_sim = cosine_sim / 0.3
        I = torch.eye(cosine_sim.size(0)).bool().to(cosine_sim.device)
        cosine_sim[I] = 0

        exp_cosine_sim = torch.exp(cosine_sim)
        row_sum = torch.sum(exp_cosine_sim, dim=1, keepdim=True)

        mask = labels.view(-1, 1) == labels
        mask[I] = 0

        inner_sum = torch.log(exp_cosine_sim / row_sum) * mask.float()
        contrastive_loss = -torch.sum(inner_sum) / torch.sum(mask.float())

        # final loss
        loss = 0.04 * contrastive_loss + 6 * cross_loss
        
        return loss * 2


class UIModel(nn.Module):
    r"""note::
            The user and item embeddings are initialized with normal distribution.
        Args:
            ``num_users`` (``int``): The Number of users.
            ``num_items`` (``int``): The Number of items.
            ``emb_dim`` (``int``): Embedding dimension.
            ``num_layers`` (``int``): The Number of layers. Defaults to ``3``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in training stage with probability ``drop_rate``. Default: ``0.0``.
        """

    def __init__(self, num_users: int, num_items: int, emb_dim: int, num_layers: int = 3,
                 drop_rate: float = 0.0) -> None:
        super().__init__()
        self.num_users, self.num_items = num_users, num_items
        self.num_layers = num_layers
        self.drop_rate = drop_rate
        self.u_embedding = nn.Embedding(num_users, emb_dim)
        self.i_embedding = nn.Embedding(num_items, emb_dim)
        self.reset_parameters()

    def reset_parameters(self):
        r"""Initialize learnable parameters.
        """
        nn.init.normal_(self.u_embedding.weight, 0, 0.1)
        nn.init.normal_(self.i_embedding.weight, 0, 0.1)

    def forward(self, ui_bigraph: BiGraph) -> torch.Tensor:
        r"""The forward function.
        Args:
            ``ui_bigraph`` (``dhg.BiGraph``): The user-item bipartite graph.
        """
        drop_rate = self.drop_rate if self.training else 0.0
        u_embs = self.u_embedding.weight
        i_embs = self.i_embedding.weight
        all_embs = torch.cat([u_embs, i_embs], dim=0)

        embs_list = [all_embs]
        for _ in range(self.num_layers):
            all_embs = ui_bigraph.smoothing_with_GCN(all_embs, drop_rate=drop_rate)
            embs_list.append(all_embs)
        embs = torch.stack(embs_list, dim=1)
        embs = torch.mean(embs, dim=1)

        u_embs, _ = torch.split(embs, [self.num_users, self.num_items], dim=0)
        return u_embs


class MLPLayer(nn.Module):
    def __init__(self, num_i: int, num_h: int, num_o: int):
        super().__init__()
        self.linear1 = nn.Linear(num_i, num_h)
        #self.relu = nn.ReLu()
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.linear2 = nn.Linear(num_h, num_h)  # 2个隐层
        #self.relu2 = nn.ReLU()
        self.relu2 = nn.LeakyReLU(negative_slope=0.01)
        self.linear3 = nn.Linear(num_h, num_o)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x
