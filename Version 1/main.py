# -*- coding: utf-8 -*-

import click
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import *
from pytorch_lightning.loggers import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from config import Config
from dataset import DataReader
from dhg import Graph, BiGraph, Hypergraph
from dhg.random import set_seed
from model.model import Model
from utils import get_user_item, generate_edge_list


class Net(pl.LightningModule):
    def __init__(self, dataset, config: Config):
        super().__init__()
        self.config = config
        self.dataset = dataset
        # pagerank 数据
        self.pagerank = torch.Tensor(dataset.pr_weights).to(self.config.device)

        # 生成用户商品的二分图，从而获取用户与商品的高阶信息
        self.ui_bigraph = BiGraph.from_adj_list(self.dataset.num_users, self.dataset.num_items,
                                                self.dataset.adj_list, device=self.config.device)

        # 生成用户的kHop信息
        # 1 生成邻接矩阵
        self.social_edge_list = generate_edge_list(self.dataset.social_adj)
        # 2 生成用户图（普通）
        self.social_g = Graph(self.dataset.num_users, self.social_edge_list)
        # 3 生成用户超图
        self.social_hg = Hypergraph.from_graph_kHop(self.social_g, k=self.config.kHop, device=self.config.device)
        # 计算 用户超边组
        self.hyper_edge_num = len(self.social_hg.e[0])

        # 进入模型加载
        self.model = Model(dataset.num_users, dataset.num_items, config.emb_dim, config.hidden_dim, config.out_features,
                           self.hyper_edge_num, config.conv, config.use_pagerank, config.num_layers)

    def training_step(self, batch, batch_idx):
        trustor, trustee, labels = batch
        labels = labels.to(self.config.device)

        trustor_list = trustor.cpu().tolist()
        trustee_list = trustee.cpu().tolist()

        # 筛选 batch 中用户的 点评商品数据
        trustor_u_i = get_user_item(trustor_list, self.dataset.u_v_info)
        trustee_u_i = get_user_item(trustee_list, self.dataset.u_v_info)

        # 生成`二分图` 和 `超图`
        trustor_ui_bigraph = BiGraph.from_adj_list(self.dataset.num_users, self.dataset.num_items, trustor_u_i)
        trustor_hg = Hypergraph.from_bigraph(trustor_ui_bigraph, device=self.config.device)
        # 生成被信任者的 `二分图` 和 `超图`
        trustee_ui_bigraph = BiGraph.from_adj_list(self.dataset.num_users, self.dataset.num_items, trustee_u_i)
        trustee_hg = Hypergraph.from_bigraph(trustee_ui_bigraph, device=self.config.device)

        or_x_, ee_x_, output = self.model.forward(self.ui_bigraph, self.social_hg,
                                                  trustor_hg, trustee_hg,
                                                  self.pagerank,
                                                  trustor, trustee)
        labels = labels.float()
        loss = self.model.loss(or_x_, ee_x_, output, labels)
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        trustor, trustee, labels = batch
        labels = labels.to(self.config.device)

        trustor_list = trustor.cpu().tolist()
        trustee_list = trustee.cpu().tolist()

        # 筛选 batch 中用户的 点评商品数据
        trustor_u_i = get_user_item(trustor_list, self.dataset.u_v_info)
        trustee_u_i = get_user_item(trustee_list, self.dataset.u_v_info)

        # 生成`二分图` 和 `超图`
        trustor_ui_bigraph = BiGraph.from_adj_list(self.dataset.num_users, self.dataset.num_items, trustor_u_i)
        trustor_hg = Hypergraph.from_bigraph(trustor_ui_bigraph, device=self.config.device)
        # 生成被信任者的 `二分图` 和 `超图`
        trustee_ui_bigraph = BiGraph.from_adj_list(self.dataset.num_users, self.dataset.num_items, trustee_u_i)
        trustee_hg = Hypergraph.from_bigraph(trustee_ui_bigraph, device=self.config.device)

        or_x_, ee_x_, output = self.model.forward(self.ui_bigraph, self.social_hg,
                                                  trustor_hg, trustee_hg,
                                                  self.pagerank,
                                                  trustor, trustee)

        loss = self.model.loss(or_x_, ee_x_, output, labels)
        labels = labels.cpu().numpy()
        output = output.cpu().numpy()
        predicts = output >= 0.5

        self.log_dict({'valid_loss': loss.item(),
                       'acc': accuracy_score(labels, predicts),
                       'pre': precision_score(labels, predicts, zero_division=0),
                       'recall': recall_score(labels, predicts, zero_division=0),
                       'f1': f1_score(labels, predicts, zero_division=0),
                       'auc': roc_auc_score(labels, output),
                       'positive_ratio': predicts.sum() / len(predicts),
                       'min_out': output.min()})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        return optimizer


@click.command()
@click.argument('task', default='train')
@click.option('--data', default='epinions', help='Which dataset to use.')
def main(task, data):
    config = Config()
    set_seed(config.seed)

    data_reader = DataReader(config.data_paths[data])
    train_loader = DataLoader(data_reader.train_dual_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=True)
    test_loader = DataLoader(data_reader.test_dual_dataset, batch_size=config.batch_size, shuffle=False,
                             num_workers=config.num_workers, pin_memory=True)

    net = Net(data_reader, config)
    # trainer = pl.Trainer(max_epochs=config.epoch_max, logger=CSVLogger('.'), log_every_n_steps=1,
    #                      callbacks=[
    #                          EarlyStopping(monitor='f1', mode='max', patience=5, verbose=True),
    #                          ModelCheckpoint(every_n_epochs=1, save_top_k=10, monitor='f1', mode='max')],
    #                      accelerator='gpu' if config.device.type == 'cuda' else 'auto')
    trainer = pl.Trainer(max_epochs=config.epoch_max, logger=CSVLogger('.'), log_every_n_steps=1,
                         callbacks=[
                             EarlyStopping(monitor='f1', mode='max', patience=5, verbose=True)],
                         accelerator='gpu' if config.device.type == 'cuda' else 'auto')
    if task == 'train':
        trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=test_loader)
    elif task == 'test':
        output = trainer.test(net, test_loader)
        print(output)
    else:
        raise ValueError('Unknown task! (should be `train` or `test`)')


if __name__ == "__main__":
    '''
    运行方法： 
        python main.py [TASK] --data=[DATA]
        其中，task默认值为train，data默认值为ciao。
        那么，pycharm里不带参数直接运行，就等同于： python main.py train --data=ciao
    '''
    main()
