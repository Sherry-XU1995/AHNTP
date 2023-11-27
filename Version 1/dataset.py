# -*- coding: utf-8 -*-

import random
from pathlib import Path
from typing import Optional
from dhg.datapipe import load_from_pickle, load_from_txt


class TripletDataset:
    def __init__(self, pairs):
        self.pos_pairs = [_ for _ in pairs if _[2] == 1]
        self.neg_pairs = [_ for _ in pairs if _[2] == 0]
        self.trustor_neg_map = {}
        for pair in self.neg_pairs:
            self.trustor_neg_map.setdefault(pair[0], []).append(pair[1])

    def __getitem__(self, idx):
        pos_pair = self.pos_pairs[idx]
        return pos_pair[0], pos_pair[1], random.choice(self.trustor_neg_map[pos_pair[0]])

    def __len__(self):
        return len(self.pos_pairs)


class DataReader:
    def __init__(self, data_root: Optional[str] = None) -> None:
        self.data_root = data_root
        self.adj_list = load_from_txt(Path(data_root, 'adj_list.txt'), dtype="int", sep=" ")
        self.rating = load_from_pickle(Path(data_root, 'rating.pkl'))
        self.train_mask = load_from_pickle(Path(data_root, 'train_mask.pkl'))
        self.test_mask = load_from_pickle(Path(data_root, 'test_mask.pkl'))
        self.total_trust_pair = load_from_pickle(Path(data_root, 'total_trust_pair.pkl'))
        self.full_adj = load_from_pickle(Path(data_root, 'full_adj.pkl'))
        self.social_adj = load_from_pickle(Path(data_root, 'social_adj.pkl'))
        self.u_v_info = load_from_pickle(Path(data_root, 'u_v_info.pkl'))
        self.u_r_info = load_from_pickle(Path(data_root, 'u_r_info.pkl'))
        self.v_u_info = load_from_pickle(Path(data_root, 'v_u_info.pkl'))
        self.v_r_info = load_from_pickle(Path(data_root, 'v_r_info.pkl'))
        self.pr_weights = load_from_pickle(Path(data_root, 'pr_weights.pkl'))

    @property
    def num_users(self):
        return len(self.u_v_info)

    @property
    def num_items(self):
        return len(self.v_u_info)

    @property
    def num_edges(self):
        return len(self.rating)

    @property
    def train_dual_dataset(self):
        return self.train_mask

    @property
    def test_dual_dataset(self):
        return self.test_mask

    @property
    def train_triplet_dataset(self):
        return TripletDataset(self.train_mask)

    @property
    def test_triplet_dataset(self):
        return TripletDataset(self.test_mask)


if __name__ == '__main__':
    #ciao_reader = DataReader('/scratch/nh52/rx3582/tmp/AHNTE/data/ciao')
    #assert ciao_reader.num_users, 4104
    #assert ciao_reader.num_items, 75071
    #assert ciao_reader.num_edges, 171405

    epinions_reader = DataReader('/scratch/nh52/rx3582/tmp/AHNTE/data/epinions')
    assert epinions_reader.num_users, 8935
    assert epinions_reader.num_items, 21335
    assert epinions_reader.num_edges, 220673
