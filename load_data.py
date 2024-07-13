from collator import collator
from torch.utils.data import DataLoader
from functools import partial

import torch
from wrapper import MyDataset
from torch_geometric.utils import to_undirected

from torch_geometric.datasets import Planetoid, WikiCS, Amazon
from torch_geometric.loader import NeighborSampler
import torch_geometric.transforms as T

from utils import preprocess_dataset

dataset = None





class GraphDataModule():

    def __init__(self,args):
        super().__init__()
        self.dataset_name = args.dataset_name
        self.data = preprocess_dataset(self.dataset_name)
        self.seed = args.seed
        self.l1 = args.l1
        self.l2 = args.l2

        # self.batch_size = args.batch_size
        self.dataset_train = ...



    def process_samples(self, batch_size, n_id, adj):
        edge_index = adj[0].edge_index
        if edge_index.size(1) != 0:
            edge_index = to_undirected(edge_index)
        #     print('undirect?')
        n_nodes = len(n_id)
        edge_sp_adj = torch.sparse.FloatTensor(edge_index,
                                               torch.ones(edge_index.shape[1]),
                                               [n_nodes, n_nodes])
        edge_adj = edge_sp_adj

        return [self.data.x[n_id], self.data.y[n_id[0]], edge_adj]
        # return [self.data.x[n_id], edge_adj]

    def data_dataloader(self, batch_size=128, shuffle=True):
        sampler = NeighborSampler(self.data.edge_index, sizes=[self.l1, self.l2], batch_size=1,
                                  shuffle=False, node_idx=None) #node_idx=self.shuffled_index means only train ; None means all node
        items = []
        for s in sampler:
            items.append(self.process_samples(s[0], s[1], s[2]))
        self.dataset_train = MyDataset(items)
        loader = DataLoader(self.dataset_train, batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=0,
                            # num_workers=self.num_workers,
                            collate_fn=partial(collator),
                            )
        return loader


