import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances

import time
from tqdm import tqdm

from torch_geometric.utils import remove_self_loops, k_hop_subgraph





def norVSnor_similarity():
    # convert to nor matrix and abnor matrix on embedding
    #pairwise_distances(node_x, node_y, metric='l1')
    # homophily(edge_index, y, method='edge')
    # homophily(edge_index, y, method='node')
    # homophily(edge_index, y, method='edge_insensitive')
    return


def norVSabnor_similarity():
    #pairwise_distances(node_x, node_y, metric='l1')
    return


def abnorVSabnor_similarity():
    #pairwise_distances(node_x, node_y, metric='l1')
    return




def neighbor_similarity(embedding, edge_index, label, sample_type, k_hop=1):
    embedding = embedding.detach().cpu().numpy()

    edge_index, _ = remove_self_loops(edge_index) # no use
    n_samples, n_dim = embedding.shape

    nor_label = torch.where(label == 0)
    abnor_label = torch.where(label == 1)
    nor_index = nor_label[0].numpy().tolist()
    abnor_index = abnor_label[0].numpy().tolist()

    if sample_type == 'abnormal':
        center_list = abnor_index
    elif sample_type == 'normal':
        center_list = nor_index
    elif sample_type == 'all':
        center_list = list(range(0, n_samples))

    node_num = len(center_list)
    total_similarity = 0

    # with tqdm(total=len(center_list)) as pbar:
    #     pbar.set_description('Similarity')
    for node in center_list:
        neighbor_subset, neighbor_edge_index, _, _ = k_hop_subgraph(node, k_hop, edge_index, relabel_nodes=True)
        neighbor_subset = neighbor_subset.tolist()
        neighbor_subset.remove(node)
        if len(neighbor_subset) != 0:
            a = embedding[node].reshape(1, -1)
            b = embedding[neighbor_subset]
            similarity = pairwise_distances(a, b, metric='cosine')
            total_similarity += similarity.mean()
        else:
            node_num -= 1
            similarity = 0
            # pbar.update(1)


    return total_similarity/node_num


