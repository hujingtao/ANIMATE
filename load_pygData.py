import os
import torch
import pickle
import scipy.io as sio
from torch_geometric.utils import dense_to_sparse
# from torch_geometric.utils.convert import from_dgl
from torch_geometric.data import Data
from dgl.data.utils import load_graphs


import scipy.sparse as sp
import numpy as np


# ['Amazon', 'Enron']
def load_mat(dataset):

    data = sio.loadmat("./data/" + dataset + ".mat")
    key = data.keys()
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    if dataset == "Enron":
        network = network.toarray()

    adj = torch.LongTensor(network)
    attr = torch.FloatTensor(attr)
    labels = torch.LongTensor(label.reshape(-1))
    # temp = np.where(label.reshape(-1) == 1)
    edge_index = dense_to_sparse(adj)[0]

    pygData = Data(x=attr, edge_index=edge_index, y=labels)

    return pygData




# ['Yelp', 'Elliptic']
def load_dat(dataset):
    data = pickle.load(open('./data/{}.dat'.format(dataset), 'rb'))
    return data


# from pygod.utils import load_data as ld
# ['Disney', 'Books', 'Reddit', 'Weibo']
def load_pt(dataset):
    file_path = os.path.join('./data/', dataset + '.pt')
    data = torch.load(file_path)
    # data = ld(name=dataset)
    # data.str_y = data.attr_y = data.y
    return data


# ['cora', ...]
def load_mat2pyg(dataset, path='./data'):
    data = sio.loadmat(f'{path}/{dataset}.mat')
    adj = torch.LongTensor(data['Network'].toarray())
    attr = torch.FloatTensor(data['Attributes'].toarray())
    label = torch.LongTensor(data['Label'].reshape(-1))
    str_label = torch.LongTensor(data['str_anomaly_label'].reshape(-1))
    attr_label = torch.LongTensor(data['attr_anomaly_label'].reshape(-1))
    edge_index = dense_to_sparse(adj)[0]

    pygData = Data(x=attr,edge_index=edge_index,y=label,str_y=str_label,attr_y=attr_label)
    return pygData


def from_dgl(g):
    import dgl
    from torch_geometric.data import Data, HeteroData

    if not isinstance(g, dgl.DGLGraph):
        raise ValueError(f"Invalid data type (got '{type(g)}')")

    if g.is_homogeneous:
        data = Data()
        data.edge_index = torch.stack(g.edges(), dim=0)

        for attr, value in g.ndata.items():
            data[attr] = value
        for attr, value in g.edata.items():
            data[attr] = value

        return data

    data = HeteroData()

    for node_type in g.ntypes:
        for attr, value in g.nodes[node_type].data.items():
            data[node_type][attr] = value

    for edge_type in g.canonical_etypes:
        row, col = g.edges(form="uv", etype=edge_type)
        data[edge_type].edge_index = torch.stack([row, col], dim=0)
        for attr, value in g.edge_attr_schemes(edge_type).items():
            data[edge_type][attr] = value

    return data


def load_dgl2pyg(dataset, path='./dgl_data/'):
    dglData = load_graphs(path + dataset.lower())[0][0]
    pygData = from_dgl(dglData)
    pygData.x = pygData.feature
    pygData.y = pygData.label
    return pygData
