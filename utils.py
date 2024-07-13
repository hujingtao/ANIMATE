from load_pygData import *
import torch.nn.functional as F
from sklearn.impute import SimpleImputer, KNNImputer
import pyximport
# pyximport.install()
pyximport.install(setup_args={"include_dirs": np.get_include()})
import algos

from torch_geometric.utils import to_undirected
from torch_geometric.transforms import NormalizeFeatures
import torch_geometric.utils as ut
from sklearn.preprocessing import StandardScaler,MinMaxScaler


real_data_list = ['Disney', 'Books', 'Reddit', 'Weibo', 'Enron', 'inj_amazon', 'Yelp', 'Elliptic']
def preprocess_dataset(dataset_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = load_dataset(dataset_name)

    if dataset_name in real_data_list: # dataset_name == 'weibo':
        # trans = NormalizeFeatures()
        # data = trans(data)
        data = standScale(data)
    else:
        if dataset_name != 'Flickr':
            edge_index = ut.add_self_loops(data.edge_index)[0]
            data.edge_index = edge_index
        else:
            data = standScale(data)

    data = data.to(device)
    return data


def preprocess_dataset_old(dataset_name):
    data = load_dataset(dataset_name)
    data.edge_index = to_undirected(data.edge_index)

    X = data.x.numpy()
    X = normalize_numpy(X)
    # X_mask, _ = mask_feature(X, missing_ratio=mask_ratio, missing_type='uniform', missing_value=0)
    # data.x = torch.tensor(X_mask)
    data.x = torch.tensor(X)

    # X = data.x
    # X_norm = normalize_numpy(X)
    # X_norm = normalize_col(X)
    # X_norm = scale_feats(X)
    # X_norm = normalize_row(X_norm)
    # data.x = X_norm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    return data


def load_dataset(dataset_name):
    if dataset_name in ['Disney', 'Books', 'Reddit', 'Weibo', 'Enron', 'inj_amazon']:
        PygData = load_pt(dataset_name)
    elif dataset_name in ['Yelp', 'Elliptic']:
        PygData = load_dat(dataset_name)
    elif dataset_name in ['cora', 'citeseer', 'pubmed', 'BlogCatalog', 'ACM', 'Flickr']:
        PygData = load_mat2pyg(dataset_name)
    elif dataset_name in ['Amazon', 'Tolokers', 'Questions']: #, 'Elliptic', 'YelpChi'
        PygData = load_dgl2pyg(dataset_name)
    else:
        raise ValueError("Dataset file not provided!")

    return PygData




def NormalizeToOne(data):
    x = data.x/(torch.norm(data.x,dim=-1).reshape(-1,1)+1e-10)
    data.x = x
    return data

def standScale(data):
    x = data.x.numpy()
    enc = StandardScaler()
    x = torch.from_numpy(enc.fit_transform(x)).type(torch.FloatTensor)
    data.x = x
    return data

def minMaxScale(data):
    x = data.x.numpy()
    enc = MinMaxScaler()
    x = torch.from_numpy(enc.fit_transform(x)).type(torch.FloatTensor)
    data.x = x
    return data


def normalize_numpy(X): # all?
    # X = X.numpy()
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X_norm = (X - x_min) / (x_max - x_min)
    if np.isnan(X_norm).any():
        X_norm = filling(X_norm, 'constant')
    # X_norm = torch.tensor(X_norm)
    return X_norm


def normalize_col(X): # according to col
    X_norm = F.normalize(X, p=1, dim=0) #p:l1; dim=0 col
    return X_norm

def normalize_row(X): # according to row
    X_norm = F.normalize(X, p=1, dim=1) #p:l1; dim=0 col
    return X_norm

# def normalize_row(X): # according to node(row)
#     X = X.tolist()
#     trans = NormalizeFeatures()
#     X_norm = trans(X)
#     X_norm = torch.tensor(X_norm)
#     return X_norm

def scale_feats(X):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    feats = X.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats


def filling(missing_matrix, fill_strategy, fill_value=0, knn_neighbors=5):

    if fill_strategy in ['mean', 'median', 'most_frequent', 'constant']:
        imputer = SimpleImputer(strategy=fill_strategy, fill_value=fill_value,
                            add_indicator=False, copy=False, missing_values=np.nan)
        fill_matrix = imputer.fit_transform(missing_matrix)
    elif fill_strategy == 'knn':
        imputer = KNNImputer(n_neighbors=knn_neighbors, weights="uniform")
        fill_matrix = imputer.fit_transform(missing_matrix)

    return fill_matrix


def preprocess_data(data):
    X = data.x.numpy()
    edge_index = data.edge_index
    num_samples, num_dim = X.shape

    # node adj matrix [N, N] bool
    adj = torch.zeros([num_samples, num_samples], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True


    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([num_samples, num_samples], dtype=torch.float)
    # attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

    degree = adj.long().sum(dim=0).view(-1)

    data.attn_bias = attn_bias
    data.spatial_pos = spatial_pos
    data.degree = degree

    # return attn_bias, spatial_pos, degree