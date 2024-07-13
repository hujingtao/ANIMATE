import torch
from torch import Tensor, nn
from torch_geometric.nn import GIN, MessagePassing, GAT, GATConv, GINConv,GCN,MLP
# from torch_geometric.typing import OptTensor, OptPairTensor
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
import copy

def setup_module(mod_type, in_dim, num_hidden, out_dim, num_layers) -> nn.Module:
    if mod_type == 'gat':
        model = GAT(in_channels=in_dim, hidden_channels=num_hidden, num_layers=num_layers, out_channels=out_dim)
    elif mod_type == 'gin':
        model = GIN(in_channels=in_dim, hidden_channels=num_hidden, num_layers=num_layers, out_channels=out_dim)
    elif mod_type == 'gcn':
        model = GCN(in_channels=in_dim, hidden_channels=num_hidden, num_layers=num_layers, out_channels=out_dim)

    elif mod_type == "mlp":
        # * just for decoder
        model = nn.Sequential(
            nn.Linear(in_dim, num_hidden * 2),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden * 2, out_dim)
        )

    elif mod_type == "linear":
        model = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError

    return model





class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class GraphMAE(nn.Module):
    def __init__(
            self,
            in_dim: int,
            emb_dim: int,
            # remask_num: int,
            # seed: int = 6,
            ffn_dim: int = 64,                  # graph transformer
            num_heads: int = 8,                 # graph transformer
            n_encoder_layers: int = 4,          # graph transformer
            n_decoder_layers: int = 2,          # graph transformer
            dropout_rate: float = 0.1,          # graph transformer
            intput_dropout_rate: float = 0.1,   # graph transformer
            attention_dropout_rate: float = 0.1,# graph transformer
            return_all: bool = True,
            mask_rate: float = 0.3,
            # remask_rate: float = 0.5,
            # remask_method: str = "random",
    ):

        super(GraphMAE, self).__init__()
        self.mask_rate = mask_rate
        self.num_heads = num_heads
        self.return_all = return_all
        self.emb_dim = emb_dim

        # np.random.seed(seed)

        self.encoder_to_decoder = nn.Linear(emb_dim, emb_dim, bias=False)

        # 损失函数
        # self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

        # graph transformer
        self.input_proj = nn.Linear(in_dim, emb_dim)
        self.spatial_pos_encoder = nn.Embedding(
            512, num_heads, padding_idx=0)
        self.degree_encoder = nn.Embedding(
            512, emb_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(
            512, emb_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(
            512, emb_dim, padding_idx=0)

        self.input_dropout = nn.Dropout(intput_dropout_rate)
        encoders = [EncoderLayer(emb_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads)
                    for _ in range(n_encoder_layers)]
        self.encoder_layers = nn.ModuleList(encoders)
        self.encoder_final_ln = nn.LayerNorm(emb_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

        decoders = [EncoderLayer(emb_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads)
                    for _ in range(n_decoder_layers)]
        self.decoder_layers = nn.ModuleList(decoders)
        self.decoder_final_ln = nn.LayerNorm(emb_dim)
        self.out_proj = nn.Linear(emb_dim, in_dim)

    def compute_pos_embeddings(self, data):
        # attn_bias, spatial_pos, degree, x = data.attn_bias, data.spatial_pos, data.degree, data.x
        attn_bias, spatial_pos, x = data.attn_bias, data.spatial_pos, data.x
        in_degree, out_degree = data.in_degree, data.in_degree

        # graph_attn_bias
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # [batch_size, n_head, n_node, n_node]
        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos)
        spatial_pos_bias = spatial_pos_bias.permute(0, 3, 1, 2) #(2, 0, 1)

        graph_attn_bias = graph_attn_bias + spatial_pos_bias
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        node_feature = self.input_proj(x)
        # node_feature = node_feature + self.degree_encoder(degree)
        node_feature = node_feature + \
                       self.in_degree_encoder(in_degree) + \
                       self.out_degree_encoder(out_degree)
        graph_node_feature = node_feature

        return graph_node_feature, graph_attn_bias

    


    def encoder(self, graph_node_feature, graph_attn_bias, mask=None):
        if mask is not None:
            graph_node_feature_unmasked = graph_node_feature[:, ~mask]  # [n graph, n non-masked nodes, n hidden]
            graph_attn_bias_unmasked = graph_attn_bias[:, :, ~mask, :][:, :, :, ~mask]  # [n graph, n heads, n non-masked nodes, n non-masked nodes]
        else:
            graph_node_feature_unmasked = graph_node_feature
            graph_attn_bias_unmasked = graph_attn_bias

        # transfomrer encoder
        output = self.input_dropout(graph_node_feature_unmasked)
        for enc_layer in self.encoder_layers:
            output = enc_layer(output, graph_attn_bias_unmasked)
        output = self.encoder_final_ln(output)
        return output

    def decoder(self, output, in_degree, out_degree, graph_attn_bias, mask=None, return_all=True):
        if mask is not None:
            pos_embed = self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)
            pos_embed_vis = pos_embed[:, ~mask]
            pos_embed_mask = pos_embed[:, mask]
            node_index_mask = mask.nonzero().view(-1)
            node_index_vis = (~mask).nonzero().view(-1)
            new_node_index = torch.cat([node_index_vis, node_index_mask])
            graph_attn_bias = graph_attn_bias[:, :, new_node_index][:, :, :, new_node_index]
            output = torch.cat([output + pos_embed_vis, self.mask_token + pos_embed_mask], dim=1)
            # output = [output + pos_embed_vis, self.mask_token + pos_embed_mask]
            num_masked = pos_embed_mask.shape[1]
            num_node = new_node_index.shape[0]
        else:
            num_masked = 0
            num_node = output.shape[1]
            node_index_mask = None

        for enc_layer in self.decoder_layers:
            output = enc_layer(output, graph_attn_bias)

        if not return_all:
            output_mask = self.decoder_final_ln(output[:, -num_masked:])  # only mask part
            output_mask = self.out_proj(output_mask)
        else:
            output = self.decoder_final_ln(output)
            output = self.out_proj(output)  # [batch_size, n_node, n_feature]
            # output_vis = output[:, :(num_node-num_masked)]
            output_mask = output[:, -num_masked:]

            
        return output_mask, node_index_mask
    

    def decoder_eval(self, output, graph_attn_bias): 
        for enc_layer in self.decoder_layers:
            output = enc_layer(output, graph_attn_bias)
       
        output = self.decoder_final_ln(output)
        output = self.out_proj(output)  # [n_node, n_feature]
        
        return output


    def forward(self, data):
        x = data.x
        mask = self.mask_node(x, self.mask_rate)

        graph_node_feature, graph_attn_bias = self.compute_pos_embeddings(data)
        in_degree = data.in_degree
        out_degree = data.out_degree

        graph_mask = None
        if mask is not None:
            graph_attn_bias_masked = graph_attn_bias[:, :, ~mask, :][:, :, :, ~mask]
            if graph_attn_bias_masked.size(3) == torch.isinf(graph_attn_bias_masked).sum(3).max().item():
                n_graph = graph_attn_bias_masked.size(0)
                sup = graph_attn_bias_masked.reshape(n_graph, -1)
                length = sup.size(1)
                infs = torch.isinf(sup).sum(1)
                graph_mask = ~(infs == length).bool()
                graph_node_feature = graph_node_feature[graph_mask]
                graph_attn_bias = graph_attn_bias[graph_mask]
                in_degree = in_degree[graph_mask]
                out_degree = out_degree[graph_mask]

        output = self.encoder(graph_node_feature, graph_attn_bias, mask)
        output = self.encoder_to_decoder(output)
        output_mask, node_index_mask = self.decoder(output, in_degree, out_degree, graph_attn_bias, mask, return_all=self.return_all)

        # loss_recon = self.criterion(x, x_pred).mean()
        return output_mask, node_index_mask, graph_mask





    def evaluation(self, data):
        graph_node_feature, graph_attn_bias = self.compute_pos_embeddings(data)

        output = self.encoder(graph_node_feature, graph_attn_bias)
        output = self.encoder_to_decoder(output)
        output = self.decoder_eval(output, graph_attn_bias)

        return output


    def mask_node(self, X, mask_rate=0.3):
        if mask_rate == 0 or mask_rate > 1:
            return None
        num_nodes = X.size(1)
        # perm_indices = torch.randperm(num_nodes, device=X.device)  # 随机排列
        perm_indices = np.arange(num_nodes)
        perm_indices = np.random.permutation(perm_indices)

        num_mask_nodes = int(mask_rate * num_nodes)
        if num_mask_nodes == 0:
            return None
        mask_nodes_index = perm_indices[: num_mask_nodes]
        unmask_nodes_index = perm_indices[num_mask_nodes:]
        # print(mask_nodes_index)
        mask = np.zeros(num_nodes)
        mask[mask_nodes_index] = 1
        mask = torch.Tensor(mask).bool()

        # out_x = copy.deepcopy(X)
        # unmask_nodes = out_x[unmask_nodes_index]
        unmask_nodes = X[:, unmask_nodes_index]
        # unmask_nodes.requires_grad = True

        # return unmask_nodes, mask
        return mask
    

    def mask_Graphormer(self, X, mask_rate=0.8):
        num_nodes = X.size(0)
        num_mask = int(self.mask_ratio * num_nodes)
        mask = np.hstack([
            np.zeros(num_nodes - num_mask),
            np.ones(num_mask),
        ])
        np.random.shuffle(mask)
        mask = torch.Tensor(mask).bool()
        X_masked = X[mask]
        
        return X_masked, mask


    def random_remask(self, x, remask_rate=0.5):
        num_nodes = x.size(0)
        perm = torch.randperm(num_nodes, device=x.device)
        num_remask_nodes = int(remask_rate * num_nodes)
        remask_nodes = perm[: num_remask_nodes]
        rekeep_nodes = perm[num_remask_nodes:]

        remask_x = x.clone()
        remask_x[remask_nodes] = 0
        remask_x[remask_nodes] += self.dec_mask_token

        return remask_x, remask_nodes, rekeep_nodes

    def fixed_remask(self, x, masked_nodes):
        x[masked_nodes] = 0
        return x




class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x

