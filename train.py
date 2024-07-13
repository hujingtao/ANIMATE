import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from torch_geometric.utils import to_scipy_sparse_matrix

from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve


import time
from tqdm import tqdm


from utils import *
from model.GraphMAE_Transformer import *
from loss_func import setup_loss_fn

import matplotlib.pyplot as plt
from load_data import GraphDataModule
import random

import datetime
current_date = str(datetime.datetime.now().date())



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = args.lr
    seed = args.seed
    epochs = args.epoch
    dataset_name = args.dataset_name
    emb_dim = args.emb_dim
    # mask_rate = args.mask_rate
    weight_decay = args.weight_decay
    batch_size = args.batch_size            # train
    val_batch_size = args.val_batch_size    # inference
    warmup_epoch = args.warmup_epoch
    shrink_rate = args.shrink_rate

    # Set random seed
    # dgl.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
    dataset = GraphDataModule(args)
    data = dataset.data

    X = data.x.cpu()
    label = data.y.cpu()
    # edge_index = data.edge_index
    num_samples, num_dim = X.shape

    # print(f'Finish load {dataset}')



    model = GraphMAE(
        # seed = seed,
        in_dim = num_dim,
        emb_dim = emb_dim,
        # remask_num = args.remask_num,
        mask_rate = args.mask_rate,
        # remask_rate = args.remask_rate,
        # remask_method = args.remask_method,
        ffn_dim = 64,
        num_heads = 8,
        n_encoder_layers = 4,
        n_decoder_layers = 2,
        dropout_rate = 0.1,
        intput_dropout_rate = 0.1,
        attention_dropout_rate = 0.1,
        ).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss(reduction='mean')
    loss_fn_sample = nn.MSELoss(reduction='none')
    score_fn = nn.MSELoss(reduce=False)

    best = 1e9
    train_losses = []

    batch_cnt = 0
    if num_samples % batch_size == 0:
        batch_num = num_samples // batch_size
    else:
        batch_num = num_samples // batch_size + 1
        last_batch_num = num_samples % batch_size

    with tqdm(total=epochs) as pbar:
        pbar.set_description('Training')

        t = 0
        init_std_range = 4
        for epoch in range(epochs):
            dataloader = dataset.data_dataloader(batch_size=batch_size,shuffle=True)
            model.train()
            epoch_loss = 0.

            for batched_data in dataloader:

                is_final_batch = (batch_cnt == (batch_num - 1))
                if batch_num == 0:
                    is_final_batch = True
                batch_cnt += 1




                recon_mask, index_mask, graph_mask = model(batched_data)
                # recon_vis, index_vis, recon_mask, index_mask, graph_mask = model(batched_data)
                if index_mask is None:
                    x_mask = batched_data.x.float()
                else:
                    if graph_mask is not None:
                        # x_vis = batched_data.x[graph_mask][:, index_vis].float()
                        x_mask = batched_data.x[graph_mask][:, index_mask].float()
                    else:
                        # x_vis = batched_data.x[:, index_vis].float()
                        x_mask = batched_data.x[:, index_mask].float()

                recon_mask = recon_mask.reshape(-1, recon_mask.size(2))  # [n_graph*n_masked_node, n_feature]
                x_mask = x_mask.reshape(-1, x_mask.size(2))  # [n_graph*n_masked_node, n_feature]
                # recon_x = recon_x.reshape(recon_x.size(0), -1)  # [n_graph*n_masked_node, n_feature]
                # target_x = target_x.reshape(batched_data.x.size(0), -1)

                # new
                # pad_mask = torch.nonzero(x_mask.sum(-1))
                # x_mask = x_mask[pad_mask, :].squeeze()
                # recon_mask = recon_mask[pad_mask, :].squeeze()

                # -----------------SP------------------#
                weights = torch.ones(x_mask.size(0)).cuda().type(torch.cuda.FloatTensor)
                # select = (shrink_rate > 0)
                if epoch >= warmup_epoch and args.select:
                    model.eval()
                    n_nodes = x_mask.shape[0]
                    x_new = x_mask.clone().detach()
                    recon_new = recon_mask.clone().detach()

                    scores = score_fn(x_new, recon_new)
                    scores = torch.sum(scores, dim=1)

                    weights = torch.ones_like(scores)
                    avg = torch.mean(scores)
                    std = torch.std(scores)

                    # lambda1_sp = avg - (t * shrink_rate) * std
                    # lambda2_sp = avg + 1.0 * std

                    # if lambda1_sp <= 0:
                    #     lambda1_sp = int(n_nodes / 2)
                    # if lambda1_sp > lambda2_sp:
                    #     lambda1_sp = lambda2_sp

                    lambda1_sp = avg + 1.0 * std
                    lambda2_sp = avg + (init_std_range - t * shrink_rate) * std
                    if lambda2_sp < lambda1_sp:
                        lambda2_sp = lambda1_sp

                    t += 1
                    easy_idx = torch.where(scores <= lambda1_sp)[0]     # normal
                    median_idx = torch.where((scores > lambda1_sp) & (scores < lambda2_sp))[0]
                    hard_idx = torch.where(scores >= lambda2_sp)[0]     # anomalies
                    weights[easy_idx] = 1.0
                    if len(hard_idx) > 0:
                        weights[hard_idx] = 0.0
                    if len(median_idx) > 0:
                        weights[median_idx] = (lambda1_sp * (lambda2_sp - scores[median_idx])) / (scores[median_idx] * (lambda2_sp - lambda1_sp))
                        # weights[median_idx] = (lambda1_sp * (lambda2_sp - 1)) / (scores[median_idx] * (lambda2_sp - lambda1_sp))
                    model.train()


                loss_mask = loss_fn_sample(x_mask, recon_mask)
                loss_mask = loss_mask.mean(dim=1)

                recon_loss = loss_mask @ weights
                recon_loss_nonzero = torch.count_nonzero(weights)
                recon_loss = recon_loss / recon_loss_nonzero
                # recon_loss = loss_mask @ weights / x_mask.size(0) / x_mask.size(1)

                recon_loss.backward()
                optimizer.step()

                loss = float(recon_loss.detach().cpu().numpy())
                if not is_final_batch:
                    epoch_loss += loss

            mean_loss = (epoch_loss * batch_size + loss * last_batch_num) / num_samples
            train_losses.append(mean_loss)

            if mean_loss < best:
                best = mean_loss
                best_e = epoch
                torch.save(model.state_dict(), 'result/best_model_' + dataset_name + '.pkl')

            pbar.set_postfix(loss=mean_loss)
            pbar.update(1)




    # Inference
    print('Loading {}th epoch model'.format(best_e))
    path = 'result/best_model_' + dataset_name + '.pkl'
    model.load_state_dict(torch.load(path))

    multi_round_ano_score = np.zeros((args.auc_test_rounds, num_samples))
    with tqdm(total=args.auc_test_rounds) as pbar_test:
        pbar_test.set_description('Testing')
        for round in range(args.auc_test_rounds):
            dataloader = dataset.data_dataloader(batch_size=val_batch_size, shuffle=False)
            model.eval()
            label_gt = []
            ano_score_all = []
            for batched_data in dataloader:
                with torch.no_grad():
                    recon_x = model.evaluation(batched_data)
                    loss = score_fn(batched_data.x, recon_x)
                    loss = loss[:, 0, :]
                    label_gt.append(batched_data.y.cpu().numpy())

                ano_score = loss.mean(dim=1) #_mask + lamda_test * loss_recon
                ano_score = ano_score.detach().cpu().numpy()
                ano_score_all.append(ano_score)

            label_gt = np.concatenate(label_gt, axis=0)
            ano_score_all = np.concatenate(ano_score_all, axis=0)

            multi_round_ano_score[round, :] = ano_score_all

            pbar_test.update(1)

    ano_score_final = np.mean(multi_round_ano_score, axis=0)

    # dist = nn.PairwiseDistance(p=2)
    # ano_score = dist(node_res, x)


    AUC_ROC = roc_auc_score(label_gt, ano_score_final)
    AP = average_precision_score(label_gt, ano_score_final)
    precision, recall, _ = precision_recall_curve(label_gt, ano_score_final)
    AUPRC = auc(recall, precision)

    print('{}  AUROC:{:.2%},  AUPRC:{:.2%},  AP:{:.2%}'.format(dataset_name, AUC_ROC, AUPRC, AP))





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ANIMATE')
    # ['Disney', 'Books', 'Reddit', 'Weibo', 'Yelp', 'Elliptic', 'Enron']
    parser.add_argument('--dataset_name', type=str, default='Books')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1) 
    parser.add_argument('--lr', type=float, default=1e-7) 
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--auc_test_rounds', type=int, default=256)
    parser.add_argument("--n_encoder_layers", type=int, default=6) 
    parser.add_argument("--n_decoder_layers", type=int, default=2)
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--loss_fn", type=str, default="mse") # mse sce
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=4096) 
    parser.add_argument("--l1", type=int, default=3)
    parser.add_argument("--l2", type=int, default=3)
    parser.add_argument("--select", type=bool, default=True) # True
    parser.add_argument("--warmup_epoch", type=int, default=20) 
    parser.add_argument("--shrink_rate", type=float, default=0.05) # or 0.005
    args = parser.parse_args()



    dataset = args.dataset_name
    if dataset in ['Disney']:
        args.lr = 1e-3
        args.epoch = 500
    elif dataset in ['Books']:
        args.lr = 1e-3
        args.epoch = 300
    elif dataset in ['Reddit']:
        args.lr = 1e-6
        args.epoch = 100
    elif dataset in ['Yelp']:
        args.lr = 2e-6
        args.epoch = 50


    print(dataset)

    main(args)
    print('-------------------------')

