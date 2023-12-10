"""Defines all graph embedding models"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from torch_geometric.nn import GATConv, GATv2Conv
from utils import utils

import numpy as np
from scipy.special import log1p

import sys

'''
    문제상황(23.10.08)
    1. cosine similarity은 두 값 간의 유사도임. 때문에 GED가 정규화 되지 않았으므로 잘 예측할 수 없음
    2. cosine sim 에서 1은 동일함이지만 normed GED가 1이면 두 값 간의 차이가 큰 것임
    3. GED 예측이 목적이 아니라, 그래프를 feature space에 잘 embedding하는 것이 목적임
    
    해결 방안
    1. 두 그래프 특징값의 차이를 의미하는 cosine similarity 값에 -log(1-x) 를 취해, 차이가 클수록 더 큰 값이 나오고, 적을 수록 더 작은 값이 나오도록 변경    

'''


class BaselineMLP(nn.Module):
    # GNN -> concat -> MLP graph classification baseline
    def __init__(self, input_dim, hidden_dim, args):
        super(BaselineMLP, self).__init__()
        self.emb_model = SkipLastGNN(input_dim, hidden_dim, hidden_dim, args)
        
        self.mlp = nn.Sequential(nn.Linear(2 * hidden_dim, 256), nn.ReLU(),
                                 nn.Linear(256, 2))

    def forward(self, emb_motif, emb_motif_mod):
        pred = self.mlp(torch.cat((emb_motif, emb_motif_mod), dim=1))
        pred = F.log_softmax(pred, dim=1)
        return pred

    # def predict(self, pred):
        # return pred  # .argmax(dim=1)

    # def criterion(self, pred, _, label):
    #     return F.nll_loss(pred, label) #todo MAEloss


class GnnEmbedder(nn.Module):
    # Gnn embedder model -- contains a graph embedding model `emb_model`
    def __init__(self, input_dim, hidden_dim, args):
        super(GnnEmbedder, self).__init__()
        self.emb_model = SkipLastGNN(input_dim, hidden_dim, hidden_dim, args)
        # self.emb_model = GATv2Conv(input_dim, hidden_dim, hidden_dim, args)
        self.margin = args.margin
        self.use_intersection = False

        self.clf_model = nn.Sequential(nn.Linear(1, 1))    

    def forward(self, emb_as, emb_bs):
        return emb_as, emb_bs

    def predict(self, pred):
        """Predict graph edit distance(ged) of graph pairs, where emb_as, emb_bs = pred.
        pred: list (emb_as, emb_bs) of embeddings of graph pairs.
        Returns: list of ged of graph pairs.
        """
        emb_as, emb_bs = pred
        # s = torch.tensor([torch.dot(emb_as[i], emb_bs[i]) for i in range(len(emb_as))],requires_grad=True).to(utils.get_device())
        sim = F.cosine_similarity(emb_as, emb_bs) 
        # sim = -torch.log(1 - sim) #nan 발생
        sim = (1-sim)
        
        
        return sim

    def criterion(self, pred, intersect_embs, labels):
        """Loss function for emb.
        The e term is the predicted ged of graph pairs.

        pred: lists of embeddings outputted by forward
        intersect_embs: not used
        labels: labels for each entry in pred
        """        
        emb_as, emb_bs = pred
        # s = torch.tensor([torch.dot(emb_as[i], emb_bs[i]) for i in range(len(emb_as))],requires_grad=True).to(utils.get_device()) 
        # # <- 단순한 dot production
        sim = F.cosine_similarity(emb_as, emb_bs) 
        # 1-x 값을 0과 1 사이로 클램핑
        # clamped_x = torch.clamp(1 - sim, min=1e-7, max=1-1e-7)
        # sim = -torch.log(1 - sim) #nan 발생
        sim = (1-sim)
        

        loss_func = nn.L1Loss() # MAE
        loss = loss_func(sim, labels)
        return loss


class SkipLastGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):  # 1, 64, 64
        super(SkipLastGNN, self).__init__()
        self.dropout = args.dropout
        self.n_layers = args.n_layers

        '''
        pre MLP
        '''
        self.pre_mp = nn.Sequential(nn.Linear(input_dim, hidden_dim))
        '''
        GCN
        '''     
        conv_model = self.build_conv_model(args.conv_type, 1)  # GAT
        self.convs = nn.ModuleList()  # nn.Module을 리스트로 정리하는 방법, 파라미터는 리스트
        
        if args.skip == 'learnable':
            self.learnable_skip = nn.Parameter(torch.ones(self.n_layers,
                                                          self.n_layers))
        for l in range(args.n_layers):
            if args.skip == 'all' or args.skip == 'learnable':
                hidden_input_dim = hidden_dim * (l + 1)
            else:
                hidden_input_dim = hidden_dim
            if args.conv_type == "GAT": #GATv2Conv
                self.convs.append(conv_model(hidden_input_dim, hidden_dim, edge_dim = args.edge_attr_dim ))
            else:
                self.convs.append(conv_model(hidden_input_dim, hidden_dim))
        '''
        post MLP
        '''
        post_input_dim = hidden_dim * (args.n_layers + 1)  # 

        self.post_mp = nn.Sequential(
            nn.Linear(post_input_dim, hidden_dim),  # 
            nn.Dropout(args.dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, output_dim),      # 
            nn.ReLU(),
            nn.Linear(hidden_dim, 256), nn.ReLU(),  # 
            nn.Linear(256, hidden_dim))             # 
        self.skip = args.skip   # True
        self.conv_type = args.conv_type     # order

    def build_conv_model(self, model_type, n_inner_layers):
        if model_type == "GCN":
            return pyg_nn.GCNConv
    
        elif model_type == "SAGE":
            return SAGEConv

        elif model_type == "GAT":
            return pyg_nn.GATv2Conv
        else:
            print("unrecognized model type")
            
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.node_feature, data.edge_index, data.edge_feature, data.batch
        x = self.pre_mp(x) 

        all_emb = x.unsqueeze(1)    
        emb = x                     
        for i in range(len(self.convs_sum) if self.conv_type == "PNA" else    # i -> 0 ~ 7
                       len(self.convs)):
            if self.skip == 'learnable':
                skip_vals = self.learnable_skip[i, :i+1].unsqueeze(0).unsqueeze(-1)
                curr_emb = all_emb * torch.sigmoid(skip_vals)
                curr_emb = curr_emb.view(x.size(0), -1)         # 539 x 64
                if self.conv_type == "PNA":
                    x = torch.cat((self.convs_sum[i](curr_emb, edge_index),
                                   self.convs_mean[i](curr_emb, edge_index),
                                   self.convs_max[i](curr_emb, edge_index)), dim=-1)
                    
                elif self.conv_type == "GAT":
                    # print("@@GAT@@")
                    # print("edge_feature :" ,edge_attr)
                    # sys.exit()
                    x = self.convs[i](curr_emb, edge_index, edge_attr = edge_attr)
                else:
                    x = self.convs[i](curr_emb, edge_index)

            elif self.skip == 'all':
                if self.conv_type == "PNA":
                    x = torch.cat((self.convs_sum[i](emb, edge_index),
                                   self.convs_mean[i](emb, edge_index),
                                   self.convs_max[i](emb, edge_index)), dim=-1)
                
                elif self.conv_type == "GAT":
                    # print("@@GAT@@")
                    x = self.convs[i](curr_emb, edge_index, edge_attr)
                else:
                    x = self.convs[i](emb, edge_index)
            else:
                x = self.convs[i](x, edge_index)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            emb = torch.cat((emb, x), 1) 

            if self.skip == 'learnable':                
                # print("all_emb: " , all_emb.size())
                all_emb = torch.cat((all_emb, x.unsqueeze(1)), 1)
        emb = pyg_nn.global_add_pool(emb, batch)
        emb = self.post_mp(emb)

        return emb
    def loss(self, pred, label):
        loss = F.MAELoss(pred, label)
        print("Skip Last GNN - s : ", loss)
        return loss