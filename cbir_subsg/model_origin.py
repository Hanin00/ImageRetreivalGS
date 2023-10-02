"""Defines all graph embedding models"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from torch_geometric.nn import GATConv, GATv2Conv
from utils import utils

import sys


class BaselineMLP(nn.Module):
    # GNN -> concat -> MLP graph classification baseline
    def __init__(self, input_dim, hidden_dim, args):
        super(BaselineMLP, self).__init__()
        self.emb_model = SkipLastGNN(input_dim, hidden_dim, hidden_dim, args)
        # self.emb_model = GATv2Conv(input_dim, hidden_dim, hidden_dim, args)
        
        self.mlp = nn.Sequential(nn.Linear(2 * hidden_dim, 256), nn.ReLU(),
                                 nn.Linear(256, 2))

    def forward(self, emb_motif, emb_motif_mod):
        pred = self.mlp(torch.cat((emb_motif, emb_motif_mod), dim=1))
        pred = F.log_softmax(pred, dim=1)
        return pred

    def predict(self, pred):
        return pred  # .argmax(dim=1)

    def criterion(self, pred, _, label):
        return F.nll_loss(pred, label)


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
        s = torch.tensor([torch.dot(emb_as[i], emb_bs[i]) for i in range(len(emb_as))],requires_grad=True).to(utils.get_device())
        print("s : ", s)

        return s

    def criterion(self, pred, intersect_embs, labels):
        """Loss function for emb.
        The e term is the predicted ged of graph pairs.

        pred: lists of embeddings outputted by forward
        intersect_embs: not used
        labels: labels for each entry in pred
        """
        # e = torch.sum(torch.abs(emb_bs - emb_as), dim=1)
        emb_as, emb_bs = pred
        # s = F.cosine_similarity(emb_as, emb_bs)
        # s = torch.dot(emb_as, emb_bs)
        s = torch.tensor([torch.dot(emb_as[i], emb_bs[i]) for i in range(len(emb_as))],requires_grad=True).to(utils.get_device())
        # print("s : ", s)        
        loss_func = nn.MSELoss()
        loss = loss_func(s, labels)

        return loss


#Layer stack
class SkipLastGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):  # 1, 64, 64
        # print("SkipLastGNN: ",hidden_dim)
        super(SkipLastGNN, self).__init__()
        self.dropout = args.dropout
        self.n_layers = args.n_layers

        '''
        pre MLP
        '''
        # Linear(1, 64)
        self.pre_mp = nn.Sequential(nn.Linear(input_dim, hidden_dim))
        '''
        GCN
        '''
        conv_model = self.build_conv_model(args.conv_type, 1)  # GAT
        if args.conv_type == "PNA":
            self.convs_sum = nn.ModuleList()
            self.convs_mean = nn.ModuleList()
            self.convs_max = nn.ModuleList()
        else:
            self.convs = nn.ModuleList()  # nn.Module을 리스트로 정리하는 방법, 파라미터는 리스트

        # learnable_skip = ones(8,8)
        # nn.Parameter : 모듈의 파라미터들을 iterator로 반환
        if args.skip == 'learnable':
            self.learnable_skip = nn.Parameter(torch.ones(self.n_layers,
                                                          self.n_layers))

        for l in range(args.n_layers):
            if args.skip == 'all' or args.skip == 'learnable':
                hidden_input_dim = hidden_dim * (l + 1)
            else:
                hidden_input_dim = hidden_dim
            if args.conv_type == "GAT": #GATv2Conv
                # self.convs.append(conv_model(hidden_input_dim, hidden_dim))
                #https://github.com/tech-srl/how_attentive_are_gats/blob/main/gatv2_conv_DGL.py
                self.convs.append(conv_model(input_dim, hidden_dim, args.edge_attr_dim ))
            else:
                self.convs.append(conv_model(hidden_input_dim, hidden_dim))
        '''
        post MLP
        '''
        post_input_dim = hidden_dim * (args.n_layers + 1)  # 64 * 9
        if args.conv_type == "PNA":
            post_input_dim *= 3

        self.post_mp = nn.Sequential(
            nn.Linear(post_input_dim, hidden_dim),  # 64 * 9, 64
            nn.Dropout(args.dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, output_dim),      # 64, 64
            nn.ReLU(),
            nn.Linear(hidden_dim, 256), nn.ReLU(),  # 64 256
            nn.Linear(256, hidden_dim))             # 265 64
        #self.batch_norm = nn.BatchNorm1d(output_dim, eps=1e-5, momentum=0.1)
        self.skip = args.skip   # True
        self.conv_type = args.conv_type     # order

    def build_conv_model(self, model_type, n_inner_layers):
        if model_type == "GCN":
            return pyg_nn.GCNConv
    
        elif model_type == "SAGE":
            # print("@@GAT@@")
            return SAGEConv

        elif model_type == "GAT":
            print("@@GAT@@")
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
                if self.conv_type == "GAT":
                    x = self.convs[i](curr_emb, edge_index)
                else:
                    x = self.convs[i](curr_emb, edge_index)

            elif self.skip == 'all':
                if self.conv_type == "PNA":
                    x = torch.cat((self.convs_sum[i](emb, edge_index),
                                   self.convs_mean[i](emb, edge_index),
                                   self.convs_max[i](emb, edge_index)), dim=-1)
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
        return F.MSELoss(pred, label)



class SAGEConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, aggr="add"):
        super(SAGEConv, self).__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_update = nn.Linear(out_channels + in_channels,
                                    out_channels)

    def forward(self, x, edge_index, edge_weight=None, size=None,
                res_n_id=None):
        """
        Args:
            res_n_id (Tensor, optional): Residual node indices coming from
                :obj:`DataFlow` generated by :obj:`NeighborSampler` are used to
                select central node features in :obj:`x`.
                Required if operating in a bipartite graph and :obj:`concat` is
                :obj:`True`. (default: :obj:`None`)
        """
        # edge_index, edge_weight = add_remaining_self_loops(
        #    edge_index, edge_weight, 1, x.size(self.node_dim))
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        return self.propagate(edge_index, size=size, x=x,
                              edge_weight=edge_weight, res_n_id=res_n_id)

    def message(self, x_j, edge_weight):
        # return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        return self.lin(x_j)

    def update(self, aggr_out, x, res_n_id):
        aggr_out = torch.cat([aggr_out, x], dim=-1)
        aggr_out = self.lin_update(aggr_out)

        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

