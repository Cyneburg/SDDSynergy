import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import (GATConv,SAGPooling,global_add_pool)


class GAT_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats
        self.conv = GATConv(in_features, head_out_feats, n_heads)
        self.readout = SAGPooling(n_heads * head_out_feats, min_score=-1)

    def forward(self, x, edge_index, batch):
        x = self.conv(x, edge_index)
        att_x, att_edge_index, att_edge_attr, att_batch, att_perm, att_scores = self.readout(x, edge_index, batch=batch)
        global_graph_emb = global_add_pool(att_x, att_batch)
        return x, global_graph_emb

#co-attention mechanism
class CoAttentionLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features//2))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))
    
    def forward(self, receiver, attendant):
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q
        drug_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        attentions = torch.tanh(drug_activations) @ self.a
        return attentions

#substructure-pairs interaction
class drug_Interaction(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.M = nn.Parameter(torch.zeros(self.n_features,self.n_features))
        nn.init.xavier_uniform_(self.M)
    
    def forward(self, drug1, drug2, co_attention):
        drug1 = F.normalize(drug1, dim=-1)
        drug2 = F.normalize(drug2, dim=-1)
        scores = drug1 @ self.M @ drug2.permute(0,2,1)

        if co_attention is not None:
          scores = co_attention * scores
        scores = scores.reshape(scores.shape[0],scores.shape[1] * scores.shape[2])
        return scores 


class SelfAttentionLayer(nn.Module):
    def __init__(self, in_feature,head_num):
        super().__init__()
        self.feature = in_feature
        self.k = nn.Linear(in_feature, in_feature//head_num, bias=False)
        self.q = nn.Linear(in_feature, in_feature//head_num, bias=False)
        self.v = nn.Linear(in_feature, in_feature//head_num, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self,drug):
        drug_k = self.k(drug)
        drug_q = self.q(drug)
        drug_v = self.v(drug)
        scroe = torch.bmm(drug_k,drug_q.permute(0,2,1))
        attention = self.softmax(scroe)
        drug = torch.bmm(attention,drug_v)
        repr = F.max_pool2d(drug, (4,1)).reshape(drug.shape[0],drug.shape[2])
        return repr

#drug-cell line attention mechanism and combination action module
class Drug_Cell_In(nn.Module):
    def __init__(self, in_feature):
        super().__init__()
        self.feature = in_feature
        self.k = nn.Parameter(torch.zeros(in_feature, in_feature))
        self.q = nn.Parameter(torch.zeros(in_feature, in_feature))
        nn.init.xavier_uniform_(self.k)
        nn.init.xavier_uniform_(self.q)
        self.softmax = nn.Softmax(dim=2)

    def forward(self,drug,cell):
        d = F.normalize(drug, dim=-1)#128,4,256
        c = cell.reshape(cell.shape[0],1,cell.shape[1])#10,1,5
        c = F.normalize(c, dim=-1)

        drug_ = d @ self.k #128,4,256
        cell_ = c @ self.q #128,1,256
        drug_cell_interaction = torch.bmm(cell_,drug_.permute(0,2,1)) #128,1,4
        dc_attention = self.softmax(drug_cell_interaction)
        drug_cell_in = torch.mul(dc_attention,c @ d.permute(0,2,1)) #128,1,4
        drug_cell = drug_cell_in.reshape(drug_cell_in.shape[0],drug_cell_in.shape[2])
        return drug_cell
