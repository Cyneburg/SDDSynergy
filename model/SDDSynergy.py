import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import (GAT_Block, CoAttentionLayer, drug_Interaction, Drug_Cell_In)
from torch_geometric.nn import LayerNorm
import pandas as pd

# SDDS model
class SDDSynergyNet(torch.nn.Module):
    def __init__(self,num_features_xd=78, n_head=4, num_features_xt=954, output_dim=128, dropout=0.2, n_gats=6):
        super(SDDSynergyNet, self).__init__()

        '''
        num_features_xd: the feature dimension of drugs;
        n_head: the number of GATs attention heads;
        num_features_xt : the feature dimension of cancer cell lines;
        output_dim: the GATs hidden units, namely the embedding dimension of drug atoms;
        n_gats: the number of GATs layer, namely the range of substructures.
        '''

        # initial normal
        self.initial_norm = LayerNorm(num_features_xd)
        # graph drug convolution and drug layerNorm
        self.drug_gats = []
        self.drug_norm = []
        for i in range(n_gats):
            drug_gat = GAT_Block(n_head, num_features_xd, output_dim)
            self.add_module(f"drug_gat{i}", drug_gat)
            self.drug_gats.append(drug_gat)
            self.drug_norm.append(LayerNorm(output_dim * n_head).cuda())
            num_features_xd = output_dim * n_head
        self.drug_fc = nn.Linear(78,256)

        # drug interaction
        self.co_attention = CoAttentionLayer(n_head * output_dim)
        self.interaction = drug_Interaction(n_head * output_dim)

        # DL cell featrues
        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim * n_head),
            nn.ReLU()
        )

        #drug cell interaction
        self.drug_cell = Drug_Cell_In(output_dim * n_head)

        # combined layers
        self.fc1 = nn.Linear(n_gats**2+2*n_gats, 128)
        self.fc2 = nn.Linear(128, 2)

        # activation and regularization
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)
        self.output_dim = output_dim

    def save_num(self, d, path):
        d = d.cpu().numpy()
        ind = self.get_col_index(d)
        ind = pd.DataFrame(ind)
        ind.to_csv('data/case_study/' + path + '_index.csv', header=0, index=False)


    def forward(self, x1,x2, edge_index1,edge_index2, batch1,batch2, cell):
        # deal drug
        repr_drug1 = []
        repr_drug2 = []
        x1 = self.initial_norm(x1, batch1)
        x2 = self.initial_norm(x2, batch2)
        for i, drug_gat in enumerate(self.drug_gats):
            drug1 = drug_gat(x1, edge_index1, batch1)
            drug2 = drug_gat(x2, edge_index2, batch2)
            h_1 = drug1[0]  # x
            r_1 = drug1[1]  # emb
            h_2 = drug2[0]  # x
            r_2 = drug2[1]  # emb
            repr_drug1.append(r_1)
            repr_drug2.append(r_2)
            h1 = self.drug_norm[i](h_1, batch1)
            h2 = self.drug_norm[i](h_2, batch2)
            x1 = self.elu(h1)
            x2 = self.elu(h2)


        repr_drug1 = torch.stack(repr_drug1, dim=-2)  # [128,4,256]
        repr_drug2 = torch.stack(repr_drug2, dim=-2)  # [128,4,256]

        # deal cell
        cell = F.normalize(cell, 2, 1)
        cell_vector = self.reduction(cell)

        #co-attention and DDI
        co_attention = self.co_attention(repr_drug1,repr_drug2)
        drug_interaction = self.interaction(repr_drug1,repr_drug2,co_attention)


        #drug cell interaction
        drug1_cell = self.drug_cell(repr_drug1, cell_vector)
        drug2_cell = self.drug_cell(repr_drug2, cell_vector)
        drug_cell = torch.cat((drug1_cell,drug2_cell),dim=1)
        # print(drug_cell.shape)

        # concat
        xc = torch.cat((drug_interaction, drug_cell), 1)
        xc = F.normalize(xc, 2, 1)

        # add some FC layers
        xc = self.fc1(xc)
        xc = self.elu(xc)
        # xc = self.dropout(xc)
        out = self.fc2(xc)
        return out

