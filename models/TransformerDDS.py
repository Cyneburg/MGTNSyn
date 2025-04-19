import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GraphTransformer import GraphTransformer
from torch_geometric.nn import LayerNorm

# MGTNSyn  model
class MGTNSyn(torch.nn.Module):
    def __init__(self, num_features_xd=79, n_output=2, num_features_xt=954, output_dim=128, k_hop=4, dropout=0.2):
        super(MGTNSyn, self).__init__()

        self.transformer = GraphTransformer(num_features_xd, output_dim, num_heads=4, k_hop=k_hop, num_layers=2,
                                            dropout=dropout, gnn_type='graphsage', se="gnn", global_pool='max', num_edge_features=6)
        self.drug1_fc_g1 = nn.Linear(output_dim, output_dim * 2)

        # combined layers
        self.fc1 = nn.Linear(1466, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(1466)
        self.output_dim = output_dim

    def forward(self, data1, data2):
        x1_sub = self.transformer(data1)
        x1_sub = self.drug1_fc_g1(x1_sub)
        drug1_cell = self.relu(x1_sub)

        x2_sub = self.transformer(data2)
        x2_sub = self.drug1_fc_g1(x2_sub)
        drug2_cell = self.relu(x2_sub)


        # deal cell
        cell_vector = data1.cell
      
        # concat
        xc = torch.cat((drug1_cell, drug2_cell, cell_vector), 1)
        xc = self.batchnorm(xc)

        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)

        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)

        xc = self.fc3(xc)
        xc = self.relu(xc)

        out = self.out(xc)
        return out
