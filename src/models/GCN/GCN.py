import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.nn.functional as F

class PyGGCNModel(nn.Module):
    """
    Graph Convolutional NN: Composed of two layers with a 0.5 dropout

    The two layers allow each node to aggregate the data of the neigbours of its neighbours, this permit a very local and spatial 
    consideration of the Amino Acides
    """
    def __init__(self, n_features: int,  n_edge_attr: int, n_hidden: int, dropout: float = 0.5):
        super(PyGGCNModel, self).__init__()
        self.n_edge_attr = n_edge_attr

        # Standardisation des features:
        self.ln_input = nn.LayerNorm(n_features)
        
        self.conv1 = GCNConv(n_features, n_hidden)
        # Output with 1 logit
        self.conv2 = GCNConv(n_hidden, 1)
        
        # Añadir un dropout de 0.5
        self.dropout = nn.Dropout(p = dropout)

    
    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Capa 1
        if self.n_edge_attr == 1:
            x = self.conv1(x, edge_index, edge_weight=edge_attr)
        else:
            x = self.conv1(x, edge_index)
        
        x = F.relu(x)
        x = self.dropout(x)

        # Capa 2
        if self.n_edge_attr == 1:
            x = self.conv2(x, edge_index, edge_weight=edge_attr)
        else:
            x = self.conv2(x, edge_index)
        
        return x