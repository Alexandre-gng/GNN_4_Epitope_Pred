import torch.nn as nn
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """Single hidden GCN layer with ReLU and dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.5):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class PyGGCNModel(nn.Module):
    """
    Graph Convolutional NN: Composed of two layers with a 0.5 dropout

    The two layers allow each node to aggregate the data of the neigbours of its neighbours, this permit a very local and spatial 
    consideration of the Amino Acides
    """
    def __init__(self, n_features: int, n_edge_attr: int, n_hidden: int, n_layers: int, dropout: float = 0.5):
        super(PyGGCNModel, self).__init__()
        if n_layers < 2:
            raise ValueError("n_layers must be >= 2 for a GCN with an output layer")

        self.n_edge_attr = n_edge_attr
        self.n_layers = n_layers

        # Standardisation des features:
        self.ln_input = nn.LayerNorm(n_features)

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(GCNLayer(n_features, n_hidden, dropout=dropout))
        for _ in range(n_layers - 2):
            self.hidden_layers.append(GCNLayer(n_hidden, n_hidden, dropout=dropout))

        # Output with 1 logit (no activation on final layer)
        self.output_layer = GCNConv(n_hidden, 1)


    
    def _get_node_features(self, data: Data):
        x = getattr(data, "x", None)
        if x is not None:
            return x

        if 'node_attrs' in data:
            node_attrs = data['node_attrs']
            if hasattr(node_attrs, "shape"):
                return node_attrs
            if callable(node_attrs):
                owner = getattr(node_attrs, "__self__", None)
                if owner is not None and 'node_attrs' in owner:
                    owner_node_attrs = owner['node_attrs']
                    if hasattr(owner_node_attrs, "shape"):
                        return owner_node_attrs

        raise ValueError(
            f"No valid node feature tensor found in data object. "
            f"Available keys: {list(data.keys())}"
        )

    def forward(self, data: Data):
        x = self._get_node_features(data)
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        x = x.float()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # GCNConv expects 1-D edge_weight [num_edges]; squeeze if stored as [num_edges, 1]
        if edge_attr is not None and edge_attr.dim() == 2 and edge_attr.size(-1) == 1:
            edge_attr = edge_attr.squeeze(-1)

        if edge_attr is not None:
            edge_attr = edge_attr.float()
            edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=1.0, neginf=0.0)

        # Standardisation des features
        x = self.ln_input(x)

        use_edge_weight = self.n_edge_attr == 1 and edge_attr is not None

        # Hidden layers
        for layer in self.hidden_layers:
            if use_edge_weight:
                x = layer(x, edge_index, edge_weight=edge_attr)
            else:
                x = layer(x, edge_index)

        # Output layer
        if self.n_edge_attr == 1 and edge_attr is not None:
            x = self.output_layer(x, edge_index, edge_weight=edge_attr)
        else:
            x = self.output_layer(x, edge_index)

        return x