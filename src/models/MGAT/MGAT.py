"""
MGAT: Multi-view Graph Attention Networks, Xie et al. (2020)
- https://www.sciencedirect.com/science/article/pii/S0893608020303105?via%3Dihub

The MGAT architecture is designed to handle multi-view graph data, where each view represents a different 
type of relationship or feature set for the same set of nodes. The model consists of two main components:
    1. Graph Embedding Component: For each view, a separate GATv2 layer is applied to learn view-specific node embeddings.
    2. Attention-based Aggregation: An attention mechanism is used to learn how to weight and combine the embeddings from 
    different views into a single global representation for each node.
    3. The regularization term Ω encourages the model to learn diverse embeddings across views, preventing them from collapsing
    into similar representations.

The provided implementation focuses on the core MGAT architecture, including the GATv2 layers for each view and the attention
mechanism for aggregating the multi-view embeddings. The regularization term Ω is also implemented to encourage diversity among
the view-specific embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv



class MultiView_GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_views, reg_lambda: float = 4e-3, heads=1, edge_dim=1, dropout=0.6):
        """
        Args:
            in_channels (int): Dimension des features d'entrée (F).
            out_channels (int): Dimension de l'embedding de sortie (F').
            num_views (int): Nombre de vues (K) dans le graphe multi-vues.
            heads (int): Nombre de têtes d'attention pour GATv2.
            dropout (float): Taux de dropout.
            edge_dim (int or None): Dimension des attributs d'arêtes. Si None ou <= 0, les edge_attr ne seront pas utilisés.
        """
        super(MultiView_GAT, self).__init__()
        # Ensure edge_dim is valid; None or <= 0 means no edge attributes
        if edge_dim is None or edge_dim <= 0:
            self.use_edge_attr = False
            edge_dim = None
        else:
            self.use_edge_attr = True
        
        self.num_views = num_views
        self.out_channels = out_channels
        self.dropout = dropout
        self.view_networks = nn.ModuleList()
        self.reg_lambda = reg_lambda
        self.edge_dim = edge_dim

        # 1. Graph Embedding Component
        for _ in range(num_views):
            layers = nn.ModuleList()
            # First layer
            layers.append(GATv2Conv(in_channels, out_channels=hidden_channels, heads=heads, concat=True, edge_dim=edge_dim, residual=True))
            
            # Hidden layers (num_layers - 2) with concatenation
            for _ in range(num_layers - 2):
                layers.append(GATv2Conv(hidden_channels * heads, out_channels=hidden_channels, heads=heads, concat=True, edge_dim=edge_dim, residual=True))
            
            # Last layer outputs out_channels (not multiplied by heads since concat=False)
            layers.append(GATv2Conv(hidden_channels * heads, out_channels=out_channels, heads=1, concat=False, edge_dim=edge_dim, residual=False))
            
            self.view_networks.append(layers)

        # Semantic attention (remains the same for all views, as per the paper)
        self.semantic_attention = nn.Parameter(torch.Tensor(num_views, num_views * out_channels))
        nn.init.xavier_uniform_(self.semantic_attention)
        


    def forward(self, x, edge_indices, edge_attrs=None):
        """
        Args:
            x (Tensor): Features des nœuds [N, in_channels].
            edge_indices (List[Tensor]): Liste de K tenseurs d'adjacence, un par vue. Chaque tenseur est de forme [2, E_k] pour la k-ième vue.
            edge_attrs (List[Tensor], optional): Liste de K tenseurs d'attributs d'arêtes, un par vue. Seulement utilisés si self.use_edge_attr=True.
        Returns:
            Tensor: Représentation globale des nœuds après agrégation multi-vues [N, out_channels].
        """
        view_embeddings = []

        for k in range(self.num_views):
            h = x
            # For every layer in the k-th view's GATv2 stack
            for i, conv in enumerate(self.view_networks[k]):
                # Only pass edge_attr if model is configured to use them
                if self.edge_dim is not None and self.edge_dim > 0 and edge_attrs is not None:
                    edge_attr_k = edge_attrs[k]
                else:
                    edge_attr_k = None
                h = conv(h, edge_indices[k], edge_attr=edge_attr_k)
                # If not the last layer, apply activation and dropout
                if i < len(self.view_networks[k]) - 1:
                    h = F.elu(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)
            view_embeddings.append(h)

        # --- Step 2: Attention-based aggregation (Eq. 5 & 6) ---
        # Construction of C_i = [x'_i1 || x'_i2 || ... || x'_iK] for each node i
        concat_features = torch.cat(view_embeddings, dim=1)
        
        # Calculation of attention coefficients beta_ik = t_k^T . c_i for each view k
        attention_logits = torch.matmul(concat_features, self.semantic_attention.t())

        # Apply Softmax to obtain normalized weights
        view_weights = F.softmax(attention_logits, dim=1)
        

        # --- Step 3: Compute the global representation (Eq. 6) ---
        # h_i = sum(beta_ik * x'_ik)
        # Stack embeddings for easier multiplication: [N, K, out_channels]
        stack_embeddings = torch.stack(view_embeddings, dim=1)
        
        # Reshape weights for broadcasting : [N, K, 1]
        view_weights_expanded = view_weights.unsqueeze(-1)
        
        # Weighted sum
        global_embedding = torch.sum(view_weights_expanded * stack_embeddings, dim=1)
        
        return global_embedding, view_weights
    

    def get_regularization_loss(self):
        """
        Calculate the regularization term Omega
        Ω = sum_{i,j} ||W_i - W_j||^2
        """
        reg_loss = 0.0
        # GATv2Conv stores its projection weights in 'lin_l' (left) and 'lin_r' (right).
        # The paper mentions a matrix W, here we use lin_l (source projection)
        # as the main approximation of W for the constraint. Use the first layer per
        # view to keep shapes consistent across views.
        
        weights = [self.view_networks[v][0].lin_l.weight for v in range(self.num_views)]
        
        for i in range(self.num_views):
            for j in range(self.num_views):
                if i != j:
                    # Squared L2 norm of the difference
                    diff = weights[i] - weights[j]
                    reg_loss += torch.norm(diff, p=2) ** 2
        
        return reg_loss
    

    def get_total_loss(self, pred, target, criterion):
        """
        Calculate the total loss combining the main criterion and the regularization term.
        Total Loss = Main Loss + λ * Ω

        Args:
            pred (Tensor): Predictions from the model.
            target (Tensor): Ground truth labels.
            criterion (nn.Module): Loss function for the main task (e.g., BCEWithLogitsLoss).
        Returns:
            Tensor: Total loss combining main loss and regularization.
        """
        main_loss = criterion(pred, target)
        reg_loss = self.get_regularization_loss()
        total_loss = main_loss + self.reg_lambda * reg_loss
        return total_loss