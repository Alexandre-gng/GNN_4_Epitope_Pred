import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_undirected
from torch_scatter import scatter

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class EGNNLayer(MessagePassing):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, edge_dim: int = 1, dropout: float = 0.0,residual: bool = True, act=None):
        super().__init__(aggr='add') 
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.act = act if act is not None else nn.SiLU()
        self.residual = residual

        # 1. φ_e : Calcul du message m_ij (Eq. 3)
        edge_input_dim = 2 * in_dim + 1 + (edge_dim if edge_dim else 0)
        self.edge_func = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, hidden_dim),
            self.act
        )
        
        # 2. MÉCANISME D'ATTENTION
        # Calcule un score scalaire à partir des mêmes entrées que le message
        self.att_func = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # Pour borner l'importance entre 0 et 1
        )
        
        # 3. φ_x : Mise à jour des coordonnées (Eq. 4)
        self.coord_func = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, 1, bias=False) # Scalaire pour le vecteur (x_i - x_j)
        )
        
        # 4. φ_h : Mise à jour des nœuds (Eq. 6)
        self.node_func = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, h, x, edge_index, edge_attr=None):
        # Propagation PyG
        # On passe x pour calculer les distances dans message()
        h_new, x_new = self.propagate(edge_index, h=h, x=x, edge_attr=edge_attr)
        
        # Mise à jour finale des nœuds (Eq. 6)
        h_final = self.node_func(torch.cat([h, h_new], dim=-1))
        
        # Optionnel : Ajout d'une connexion résiduelle
        if self.residual and self.in_dim == self.out_dim:
            h_final = h_final + h
            
        return h_final, x_new

    def message(self, h_i, h_j, x_i, x_j, edge_attr, index, ptr, size_i):
        # Calcul de la distance euclidienne au carré (Invariant)
        rel_x = x_i - x_j
        dist_sq = torch.sum(rel_x**2, dim=-1, keepdim=True)
        
        # Construction de l'entrée du message (Eq. 3)
        input_list = [h_i, h_j, dist_sq]
        if edge_attr is not None:
            input_list.append(edge_attr)
        
        # Calcul du message m_ij
        m_ij = self.edge_func(torch.cat(input_list, dim=-1))
        
        # --- CALCUL DE L'ATTENTION ---
        # On calcule les scores et on utilise softmax sur les voisins de i
        alpha_ij = self.att_func(m_ij)
        alpha_ij = softmax(alpha_ij, index, ptr, size_i) 
        
        # Pondération du message feature
        m_ij_weighted = alpha_ij * m_ij
        
        # --- MISE À JOUR DES COORDONNÉES (Eq. 4) ---
        # On applique aussi l'attention sur la force de déplacement
        coord_weights = self.coord_func(m_ij)
        displacement = rel_x * coord_weights * alpha_ij 
        
        # On retourne les deux informations à agréger
        return m_ij_weighted, displacement

    def aggregate(self, inputs, index, dim_size=None):
        # inputs contient (m_ij_weighted, displacement)
        m_ij_weighted, displacement = inputs
        
        # Somme des messages et des déplacements (Eq. 4 & 5)
        m_i = torch.zeros(dim_size, self.hidden_dim, device=m_ij_weighted.device)
        m_i.index_add_(0, index, m_ij_weighted)
        
        sum_displacement = torch.zeros(dim_size, displacement.shape[-1], device=displacement.device)
        sum_displacement.index_add_(0, index, displacement)
        
        return m_i, sum_displacement

    def update(self, aggr_out, x):
        m_i, sum_displacement = aggr_out
        
        # Nouvelle position (Eq. 4)
        x_new = x + sum_displacement
        
        return m_i, x_new





class EGNN(torch.nn.Module):
    """
    Equivariant Graph Neural Network (EGNN) model that stacks multiple EGNN layers.
    
    This model maintains equivariance to rotations and translations of the input coordinates
    while updating both node features and spatial positions through the network.
    
    Args:
        num_layers: Number of EGNN layers to stack
        in_dim: Input feature dimension for each node
        hidden_dim: Hidden feature dimension for message passing and node updates
        out_dim: Output feature dimension for node features
        edge_dim: Dimension of edge attributes (optional)
        dropout: Dropout probability for regularization
        radial_layers: Number of layers in radial/edge functions
    """
    
    def __init__(self, num_layers: int, in_dim: int, hidden_dim: int, out_dim: int, edge_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        
        self.num_layers = num_layers
        self.out_dim = out_dim
        
        # Create a list of EGNN layers
        self.layers = torch.nn.ModuleList()
        
        # First layer takes input dimension and outputs hidden dimension
        # No residual connection for first layer since in_dim != hidden_dim
        self.layers.append(EGNNLayer(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, edge_dim=edge_dim, dropout=dropout, residual=False))
        
        # Intermediate layers with hidden dimension
        for _ in range(num_layers - 2):
            self.layers.append(EGNNLayer(in_dim=hidden_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, edge_dim=edge_dim, dropout=dropout, residual=True))
        
        # Final layer outputs to hidden dimension (not out_dim)
        self.layers.append(EGNNLayer(in_dim=hidden_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, edge_dim=edge_dim, dropout=dropout, residual=False))
        
        # Classification head: map from hidden_dim to 1 (for binary classification)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    

    def forward(self, x: torch.Tensor, coords: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the EGNN model.
        
        Args:
            x: Node feature tensor [num_nodes, in_dim]
            coords: Node coordinate tensor [num_nodes, 3] (or any spatial dimension)
            edge_index: Graph connectivity in COO format [2, num_edges]
            edge_attr: Optional edge attributes [num_edges, edge_dim]
        
        Returns:
            Tuple of:
            - Predicted labels [num_nodes, 1]
            - Updated coordinates [num_nodes, 3]
        """
        for layer in self.layers:
            x, coords = layer(x, coords, edge_index, edge_attr)
        
        # Apply classification head
        logits = self.classifier(x)
        
        return logits, coords