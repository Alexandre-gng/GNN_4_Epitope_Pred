import torch
import torch.nn as nn
import torch.nn.functional as F

from GAT.GAT import GATv2Layer, GATv2Net
from MGAT.MGAT_func import load_branch_weights



class SemanticAttention(nn.Module):
    def __init__(self, in_channels, hidden_size=128):
        super(SemanticAttention, self).__init__()
        # Transformation linéaire pour projeter les embeddings
        self.project = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z_list):
        """
        z_list: Liste de Tenseurs [N, out_channels], un par vue
        """
        # 1. On stacke les vues : [N, K, out_channels]
        z = torch.stack(z_list, dim=1) 
        
        # 2. Calcul des scores w_i^k : [N, K, 1]
        s = self.project(z) 
        
        # 3. Moyenne sur les nœuds (N) pour avoir le poids global de la vue : [K, 1]
        weights = s.mean(dim=0)
        
        # 4. Softmax pour avoir les bêta_k : [K, 1]
        beta = torch.softmax(weights, dim=0)
        
        # 5. Pondération des vues et somme finale
        # On multiplie chaque vue par son bêta correspondant
        final_emb = (z * beta.transpose(0, 1)).sum(dim=1)
        
        return final_emb, beta





class GATv2Branch(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int=8, heads: int=4, concat: bool=True, residual: bool=True, dropout: float=0.3, edge_dim: int = None):
        super().__init__()
        
        # 1. Node Encoder (Idem à ton GATv2Net)
        self.node_encoder = torch.nn.Linear(input_dim, hidden_dim)
    
        out_head = max(1, hidden_dim // heads)
        self.layers = torch.nn.ModuleList()
        gat_out_dim = out_head * (heads if concat else 1)
        
        # 2. Stack de couches GATv2 (0 à 7 selon ton state_dict)
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else gat_out_dim
            self.layers.append(GATv2Layer(
                in_dim=in_dim, out_dim=out_head, heads=heads, 
                concat=concat, residual=residual, dropout=dropout, edge_dim=edge_dim
            ))

        # Note: On ne définit pas self.classifier ici car on veut l'embedding final
        # La dimension de sortie de cette branche est: gat_out_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None):
        # Étape 1: Encoding
        x = self.node_encoder(x)
        
        # Étape 2: Passage dans les 8 couches GATv2
        for layer in self.layers:
            # Gestion auto du mismatch edge_attr (ton code d'origine)
            current_edge_attr = edge_attr if (edge_attr is not None and edge_attr.size(0) == edge_index.size(1)) else None
            
            x, _, _ = layer(x, edge_index, edge_attr=current_edge_attr)
        
        # On retourne l'embedding X (taille [N, gat_out_dim])
        return x





class MultiViewGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, nb_classes, weights_paths):
        super().__init__()
        
        # Création des branches (une par vue)
        # Supposons que weights_paths est une liste ['path_v1.pt', 'path_v2.pt']
        self.branches = torch.nn.ModuleList()
        for path in weights_paths:
            branch = GATv2Branch(input_dim, hidden_dim)
            load_branch_weights(branch, path) # On charge les poids pré-entraînés ici
            self.branches.append(branch)
        
        # Attention Sémantique (formules détaillées plus haut)
        # L'input_dim ici est la sortie de la dernière couche GAT (gat_out_dim)
        self.semantic_attention = SemanticAttention(in_channels=hidden_dim) 
        
        # Classifier final (Le SEUL qui part de zéro)
        self.classifier = torch.nn.Linear(hidden_dim, nb_classes)

    def forward(self, x, adjs):
        # adjs est une liste de edge_index [edge_index_v1, edge_index_v2, ...]
        embeddings = [branch(x, adjs[i]) for i, branch in enumerate(self.branches)]
        
        # Fusion
        z_fused, beta = self.semantic_attention(embeddings)
        
        # Pred
        return self.classifier(z_fused), beta