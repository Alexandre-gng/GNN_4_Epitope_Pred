import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np


# Détecter le device disponible
def get_device(use_gpu=True):
    """Détecte et retourne le device (GPU ou CPU)."""
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU détecté: {torch.cuda.get_device_name(0)}")
        print(f"CUDA disponible: {torch.cuda.is_available()}")
    else:
        device = torch.device("cpu")
        print("Utilisation du CPU")
    return device


def apply_knn_to_graphs(data_list, k, include_self=False, device=None):
    """
    Transforme une liste de graphes en utilisant une approche KNN pour les connexions.
    
    Chaque acide aminé est connecté uniquement aux K acides aminés les plus proches
    basé sur leurs positions 3D spatiales.
    
    Args:
        data_list: Liste de structures Data (PyTorch Geometric)
                   Chaque Data doit avoir:
                   - x: features des nœuds [num_nodes, num_features]
                   - pos: positions 3D [num_nodes, 3]
                   - y: labels [num_nodes]
                   - rsa: surface accessibility [num_nodes]
                   - name: identifiant de la structure
        k (int): Nombre de voisins les plus proches à considérer
        include_self (bool): Si True, inclut le nœud lui-même dans les voisins (default: False)
        device: torch.device ou str ('cuda' ou 'cpu'). Si None, détecte automatiquement
    
    Returns:
        Liste de structures Data avec edge_index et edge_attr recompilés selon KNN
    
    Example:
        >>> data_list = [Data(...), ...]
        >>> knn_data_list = apply_knn_to_graphs(data_list, k=15, device='cuda')
    """
    
    # Déterminer le device
    if device is None:
        device = get_device(use_gpu=True)
    elif isinstance(device, str):
        device = torch.device(device)
    
    knn_data_list = []
    
    for data in data_list:
        # Récupérer les positions 3D et les déplacer sur le device
        if torch.is_tensor(data.pos):
            pos = data.pos.to(device)
        else:
            pos = torch.tensor(data.pos, dtype=torch.float32, device=device)
        
        # Calculer la matrice de distances avec torch.cdist sur le GPU/CPU
        distances = torch.cdist(pos, pos)  # Shape: [num_nodes, num_nodes]
        
        # Obtenir les K voisins les plus proches (+ 1 pour inclure le nœud lui-même)
        k_neighbors = k + 1
        distances_topk, indices = torch.topk(distances, k=k_neighbors, dim=1, largest=False)
        
        # Créer edge_index à partir des indices KNN
        edge_index = []
        edge_attr = []
        
        for node_idx in range(indices.shape[0]):
            neighbors_idx = indices[node_idx].cpu().tolist()
            neighbors_dist = distances_topk[node_idx].cpu().tolist()
            
            for neighbor_idx, dist in zip(neighbors_idx, neighbors_dist):
                # Exclure la self-loop si demandé
                if not include_self and node_idx == neighbor_idx:
                    continue
                
                edge_index.append([node_idx, neighbor_idx])
                edge_attr.append([dist])
        
        # Convertir en tensors PyTorch
        # Important: Créer les tenseurs sur CPU pour garantir la compatibilité
        # Les données seront transférées vers le device approprié par le DataLoader
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().cpu()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32).cpu()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float32)
        
        # S'assurer que tous les attributs sont sur CPU pour la cohérence
        # Le DataLoader s'occupera de les transférer vers le device approprié
        x_cpu = data.x.cpu() if torch.is_tensor(data.x) else data.x
        y_cpu = data.y.cpu() if torch.is_tensor(data.y) else data.y
        pos_cpu = data.pos.cpu() if torch.is_tensor(data.pos) else data.pos
        
        # Créer une nouvelle structure Data avec les nouveaux edges
        new_data = Data(
            x=x_cpu,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y_cpu,
            pos=pos_cpu,
            rsa=data.rsa if hasattr(data, 'rsa') else None,
            name=data.name if hasattr(data, 'name') else None
        )
        
        knn_data_list.append(new_data)
    
    return knn_data_list


def apply_knn_to_single_graph(data, k, include_self=False, device=None):
    """
    Applique KNN à une seule structure Data.
    
    Args:
        data: Structure Data (PyTorch Geometric)
        k (int): Nombre de voisins les plus proches à considérer
        include_self (bool): Si True, inclut le nœud lui-même dans les voisins (default: False)
        device: torch.device ou str ('cuda' ou 'cpu'). Si None, détecte automatiquement
    
    Returns:
        Structure Data avec edge_index et edge_attr recompilés selon KNN
    """
    
    return apply_knn_to_graphs([data], k, include_self, device)[0]


# Exemple d'utilisation
if __name__ == "__main__":
    # Charger les données
    data_list = torch.load("path/to/your/data.pt")
    
    # Appliquer KNN avec k=15 voisins sur GPU (automatique)
    knn_data_list = apply_knn_to_graphs(data_list, k=15)
    
    # Ou spécifier le device manuellement
    # knn_data_list = apply_knn_to_graphs(data_list, k=15, device='cuda')
    # knn_data_list = apply_knn_to_graphs(data_list, k=15, device='cpu')
    
    # Afficher les informations
    for i, data in enumerate(knn_data_list):
        print(f"Graph {i} ({data.name}):")
        print(f"  Nodes: {data.x.shape[0]}")
        print(f"  Edges: {data.edge_index.shape[1]}")
        print(f"  Edge attributes: {data.edge_attr.shape}")
