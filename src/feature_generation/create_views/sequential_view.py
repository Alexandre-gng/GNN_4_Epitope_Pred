import torch
from torch_geometric.data import Data
from typing import List


def create_sequential_graph_k_rank(data_list: List[Data], k: int) -> List[Data]:
    """
    Crée des graphes avec connexions séquentielles de rang k à partir d'une liste de graphes.
    
    Pour chaque protéine dans la liste:
    - Si k=1: chaque acide aminé est connecté uniquement à son prédécesseur et son successeur
    - Si k=2: chaque acide aminé est connecté à son prédécesseur de prédécesseur et son successeur de successeur
    - Si k=n: chaque acide aminé est connecté au n-ième acide aminé avant lui et après lui
    
    Les distances (edge_attr) sont calculées à partir des positions 3D.
    
    Args:
        data_list: Liste de structures torch_geometric.data.Data contenant les graphes de protéines
        k: Rang de la connexion séquentielle (1 pour plus proche, 2 pour k=2, etc.)
    
    Returns:
        Liste de structures Data avec les connexions séquentielles de rang k
    
    Example:
        >>> data_list = [Data(x=[162, 1792], edge_index=[2, 2794], ...)]
        >>> k = 1
        >>> new_data_list = create_sequential_graph_k_rank(data_list, k)
    """
    new_data_list = []
    
    for data in data_list:
        # Explicitly set num_nodes if not present to avoid PyG warning
        if not hasattr(data, 'num_nodes') or data.num_nodes is None:
            data.num_nodes = data.node_attrs.shape[0]
        
        print(f"Processing graph: {data.name} with {data.num_nodes} nodes")
        print()
        # Récupérer le nombre d'acides aminés (noeuds)
        num_nodes = data.num_nodes
        
        # Vérifier que les positions 3D sont disponibles
        if hasattr(data, 'coords') and data.coords is not None:
            pos = data.coords
        else:
            raise ValueError("The graph data must have 'coords' attribute with 3D positions for each node.")
        
        # Créer les connexions séquentielles de rang k
        edge_list = []
        distances = []
        
        for i in range(num_nodes):
            # Connexion vers l'acide aminé k positions après
            edge_list
            if i + k < num_nodes:
                # Ajouter l'arête i -> i+k
                edge_list.append([i, i + k])
                # Calculer la distance euclidienne
                dist = torch.norm(pos[i + k] - pos[i], p=2, keepdim=True)
                distances.append(dist)
                
                # Ajouter l'arête i+k -> i (graphe non-orienté)
                edge_list.append([i + k, i])
                distances.append(dist)
        
        # Convertir en tensor
        if len(edge_list) > 0:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.cat(distances, dim=0)  # Shape: [num_edges, 1]
        else:
            # Si pas de connexions possibles
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float32)
        
        # Récupérer les features des nœuds (utiliser l'accès dictionnaire pour éviter le conflit avec la méthode)
        if 'node_attrs' in data:
            node_features = data['node_attrs']
        elif hasattr(data, 'x') and data.x is not None:
            node_features = data.x
        else:
            raise ValueError("No node features found in the data object")
        
        new_data = Data(
            node_attrs=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=data.y,
            coords=data.coords,
            node_id=data.node_id,
            num_nodes=data.num_nodes,
            name=data.name,
            train_mask=data.train_mask,
            rsa=data.rsa
        )
        
        new_data_list.append(new_data)
    
    return new_data_list
