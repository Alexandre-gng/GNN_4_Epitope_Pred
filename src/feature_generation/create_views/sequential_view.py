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
        # Récupérer le nombre d'acides aminés (noeuds)
        num_nodes = data.x.shape[0]
        
        # Vérifier que les positions 3D sont disponibles
        if hasattr(data, 'pos') and data.pos is not None:
            pos = data.pos
        else:
            raise ValueError("Le graphe doit posséder l'attribut 'pos' avec les positions 3D des nœuds")
        
        # Créer les connexions séquentielles de rang k
        edge_list = []
        distances = []
        
        for i in range(num_nodes):
            # Connexion vers l'acide aminé k positions après
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
        
        # Créer une copie des données avec les nouvelles connexions
        new_data = Data(
            x=data.x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=data.y if hasattr(data, 'y') else None,
            pos=data.pos
        )
        
        # Copier les attributs supplémentaires si disponibles
        if hasattr(data, 'name'):
            new_data.name = data.name
        if hasattr(data, 'rsa') and data.rsa is not None:
            new_data.rsa = data.rsa
        if hasattr(data, 'global_feat') and data.global_feat is not None:
            new_data.global_feat = data.global_feat
        
        new_data_list.append(new_data)
    
    return new_data_list


def create_sequential_graph_k_rank_bidirectional(data_list: List[Data], k: int) -> List[Data]:
    """
    Crée des graphes avec connexions séquentielles de rang k (version optimisée sans doublons).
    Cette version évite les doublons d'arêtes pour les graphes non-dirigés.
    
    Les distances (edge_attr) sont calculées à partir des positions 3D.
    Note: PyTorch Geometric gère les graphes non-dirigés avec des arêtes bidirectionnelles,
    donc cette version n'est généralement pas recommandée. Utilisez create_sequential_graph_k_rank à la place.
    
    Args:
        data_list: Liste de structures torch_geometric.data.Data
        k: Rang de la connexion séquentielle
    
    Returns:
        Liste de structures Data avec connexions séquentielles de rang k
    """
    new_data_list = []
    
    for data in data_list:
        num_nodes = data.x.shape[0]
        
        # Vérifier que les positions 3D sont disponibles
        if hasattr(data, 'pos') and data.pos is not None:
            pos = data.pos
        else:
            raise ValueError("Le graphe doit posséder l'attribut 'pos' avec les positions 3D des nœuds")
        
        edge_dict = {}  # Utiliser un dictionnaire pour stocker les arêtes uniques avec leurs distances
        
        for i in range(num_nodes):
            # Connexion vers l'acide aminé k positions avant
            if i - k >= 0:
                edge_key = (min(i, i - k), max(i, i - k))
                if edge_key not in edge_dict:
                    dist = torch.norm(pos[max(i, i - k)] - pos[min(i, i - k)], p=2, keepdim=True)
                    edge_dict[edge_key] = dist
            
            # Connexion vers l'acide aminé k positions après
            if i + k < num_nodes:
                edge_key = (min(i, i + k), max(i, i + k))
                if edge_key not in edge_dict:
                    dist = torch.norm(pos[max(i, i + k)] - pos[min(i, i + k)], p=2, keepdim=True)
                    edge_dict[edge_key] = dist
        
        # Convertir en listes
        if len(edge_dict) > 0:
            edge_list = list(edge_dict.keys())
            distances = list(edge_dict.values())
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.cat(distances, dim=0)  # Shape: [num_edges, 1]
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float32)
        
        # Créer une copie des données
        new_data = Data(
            x=data.x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=data.y if hasattr(data, 'y') else None,
            pos=data.pos
        )
        
        # Copier les attributs supplémentaires si disponibles
        if hasattr(data, 'name'):
            new_data.name = data.name
        if hasattr(data, 'rsa') and data.rsa is not None:
            new_data.rsa = data.rsa
        if hasattr(data, 'global_feat') and data.global_feat is not None:
            new_data.global_feat = data.global_feat
        
        new_data_list.append(new_data)
    
    return new_data_list


# Exemple d'utilisation
if __name__ == "__main__":
    # Créer des exemples de données
    from torch_geometric.data import Data
    
    # Exemple 1: Créer un graphe exemple
    x = torch.randn(10, 1792)  # 10 acides aminés, 1792 features
    y = torch.tensor([0, 1])  # Labels
    pos = torch.randn(10, 3)  # Positions 3D
    
    data = Data(
        x=x,
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.ones((2, 1)),
        y=y,
        pos=pos,
        name='5VX4',
        rsa=torch.ones(10)
    )
    
    # Appliquer la fonction
    result_k1 = create_sequential_graph_k_rank([data], k=1)
    result_k2 = create_sequential_graph_k_rank([data], k=2)
    
    print(f"Graphe original - Nombre d'arêtes: {data.edge_index.shape[1]}")
    print(f"Graphe k=1 - Nombre d'arêtes: {result_k1[0].edge_index.shape[1]}")
    print(f"Graphe k=2 - Nombre d'arêtes: {result_k2[0].edge_index.shape[1]}")
    print(f"\nEdges pour k=1:\n{result_k1[0].edge_index}")
    print(f"\nEdges pour k=2:\n{result_k2[0].edge_index}")
    print(f"\nDistances pour k=1 (premières 5):\n{result_k1[0].edge_attr[:5]}")
    print(f"\nDistances pour k=2 (premières 5):\n{result_k2[0].edge_attr[:5]}")
