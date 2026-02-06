import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx

from matplotlib.lines import Line2D
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from scipy.spatial.distance import cdist

class ProteinGraphBuilder:
    '''Class to build a protein graph from a DataFrame and embeddings.
    Args:
        df (pd.DataFrame): DataFrame with protein residue data.
        embeddings (torch.Tensor): Node feature embeddings.
        global_features (torch.Tensor): Global graph features.
        threshold (float): Distance threshold for edge creation.
    Returns:
        GraphData: Protein graph data object.

    '''
    def __init__(self, df, embeddings, global_features, threshold=6.0):
        self.df = df
        self.embeddings = embeddings
        self.global_features = global_features
        self.distance_threshold = threshold
        self.graph = None


    def build_graph(self, device: torch.device):
        # Build matrix of distances and create edges based on threshold
        if isinstance(device, str):
            device = torch.device(device)
            
        coords = torch.tensor(self.df[['x_coord', 'y_coord', 'z_coord']].values,
                  dtype=torch.float32,
                  device=device)
        
        distance_matrix = torch.cdist(coords, coords, p=2)
        adj = distance_matrix < self.distance_threshold
        adj.fill_diagonal_(False)

        edge_index = adj.nonzero(as_tuple=False).t().contiguous()

        # Edge features
        source_index = edge_index[0]
        target_index = edge_index[1]
        source_coords = coords[source_index]
        target_coords = coords[target_index]
        distances = torch.norm(source_coords - target_coords, dim=1, keepdim=True)
        # custom_feature_1 = 1.0 / (distances**2 + 1e-6)
        # edge_attr = torch.cat([distances, custom_feature_1], dim=1).float()
        edge_attr = distances.float()
        
        # Global features
        if self.global_features is not None:
            global_features = self.global_features.to(device).float()
            if global_features.dim() == 1:
                global_features = global_features.unsqueeze(0)
        else:
            global_features = None

        # Add label y. A epitope classification for each residue (node)
        y_label = torch.tensor(self.df['epitope'].values, dtype=torch.long, device=device)
        node_positions = coords
        # Create the graph data object with all components
        self.graph = Data(
            x=self.embeddings.float(),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y_label,
            pos=node_positions,
            global_feat=global_features
        )
        return self.graph


    def get_graph(self, highlight_epitopes=False, title=None):
        # Check if the graph has been created and has necessary attributes
        if self.graph is None or not all(attr in self.graph for attr in ['x', 'pos', 'edge_index', 'y']):
             raise ValueError("El objeto de grafo debe contener 'x', 'pos', 'edge_index' y 'y' (etiquetas de epítopo).")

        # Graph PyG -> NetworkX to visualize
        g_nx = to_networkx(self.graph, node_attrs=['pos'])
        pos_3d = nx.get_node_attributes(g_nx, 'pos')

        # Colors and legend setup
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(projection='3d')

        legend_elements = []
        node_colors = 'skyblue' # Color por defecto

        # Flatten the y tensor to safely access the labels by node index
        y_labels_flat = self.graph.y.flatten()

        if highlight_epitopes:
            # .item() safely extracts the value (0 or 1) from the tensor element
            node_colors = [
                'red' if y_labels_flat[i].item() == 1 else 'cornflowerblue' 
                for i in g_nx.nodes()
            ]

            if title is None:
                title = 'Visualización 3D con Epítopos Resaltados'

            # Create custom legend
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Epítopo', markerfacecolor='red', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='No Epítopo', markerfacecolor='cornflowerblue', markersize=10)
            ]
        else:
            if title is None:
                title = 'Visualización 3D del Grafo de la Proteína'

        # Draw nodes and edges
        ax.scatter([p[0] for p in pos_3d.values()],
                   [p[1] for p in pos_3d.values()],
                   [p[2] for p in pos_3d.values()],
                   c=node_colors, s=60, ec='black')

        for edge in g_nx.edges():
            p1 = pos_3d[edge[0]]
            p2 = pos_3d[edge[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='gray', alpha=0.6)

        # Configure plot
        ax.set_title(title)
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        if legend_elements:
            ax.legend(handles=legend_elements)

        plt.show()
