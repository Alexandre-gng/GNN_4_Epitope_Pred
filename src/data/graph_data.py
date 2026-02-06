#------------------------------- VERSIÓN ANTIGUA --------------------------------#

import os
import yaml
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

from biopandas.pdb import PandasPdb
from Bio import Entrez
from scipy.spatial.distance import cdist

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # remonte jusqu'à /src
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

amino_acid_dict = config['aminoacid_dict']
Entrez.email = config['entrez_email']  
epitope_db = config['db_path']


class GraphData:
    def __init__(self, ruta_pdb, distance_threshold=6.0, epitope_positions=None, pdb=None):
        self.ruta_pdb = ruta_pdb
        self.distance_threshold = distance_threshold
        self.epitope_positions = epitope_positions
        self.pdb = pdb
        self.df = None
        self.graph = None

    def create_df(self):
        if self.ruta_pdb is not None:
            residue_df = PandasPdb().read_pdb(self.ruta_pdb)
            residue_df = residue_df.df['ATOM']
            original_df = residue_df.copy()

        else:
            atom_df = PandasPdb().read_pdb(self.pdb)
            residue_df = atom_df.df['ATOM']

        coord_cols = residue_df.select_dtypes(include='number').columns.tolist()

        summary_df = (
            residue_df.groupby('residue_number',as_index=False)
            .agg({
                'residue_name': 'first',
                **{col: 'mean' for col in coord_cols}
            })
        )

        #Mapear nombre de residuos a una letra
        summary_df['residue_name'] = summary_df['residue_name'].map(amino_acid_dict)
        sequence = ''.join(summary_df['residue_name'].tolist())
        summary_df['epitope'] = [0 for _ in range(len(summary_df))]

        if self.epitope_positions is not None:
            summary_df["epitope"] = summary_df['residue_number'].isin(self.epitope_positions).astype(int)

        return summary_df, sequence, original_df

    def compare_epitopes(self, sequence):
        for i in range(0, len(self.list_epitopes)):
            chain = self.list_epitopes[i][0]
            position = int(self.list_epitopes[i][1:])
            print(f"Cadena: {chain}, Posicion: {position}, Residuo en secuencia: {sequence[position-1]}")

    def _edge_attributes(self, edge_index, coords):
        source_index= edge_index[0].numpy()
        target_index= edge_index[1].numpy()

        source_coords = coords[source_index]
        target_coords = coords[target_index]

        distances = np.linalg.norm(source_coords - target_coords, axis=1, keepdims=True)
        custom_feature_1 = 1.0 / (distances**2 + 1e-6)
        edge_attr = np.concatenate([distances, custom_feature_1], axis=1)

        return torch.from_numpy(edge_attr).float()

    def load_embeddings(self, embedding_path: str, key: str = "embeddings") -> torch.Tensor:
        """
        Loads the pre-calculated node embeddings for the current protein
        and performs necessary checks to ensure order and size match the DataFrame.
        Args:
            embedding_path (str): Full path to the specific .pt file for the protein.
            key (str): The key in the loaded dictionary that contains the tensor ('embeddings' by default).
        Returns:
            torch.Tensor: The ordered node feature tensor (embeddings).
        """
        if self.df is None:
            raise RuntimeError("Cannot load embeddings: The DataFrame (self.df) must be created first by calling create_df().")

        try:
            # 1. Load the data dictionary
            data = torch.load(embedding_path)
            # The structure saved in save_embedding_with_residues is { 'sequence': list, 'embeddings': tensor }
            loaded_sequence = data["sequence"]
            loaded_embeddings = data[key]

            # 2. Extract the sequence derived from the PDB file (from the DataFrame)
            pdb_sequence = ''.join(self.df['residue_name'].tolist())

            # 3. Ensure the number of residues match
            if loaded_embeddings.shape[0] != len(self.df):
                raise ValueError(
                    f"Embedding size mismatch: Loaded {loaded_embeddings.shape[0]} embeddings, "
                    f"but PDB DataFrame has {len(self.df)} residues. "
                    f"Check PDB processing (missing residues?) or embedding file."
                )

            # 4. Secondary Check: Verify sequence consistency
            # This is a strong sanity check. If the sequences don't match, the order is likely wrong.
            if len(pdb_sequence) != len(loaded_sequence) or pdb_sequence != "".join(loaded_sequence):
                 print(f" WARNING: PDB sequence and loaded embedding sequence do not exactly match. "
                       f"Proceeding, but check if PDB residues are missing/different from sequence used for embedding.")


            # Since ESM embeddings are generally generated in sequence order, 
            # and your PDB DataFrame is also in sequential order, 
            # the tensor should be ready to use directly.

            return loaded_embeddings.float()
        except FileNotFoundError:
            raise FileNotFoundError(f"Embedding file not found at: {embedding_path}")
        except KeyError:
            raise KeyError(f"Key '{key}' or 'sequence' not found in the embedding file.")
        except Exception as e:
            raise RuntimeError(f"Error loading and verifying embeddings: {e}")

    def create_graph(self, embeddings, global_features):
        residue_df = self.df.copy()

        coords = residue_df[['x_coord', 'y_coord', 'z_coord']].values
        distance_matrix = cdist(coords, coords, 'euclidean')
        adj = distance_matrix < self.distance_threshold
        np.fill_diagonal(adj, False)
        edge_index = torch.from_numpy(np.vstack(np.nonzero(adj))).long()


        node_features = embeddings.float()
        #--------------------------------------------------------
        # Edge Features (edge_attr)
        edge_attributes = self._edge_attributes(edge_index, coords)

        global_features = global_features.float()

        if global_features.dim() == 1:
            global_features = global_features.unsqueeze(0)

        y_label = torch.tensor([residue_df['epitope'].values]).long()

        node_postions = torch.from_numpy(coords).float()

        graph = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attributes,
            y=y_label,
            pos=node_postions,
            global_feat=global_features
        )

        return graph

    def visualize_graph(self, highlight_epitopes=False, title=None):
        # 1. Check if the graph has been created and has necessary attributes
        if self.graph is None or not all(attr in self.graph for attr in ['x', 'pos', 'edge_index', 'y']):
             raise ValueError("El objeto de grafo debe contener 'x', 'pos', 'edge_index' y 'y' (etiquetas de epítopo).")

        # 2. Convertir el grafo de PyG a NetworkX para la visualización
        g_nx = to_networkx(self.graph, node_attrs=['pos'])
        pos_3d = nx.get_node_attributes(g_nx, 'pos')

        # 3. Configurar colores y leyenda según la opción
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

            # Crear leyenda personalizada
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Epítopo', markerfacecolor='red', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='No Epítopo', markerfacecolor='cornflowerblue', markersize=10)
            ]
        else:
            if title is None:
                title = 'Visualización 3D del Grafo de la Proteína'

        # 4. Dibujar nodos y aristas (rest of the code is correct)
        ax.scatter([p[0] for p in pos_3d.values()],
                   [p[1] for p in pos_3d.values()],
                   [p[2] for p in pos_3d.values()],
                   c=node_colors, s=60, ec='black')

        for edge in g_nx.edges():
            p1 = pos_3d[edge[0]]
            p2 = pos_3d[edge[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='gray', alpha=0.6)

        # 5. Configurar y mostrar el gráfico
        ax.set_title(title)
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        if legend_elements:
            ax.legend(handles=legend_elements)

        plt.show()