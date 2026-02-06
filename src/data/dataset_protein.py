import torch
from torch_geometric.data import InMemoryDataset, Data
import os
import sqlite3
import sys
from pathlib import Path

# Obtener directorio raíz (que contiene 'src'), desde la ubicación actual del script
project_root = Path(__file__).resolve().parent.parent.parent  # Sube 3 niveles desde ...\src\data\dataset_protein.py
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.embeddings import EmbeddingLoader
from src.data.graph_builder import ProteinGraphBuilder
from src.data.protein_structure import ProteinStructureLoader
from src.utils.dataset_utils import *

class ProteinGraphDataset(InMemoryDataset):

    def __init__(self, db_path, table_name, embeddings_dir, pdb_dir, selected_folders=None, global_features_tensor=None, threshold=6.0, transform=None, pre_transform=None):
        self.db_path = db_path
        self.table_name = table_name
        self.embedding_dir = embeddings_dir
        self.pdb_dir = pdb_dir
        self.selected_folders = selected_folders
        self.global_features_tensor = global_features_tensor
        self.threshold = threshold

        super().__init__('.', transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['protein_graphs.pt']

    def process(self):

        rank_pdbs = []
        list_pdbs = []
        if self.selected_folders is not None and len(self.selected_folders) > 0:
            for folder in self.selected_folders:
                pdb_folder_path = os.path.join(self.pdb_dir, folder)
                pdb_files = [f for f in os.listdir(pdb_folder_path)]
                pdb_ids  = [f.replace('pdb_','').replace('.pdb','') for f in pdb_files]
                for pdb_id in pdb_ids:
                    path_full = os.path.join(self.pdb_dir, f"{folder}/{pdb_id}")
                    try:
                        pdb_rank = get_rank001_pdb(path_full)
                        rank_pdbs.append(pdb_rank)
                        list_pdbs.append(pdb_id)
                    except Exception as e:
                        continue

        pdb_files_set = set(list_pdbs)
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(f'SELECT Iedb,Protein,sequences FROM {self.table_name}').fetchall()
        data_list = []
        filtered_rows = [row for row in rows if extract_iedb_id(row[1]) in pdb_files_set]
        print(filtered_rows)
        print(len(filtered_rows))

        for id, (iedb_id, pdb_id, epitopes) in enumerate(filtered_rows):
            print(f"Processing IEDB ID {iedb_id} with PDB ID {pdb_id}...")
            try:
                iedb_id = extract_iedb_id(iedb_id)
                pdb_id = extract_iedb_id(pdb_id)

                pdb_path = next((item for item in rank_pdbs if pdb_id in item), None)
                embedding_path = os.path.join(self.embedding_dir, f"{iedb_id}_embedding.pt")
                list_epitope = extract_numbers(epitopes)

                protein_df = ProteinStructureLoader(pdb_path, list_epitope)
                df, sequence, _ = protein_df.create_df()
                embeddings = EmbeddingLoader(embedding_path=embedding_path)
                embedding_tensor = embeddings.load(df)

                N_PDB = len(df)
                N_emb = embedding_tensor.shape[0]
                if N_PDB != N_emb:
                    raise ValueError(f"Mismatch in number of residues for IEDB ID {iedb_id}: PDB has {N_PDB}, embeddings have {N_emb}")
                else:
                    graph_builder = ProteinGraphBuilder(
                        df=df,
                        embeddings=embedding_tensor,
                        global_features=None,
                        threshold=self.threshold)

                    graph_builder.build_graph()
                    data_list.append(graph_builder.graph)
                    print(f"Graph for IEDB ID {iedb_id} with PDB ID {pdb_id} processed successfully.")
                    print(f"Processed IEDB ID {iedb_id} ({id + 1}/{len(rows)})")


            except Exception as e:
                print(f"Error in IEDB ID {iedb_id} due to error: {e}")
                continue
        conn.close()
        print(f"Total graphs processed: {len(data_list)}")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])