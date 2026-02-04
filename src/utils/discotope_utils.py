import os
from pathlib import Path

import torch
from Bio import SeqIO

from utils.dataset_utils import THRESHOLD, create_graph_list
from data.graph_builder import ProteinGraphBuilder
from data.protein_structure import ProteinStructureLoader
from data.embeddings import EmbeddingLoader

"""
This file contains utility functions for processing the discotope Dataset.
this dataset lays on a FASTA file where the epitope positions are indicated by uppercase letters in the sequence.
"""

def extract_sequence_and_epitopes_from_fasta(fasta_file):
    """
    Extracts all sequences and epitope positions from a FASTA file.
    Uppercase letters indicate epitope positions.
    
    Args:
        fasta_file (str): Path to FASTA file.
    
    Returns:
        dict: Dictionary with protein_id as key and dict containing:
              - 'sequence': full uppercase sequence
              - 'epitope_positions': list of epitope residue indices
    """
    sequences_dict = {}
    print(f"extricting : Extracting sequences and epitopes from FASTA file: {fasta_file}")
    print(f"extricting current working directory: {Path.cwd()}")
    script_dir = Path(__file__).resolve().parent.parent
    project_root = script_dir.parent  # Exemple : on remonte d'un niveau
    fasta_path = project_root / fasta_file
    with open(fasta_path, 'r') as handle:
        for record in SeqIO.parse(handle, "fasta"):
            protein_id = record.id
            mixed_sequence = str(record.seq)
            
            # Extract epitope positions (uppercase letters)
            epitope_positions = [i for i, char in enumerate(mixed_sequence) if char.isupper()]
            
            # Convert entire sequence to uppercase
            full_sequence = mixed_sequence.upper()
            
            sequences_dict[protein_id] = {
                'sequence': full_sequence,
                'epitope_positions': epitope_positions
            }
    return sequences_dict



def create_graph_list(path_to_fasta, embeddings_type: list, device: torch.device):
    """
    Because of the db epitope sequence fetching part in the original function, we have to redfined it here.
    This function use the FASTA file to create the graph list directly.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    sequences = extract_sequence_and_epitopes_from_fasta(path_to_fasta)
    print(sequences[list(sequences.keys())[0]])
    graph_list = []
    for id, (protein_id, data) in enumerate(sequences.items()):
        list_epitopes = data['epitope_positions']

        print(f"Extracted epitope  {len(list_epitopes)} vs number of residues {len(data['sequence'])} for protein {protein_id}")
        if len(list_epitopes) == 0:
            print(f"[WARN] No epitope positions parsed for IEDB {protein_id} / PDB {id}. Skipping graph.")
            continue

        # print(f"-------------------Proteina número: {id+1}-----------------------")
        # print(f"Creación de la gráfica para IEDB ID: {iedb_id}")
        # print(f"PDB Path: {pdb_rank}")
        # print(f"Embeddings Path: {embedding_path}")
        pdb_file = "data/outputs_pdb/pdb_discotope_test/" + protein_id + ".pdb"
        pdb_file = project_root / pdb_file
        # 1. Instantiate the GraphData class
        protein_df = ProteinStructureLoader(
            pdb_path=pdb_file,
            epitope_positions=list_epitopes,
        )

        # 2. Create the DataFrame and set self.df
        summary_df, sequence, original_df = protein_df.create_df()
        N_PDB = len(summary_df) # The EXPECTED number of residues/nodes
        # pprint(f" PDB Data loaded. EXPECTED Number of Nodes (N_PDB): {N_PDB}")

        # 3. Load Node Features (Embeddings)
        embeddings_list = []
        skip_protein = False
        base_emb_path = project_root / "data/embeddings/discotope"

        for emb_type in embeddings_type:
            if emb_type == "ESM2":
                file_name = f"{protein_id}_embedding.pt"
            elif emb_type == "ESM_IF1":
                file_name = f"{protein_id}.pt"
            else:
                file_name = f"{protein_id}.pt"

            current_emb_path = base_emb_path / emb_type / file_name
            loader = EmbeddingLoader(embedding_path=str(current_emb_path))

            try:
                emb = loader.load(summary_df)
                embeddings_list.append(emb.to(device))
            except Exception as e:
                print(f"Error loading {emb_type} embeddings for {protein_id}: {e}")
                skip_protein = True
                break
        
        if skip_protein or not embeddings_list:
            continue

        embeddings_tensor = torch.cat(embeddings_list, dim=1)
        N_EMBED = embeddings_tensor.shape[0] # The ACTUAL number of residues/nodes in the tensor
        D_EMBED = embeddings_tensor.shape[1] # The dimension of the features

        # pprint(f" Embeddings loaded. ACTUAL Number of Nodes (N_EMBED): {N_EMBED}")
        # pprint(f"  Embeddings shape: {N_EMBED} x {D_EMBED}")

        # 3. Validation Check. Residues in PDB vs. Embeddings 
        if N_PDB != N_EMBED:

            raise ValueError(
                f"\n CRITICAL SHAPE MISMATCH: The number of residues must be equal. "
                f"PDB (DataFrame) N={N_PDB} vs. Embeddings N={N_EMBED}. "
                f"This indicates a problem with PDB processing or embedding generation."
            )

        print(" Shape verification successful: N_PDB matches N_EMBED.")

        # 4. Create the Graph
        GRAPH_protein = ProteinGraphBuilder(
            df=summary_df,
            embeddings=embeddings_tensor,
            global_features=None,
            threshold=THRESHOLD,
        )

        # Pass the actual device instance instead of the torch.device type
        graph = GRAPH_protein.build_graph(device=device)
        # Ensure at least one positive label exists (sanity check)
        if graph.y.sum().item() == 0:
            print(f"[WARN] Graph built without positive labels for protein {protein_id}. Skipping.")
            continue
        graph_list.append(graph)
        # print("\n--- Final Graph Object Summary ---")
        # print(graph)
        # print(f"Node Features (x) shape: {graph.x.shape}")
        # print(f"Edge Features (edge_attr) shape: {graph.edge_attr.shape}")"""
    return graph_list


