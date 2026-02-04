import sys
from pathlib import Path

import re
import os
import sqlite3
from pathlib import Path
import torch
from torch_geometric.data import Data
from src.data.embeddings import EmbeddingLoader
from src.data.graph_builder import ProteinGraphBuilder
from src.data.protein_structure import ProteinStructureLoader


THRESHOLD = 10.0



def extract_numbers(epitope_str):
# Usamos regex para encontrar todas las secuencias de dígitos
    numbers = re.findall(r'\d+', epitope_str)
    # Convertimos a enteros
    return [int(num) for num in numbers]

def get_rank001_pdb(pdb_path):
    """
    Extracts the PDB file with 'rank_001' in its name from the given directory.
    Args:
        pdb_path (str): Path to the directory containing PDB files.
    Returns:
        str: Path to the PDB file with 'rank_001' in its name.
    """
    path = Path(pdb_path)
    pdb_files = [
        str(f)  # puedes usar f.name si solo quieres el nombre
        for f in path.glob("*.pdb")
        if "rank_001" in f.name
    ]
    if not pdb_files:
        return None
    return pdb_files[0]

# Funcion para pasar de http://www.iedb.org/assay/1410447 a 1410447
def extract_iedb_id(iedb_url):
    return iedb_url.rstrip('/').split('/')[-1]


def _detect_epitope_column(conn: sqlite3.Connection, table_name: str) -> str:
    """
    Heuristically detect which column holds epitope positions.
    Strategy:
    - Exclude obvious id/path columns like 'Iedb', 'Protein'.
    - Prefer columns whose sample values contain digits (e.g., "A72, C73").
    - Avoid columns that look like raw sequences (very long, alphabetic only, no digits).
    Returns the detected column name or raises ValueError if none found.
    """
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    cols = [row[1] for row in cur.fetchall()]

    candidates = []
    # Sample a few rows
    cur.execute(f"SELECT * FROM {table_name} LIMIT 100")
    rows = cur.fetchall()
    if not rows:
        raise ValueError(f"Table {table_name} is empty")

    # Build column index map
    cur.execute(f"PRAGMA table_info({table_name})")
    schema = cur.fetchall()
    col_index = {row[1]: idx for idx, row in enumerate(schema)}

    def looks_like_sequence(val: str) -> bool:
        if not isinstance(val, str):
            return False
        # Long string with only letters (and maybe few special chars), no digits
        return len(val) > 100 and not any(ch.isdigit() for ch in val)

    def contains_positions(val: str) -> bool:
        if not isinstance(val, str):
            return False
        # Has at least one digit; typical formats: "A72, C73" or "72,73" etc.
        return any(ch.isdigit() for ch in val)

    for c in cols:
        if c.lower() in {"iedb", "protein", "pdb", "pdb_id"}:
            continue
        # Inspect up to first 20 non-null values
        idx = col_index[c]
        non_null_samples = [r[idx] for r in rows if r[idx] is not None][:20]
        if not non_null_samples:
            continue
        # If any sample looks like positions and majority are not sequences, mark as candidate
        has_positions = any(contains_positions(v) for v in non_null_samples)
        mostly_not_sequences = sum(1 for v in non_null_samples if looks_like_sequence(v)) < len(non_null_samples) // 2
        if has_positions and mostly_not_sequences:
            candidates.append(c)

    if candidates:
        # Prefer common names
        preferred = [c for c in candidates if c.lower() in {"epitopes", "epitope", "epitope_positions", "positions"}]
        return preferred[0] if preferred else candidates[0]

    raise ValueError("Could not detect epitope positions column. Please specify correct column in DB.")




def extract_protein_id(protein_str):
    """
    Extracts the protein ID from a full URL or reference.
    Converts: http://www.ncbi.nlm.nih.gov/protein/ADA79546.1 → ADA79546.1
    """
    if '/' in protein_str:
        return protein_str.rstrip('/').split('/')[-1]
    return protein_str


def create_graph_list(db_path, table_name, embeddings_dir: list, pdb_dir, selected_folders, device: torch.device):
    """
    Creates a list of graph data objects from the Proteins table and corresponding PDB files and embeddings.
    Args:
        db_path (str): Path to the SQLite database file.
        table_name (str): Name of the table containing epitope data (should be "Proteins").
        embeddings_dir (list): List of directories containing embedding files: ESM2 and ESM-IF1.
        pdb_dir (str): Directory containing PDB files.
        selected_folders (list): List of selected folders to process (pdb_high, pdb_very_high, pdb_low, pdb_very_low).
        device (torch.device): Device to load the graphs onto.
    Returns:
        list: List of torch_geometric.data.Data objects representing the graphs.
    """
    # this list contains all the proteins that caused mismatches with ESM2
    ERRORS = ['6IEQ_G', 'AEM60113.1', 'CAA68170.1', 'AWX63617.1', 'AAB27209.1', 'AAL05536.1', 'NP_001005726.1', 'ABI16232.1', 'AAA79214.1', 'CAF24776.1', 'P17466.1', 'P03441.3', '2GHV_E', 'UJX89775.1', 'ART30134.1', 'P0DOX5.1', 'AEI71367.1', 'ACR15732.1', 'AMB66463.1', 'ARQ32975.1', 'ADG21447.1', 'AHI48799.1', 'UBE67681.1', 'ATG80981.1', 'QBF80607.1']

    prot_errors=[]
    prot_skipped = []
    skipped = 0
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build a map of protein IDs to their PDB rank_001 files
    rank_pdbs_map = {}
    if selected_folders is not None and len(selected_folders) > 0:
        for folder in selected_folders:
            print(f"Scanning folder: {folder}")
            pdb_folder_path = os.path.join(pdb_dir, folder)
            if not os.path.exists(pdb_folder_path):
                print(f"  [WARN] Folder does not exist: {pdb_folder_path}")
                continue
            
            for protein_folder in os.listdir(pdb_folder_path):
                protein_id = extract_protein_id(protein_folder)
                path_full = os.path.join(pdb_folder_path, protein_folder)
                try:
                    pdb_rank = get_rank001_pdb(path_full)
                    if pdb_rank is not None:
                        rank_pdbs_map[protein_id] = pdb_rank
                except Exception as e:
                    continue
    
    print(f"Total PDB rank_001 files found: {len(rank_pdbs_map)}")
    
    # Load proteins from database
    conn = sqlite3.connect(db_path)
    rows = conn.execute(f'SELECT Protein, Epitopes FROM {table_name}').fetchall()
    conn.close()
    
    print(f"Total entries from {table_name} table: {len(rows)}")
    
    graph_list = []
    skipped = 0
    
    for idx, (protein_ref, epitopes) in enumerate(rows):
        # Extract clean protein ID
        protein_id = extract_protein_id(protein_ref)
        
        # Check if PDB file exists for this protein
        if protein_id not in rank_pdbs_map:
            skipped += 1
            prot_skipped.append(protein_id)
            print(f"[SKIP {selected_folders}] No PDB rank_001 found for {protein_id}")
            continue
        if protein_id in ERRORS:
            skipped += 1
            prot_skipped.append(protein_id)
            print(f"[SKIP {selected_folders}] Protein {protein_id} is in the known errors list.")
            continue
        
        pdb_rank = rank_pdbs_map[protein_id]
        
        # List of epitope residues ["A12", "C15", ...] or numeric positions
        epitopes_residues = epitopes if isinstance(epitopes, str) else str(epitopes)
        epitopes_residues = epitopes_residues.strip()
        
        # Extract list of epitope residue numbers [12, 15, ...]
        list_epitopes = extract_numbers(epitopes_residues)
        
        if len(list_epitopes) == 0:
            skipped += 1
            prot_skipped.append(protein_id)
            print(f"[WARN {selected_folders}] No epitope positions parsed for {protein_id}. Skipping graph.")
            continue

        print(f"Processing {idx+1}: {protein_id}")
        print(f"  PDB Path: {pdb_rank}")
        print(f"  Epitopes: {list_epitopes}")

        # 1. Load protein structure from PDB
        try:
            protein_df = ProteinStructureLoader(
                pdb_path=pdb_rank,
                epitope_positions=list_epitopes,
            )
            summary_df, sequence, original_df = protein_df.create_df()
        except Exception as e:
            prot_errors.append(protein_id)
            prot_skipped.append(protein_id)
            skipped += 1
            print(f"[ERROR {selected_folders}] Failed to load PDB structure for {protein_id}: {e}")
            continue
            
        N_PDB = len(summary_df) # The EXPECTED number of residues/nodes

        # 2. Load Node Features (Embeddings)
        # Handle list of embedding directories
        if isinstance(embeddings_dir, str):
            embeddings_dirs_list = [embeddings_dir]
        else:
            embeddings_dirs_list = embeddings_dir

        list_tensors = []
        files_found = True

        for emb_dir in embeddings_dirs_list:
            # Embeddings are now simply named "[protein_id].pt" in the root directory
            embedding_path = os.path.join(emb_dir, f"{protein_id}.pt")
            
            if not os.path.exists(embedding_path):
                prot_errors.append(protein_id)
                prot_skipped.append(protein_id)
                skipped += 1
                print(f"  [ERROR {selected_folders}] Embedding not found: {embedding_path}")
                files_found = False
                break

            try:
                embeddings_loader = EmbeddingLoader(embedding_path=embedding_path)
                t = embeddings_loader.load(summary_df)
                list_tensors.append(t.cpu())
                print(f"  ✓ Loaded embedding from {emb_dir}")
            except Exception as e:
                prot_errors.append(protein_id)
                prot_skipped.append(protein_id)
                skipped += 1
                print(f"  [ERROR {selected_folders}] Failed to load embeddings for {protein_id}: {e}")
                files_found = False
                break
        
        if not files_found:
            continue
            
        embeddings_tensor = torch.cat(list_tensors, dim=1)

        N_EMBED = embeddings_tensor.shape[0] # The ACTUAL number of residues/nodes in the tensor
        D_EMBED = embeddings_tensor.shape[1] # The dimension of the features

        # 3. Validation Check. Residues in PDB vs. Embeddings 
        if N_PDB != N_EMBED:
            prot_errors.append(protein_id)
            prot_skipped.append(protein_id)
            skipped += 1

            print(f"[ERROR {selected_folders}] Shape mismatch for {protein_id}: PDB N={N_PDB} vs. Embeddings N={N_EMBED}")
            continue

        print(f"  ✓ Shape verification: N={N_PDB} residues")

        # 4. Create the Graph
        try:
            GRAPH_protein = ProteinGraphBuilder(
                df=summary_df,
                embeddings=embeddings_tensor,
                global_features=None,
                threshold=THRESHOLD,
            )
            graph = GRAPH_protein.build_graph(device=device)
        except Exception as e:
            prot_errors.append(protein_id)
            prot_skipped.append(protein_id)
            skipped += 1
            print(f"  [ERROR {selected_folders}] Failed to build graph for {protein_id}: {e}")
            continue
        
        # Ensure at least one positive label exists (sanity check)
        if graph.y.sum().item() == 0:
            skipped += 1
            prot_skipped.append(protein_id)
            print(f"  [WARN {selected_folders}] Graph built without positive labels. Skipping.")
            continue
        
        graph_list.append(graph)
        print(f"  ✓ Graph created successfully")

    print(f"\n=== Summary ===")
    print(f"✓ Graphs created: {len(graph_list)}")
    print(f"✗ Skipped: {skipped}")
    print(f"✗ Proteins with errors: {len(prot_errors)}")
    print(f"  List of proteins with errors: {prot_errors}")
    print(f"✗ Proteins skipped: {len(prot_skipped)}")
    print(f"  List of proteins skipped: {prot_skipped}")
    return graph_list