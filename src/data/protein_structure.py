import yaml
from pathlib import Path
from biopandas.pdb import PandasPdb

# Get path to gnn_epitope root directory
current_file = Path(__file__).resolve()  # absolute path to this file
gnn_epitope_root = current_file.parents[2]  # src/data/ -> gnn_epitope

config_path = gnn_epitope_root / "config" / "config.yaml"

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Load aminoacid mapping dictionary from config
aminoacid_dict = config['aminoacid_dict']

class ProteinStructureLoader:
    '''Class to load and process protein structure from a PDB file.
    Args:
        pdb_path (str): Path to the PDB file.
        epitope_positions (list, optional): List of residue numbers that are epitopes.
    Returns:
        pd.DataFrame: DataFrame with processed protein structure data.
        str: Protein sequence as a string.
        pd.DataFrame: Original atom-level DataFrame from the PDB file.
    '''

    def __init__(self, pdb_path, epitope_positions=None):
        self.pdb_path = pdb_path
        self.epitope_positions = epitope_positions

    def create_df(self):
        # Load PDB and create summary DataFrame. Select only ATOM records.
        residue_df = PandasPdb().read_pdb(self.pdb_path).df['ATOM']
        original_df = residue_df.copy()
        # Get coordinate columns
        coord_cols = residue_df.select_dtypes(include='number').columns.tolist()

        # Create summary DataFrame grouped by residue number and averaging coordinates
        summary_df = (
            residue_df.groupby('residue_number',as_index=False)
            .agg({'residue_name': 'first', **{col: 'mean' for col in coord_cols}})
        )

        summary_df['residue_name'] = summary_df['residue_name'].map(aminoacid_dict)
        # Extract sequence
        sequence = ''.join(summary_df['residue_name'].tolist())
        # Add epitope column initialized to 0
        summary_df['epitope'] = [0 for _ in range(len(summary_df))]
        # Mark epitopes if positions provided
        if self.epitope_positions is not None:
            summary_df["epitope"] = summary_df['residue_number'].isin(self.epitope_positions).astype(int)

        return summary_df, sequence, original_df