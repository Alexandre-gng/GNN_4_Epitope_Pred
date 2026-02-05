import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

import esm
from esm.inverse_folding import util as if1_util

device = "cpu"
print(f"Using device: {device}")


# 3-letter to 1-letter amino acid codes
AA_3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def parse_pdb_file(pdb_path: Path):
	"""
	Parse a PDB file and extract backbone coordinates and sequence.
	
	Returns:
		coords: numpy array of shape (num_residues, 3, 3) with N, CA, C coords
		seq: string of amino acids
		chain_id: the chain identifier used
	"""
	residues = defaultdict(lambda: {"N": None, "CA": None, "C": None, "resname": None})
	chain_id = None
	
	with open(pdb_path, "r") as f:
		for line in f:
			if not line.startswith("ATOM"):
				continue
			
			atom_name = line[12:16].strip()
			if atom_name not in ["N", "CA", "C"]:
				continue
			
			chain = line[21].strip()
			if chain_id is None:
				chain_id = chain
			if chain != chain_id:
				continue
			
			res_num = int(line[22:26].strip())
			res_name = line[17:20].strip()
			
			try:
				x = float(line[30:38].strip())
				y = float(line[38:46].strip())
				z = float(line[46:54].strip())
			except ValueError:
				continue
			
			residues[res_num]["resname"] = res_name
			residues[res_num][atom_name] = np.array([x, y, z])
	
	if not residues:
		raise ValueError(f"No backbone atoms found in {pdb_path}")
	
	# Sort by residue number
	sorted_res_nums = sorted(residues.keys())
	coords_list = []
	seq_list = []
	
	for res_num in sorted_res_nums:
		res = residues[res_num]
		if res["N"] is None or res["CA"] is None or res["C"] is None:
			# Skip residues with missing backbone atoms
			continue
		
		coords_list.append([res["N"], res["CA"], res["C"]])
		res_name = res["resname"]
		seq_list.append(AA_3_TO_1.get(res_name, "X"))
	
	if not coords_list:
		raise ValueError(f"No complete residues (with N, CA, C) found in {pdb_path}")
	
	coords = np.array(coords_list, dtype=np.float32)
	seq = "".join(seq_list)
	
	return coords, seq, chain_id or "A"


def load_if1_model(model_path: Path, device: torch.device):
	"""Load ESM-IF1 model from path or download if not exists."""
	try:
		model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
	except Exception:
		raise RuntimeError(
			f"Could not load ESM-IF1 model. "
			f"Please ensure the model is available via esm.pretrained or at {model_path}"
		)
	
	model = model.to(device)
	model.eval()
	return model, alphabet


def compute_embedding(model, alphabet, coords: np.ndarray, device: torch.device) -> torch.Tensor:
	"""
	Compute ESM-IF1 embeddings for given coordinates.
	Ensures all tensors stay on the same device throughout computation.
	"""
	coords = np.asarray(coords, dtype=np.float32)
	coords[np.isnan(coords)] = np.inf
	
	# Keep coords as numpy array for get_encoder_output (it handles the conversion)
	# but ensure model is on device
	with torch.no_grad():
		# get_encoder_output returns embeddings directly on the model's device
		embeddings = if1_util.get_encoder_output(model, alphabet, coords)
	
	# Handle different return types
	if isinstance(embeddings, dict):
		if "encoder_out" in embeddings:
			embeddings = embeddings["encoder_out"]
		elif "representations" in embeddings:
			embeddings = embeddings["representations"]
		else:
			embeddings = next(iter(embeddings.values()))
	
	if isinstance(embeddings, (list, tuple)):
		embeddings = embeddings[0]
	
	# Ensure embeddings are on CPU before returning
	if hasattr(embeddings, 'device'):
		embeddings = embeddings.to("cpu")
	
	return embeddings.detach()


def generate_embeddings(pdb_dir: Path, model_path: Path, output_dir: Path, overwrite: bool):
	output_dir.mkdir(parents=True, exist_ok=True)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	device = "cpu"
	print(f"🔧 Using device: {device}")
	
	model, alphabet = load_if1_model(model_path, device)
	print(f"✓ Model loaded on device: {device}")

	pdb_files = sorted([p for p in pdb_dir.iterdir() if p.suffix.lower() == ".pdb"])
	if not pdb_files:
		raise FileNotFoundError(f"No .pdb files found in {pdb_dir}")

	print(f"📁 Found {len(pdb_files)} PDB files to process")
	
	success_count = 0
	for pdb_path in tqdm(pdb_files, desc="ESM-IF1 embeddings"):
		out_path = output_dir / f"{pdb_path.stem}.pt"
		if out_path.exists() and not overwrite:
			success_count += 1
			continue

		try:
			coords, seq, chain_id = parse_pdb_file(pdb_path)
			embeddings = compute_embedding(model, alphabet, coords, device)
			
			# Ensure embeddings is on CPU before saving
			if isinstance(embeddings, torch.Tensor):
				embeddings = embeddings.detach().cpu()

			torch.save(
				{
					"sequence": list(seq),
					"embeddings": embeddings,
					"chain_id": chain_id,
				},
				out_path,
			)
			success_count += 1
		except Exception as e:
			print(f"\n⚠ Error processing {pdb_path.name}: {e}")
			continue
	
	print(f"\n✓ Completed: {success_count}/{len(pdb_files)} embeddings generated")


def parse_args():
	parser = argparse.ArgumentParser(description="Generate ESM-IF1 embeddings from PDB files.")
	parser.add_argument(
		"--pdb-dir",
		type=Path,
		default=Path("data/epitope3d/pdb_structure"),
		help="Directory containing PDB files.",
	)
	parser.add_argument(
		"--model-path",
		type=Path,
		default=Path("esm_if1_gvp4_t16_142M_UR50.pt"),
		help="Path to esm_if1_gvp4_t16_142M_UR50.pt.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("data/epitope3d/embeddings/ESMIF1"),
		help="Output directory for embeddings.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite existing embedding files.",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	generate_embeddings(args.pdb_dir, args.model_path, args.output_dir, args.overwrite)


if __name__ == "__main__":
	main()
