import pandas as pd
import os
from Bio.PDB import PDBList
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def fetch_pdb_worker(pdb_id, output_dir):
    """Worker function to download a single PDB file."""
    pdbl = PDBList(verbose=False)
    try:
        # retrieve_pdb_file downloads the file to the specified dir
        # file_format='pdb' yields .ent files (standard legacy format)
        pdbl.retrieve_pdb_file(pdb_id, pdir=output_dir, file_format='pdb', overwrite=False)
        return True
    except Exception as e:
        return False

def download_pdbs_parallel(csv_file, output_dir='data/pdb', max_workers=20):
    df = pd.read_csv(csv_file)
    pdb_ids = df['pdb_id'].unique().tolist()
    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading {len(pdb_ids)} PDB files to {output_dir} using {max_workers} threads...")
    # Run parallel downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Wrap in list to force execution; tqdm shows progress bar
        results = list(tqdm(executor.map(lambda p: fetch_pdb_worker(p, output_dir), pdb_ids), 
                            total=len(pdb_ids), unit="file"))
    success_count = sum(results)
    print(f"\nDownload complete. Success: {success_count}/{len(pdb_ids)}")


if __name__ == "__main__":
    # Uses the file you just created
    download_pdbs_parallel('data/scop/sampled_dataset_with_seq.csv')