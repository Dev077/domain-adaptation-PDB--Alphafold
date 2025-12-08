import numpy as np
import pandas as pd
from Bio import PDB, Align
from scipy.ndimage import zoom
import os
from tqdm import tqdm
import warnings

# --- CONFIGURATION ---
CSV_FILE = 'data/mappingPDBtoUniProt.csv'
PDB_DIR = 'data/pdb'
AF_DIR = 'data/alphafold'
OUTPUT_DIR = 'data/features'
TARGET_SIZE = 128
CONTACT_THRESHOLD = 8.0

os.makedirs(OUTPUT_DIR, exist_ok=True)
warnings.filterwarnings("ignore")

# Setup Fast Aligner
aligner = Align.PairwiseAligner()
aligner.mode = 'local'
aligner.match_score = 2
aligner.mismatch_score = -1
aligner.open_gap_score = -0.5
aligner.extend_gap_score = -0.1

AA_CODES = {
    'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E', 'PHE':'F', 'GLY':'G',
    'HIS':'H', 'ILE':'I', 'LYS':'K', 'LEU':'L', 'MET':'M', 'ASN':'N',
    'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S', 'THR':'T', 'VAL':'V',
    'TRP':'W', 'TYR':'Y'
}

def get_cropped_coords(filepath, target_seq):
    if not os.path.exists(filepath):
        return None

    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('prot', filepath)
    except:
        return None
    
    file_seq = ""
    file_coords = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in AA_CODES:
                    try:
                        file_seq += AA_CODES[residue.get_resname()]
                        file_coords.append(residue['CA'].get_coord())
                    except:
                        pass
        break
    
    if not file_coords:
        return None

    try:
        alignments = aligner.align(target_seq.upper(), file_seq.upper())
        if not alignments:
            return None
        
        best = alignments[0]
        
        # Use .aligned to get coordinate blocks
        # aligned[0] = target coords, aligned[1] = file coords
        aligned = best.aligned
        
        final_coords = []
        for (t_start, t_end), (f_start, f_end) in zip(aligned[0], aligned[1]):
            for i in range(f_start, f_end):
                if i < len(file_coords):
                    final_coords.append(file_coords[i])
        
        if len(final_coords) < 10:
            return None
        return np.array(final_coords)
        
    except:
        return None

def process_and_resize(coords):
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    cmap = (dist < CONTACT_THRESHOLD).astype(np.float32)
    
    h, w = cmap.shape
    zoom_factor = TARGET_SIZE / h
    resized = zoom(cmap, (zoom_factor, zoom_factor), order=1)
    
    return (resized > 0.5).astype(np.float32)

# --- MAIN ---
if __name__ == "__main__":
    print("--- Fast Feature Generator ---")
    
    df = pd.read_csv(CSV_FILE)
    class_map = {c: i for i, c in enumerate(sorted(df['scop_class'].unique()))}
    
    X_pdb, X_af, y, ids = [], [], [], []
    failed = 0
    
    print(f"Processing {len(df)} proteins...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        pdb_path = os.path.join(PDB_DIR, f"pdb{row['pdb_id'].lower()}.ent")
        af_path = os.path.join(AF_DIR, f"{row['uniprot_id']}.pdb")
        
        c_pdb = get_cropped_coords(pdb_path, row['sequence'])
        c_af = get_cropped_coords(af_path, row['sequence'])
        
        if c_pdb is None or c_af is None:
            failed += 1
            continue
            
        X_pdb.append(process_and_resize(c_pdb))
        X_af.append(process_and_resize(c_af))
        y.append(class_map[row['scop_class']])
        ids.append(row['scop_id'])

    print(f"\nDone! Success: {len(y)}, Failed: {failed}")
    
    np.save(os.path.join(OUTPUT_DIR, 'pdb_maps.npy'), np.array(X_pdb))
    np.save(os.path.join(OUTPUT_DIR, 'af_maps.npy'), np.array(X_af))
    np.save(os.path.join(OUTPUT_DIR, 'labels.npy'), np.array(y))
    np.save(os.path.join(OUTPUT_DIR, 'ids.npy'), np.array(ids))
    print("Saved.")