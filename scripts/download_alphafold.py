import pandas as pd
import requests
import os
import io
import gzip
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

INPUT_CSV = 'data/scop/sampled_dataset_with_seq.csv'
OUTPUT_DIR = 'data/alphafold'
MAX_WORKERS = 20
SIFTS_URL = "http://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz"

def parse_chain_from_scop(scop_id):
    if len(str(scop_id)) >= 6:
        return scop_id[5].lower()
    return 'a'

def download_af_worker(uid):
    """Download AlphaFold structure using API to get correct version."""
    dest = os.path.join(OUTPUT_DIR, f"{uid}.pdb")
    
    if os.path.exists(dest):
        return True
    
    try:
        # First get the correct URL from API
        api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uid}"
        r = requests.get(api_url, timeout=10)
        if r.status_code != 200:
            return False
        
        data = r.json()
        if not data:
            return False
        
        pdb_url = data[0].get('pdbUrl')
        if not pdb_url:
            return False
        
        # Download the actual PDB file
        r2 = requests.get(pdb_url, timeout=30)
        if r2.status_code == 200:
            with open(dest, 'wb') as f:
                f.write(r2.content)
            return True
    except:
        pass
    return False

def main():
    print("--- AlphaFold Downloader (v6) ---")
    
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    print("Fetching SIFTS database...")
    r = requests.get(SIFTS_URL)
    with gzip.open(io.BytesIO(r.content), 'rt') as f:
        sifts = pd.read_csv(f, skiprows=1, low_memory=False,
                           names=['PDB', 'CHAIN', 'SP_PRIMARY', 'RES_BEG', 'RES_END', 'PDB_BEG', 'PDB_END', 'SP_BEG', 'SP_END'])
    
    print("Mapping to UniProt IDs...")
    df['pdb_lower'] = df['pdb_id'].str.lower()
    df['chain_guess'] = df['scop_id'].apply(parse_chain_from_scop)
    sifts['PDB'] = sifts['PDB'].str.lower()
    sifts['CHAIN'] = sifts['CHAIN'].str.lower()
    
    merged = pd.merge(df, sifts[['PDB', 'CHAIN', 'SP_PRIMARY']].drop_duplicates(),
                      left_on=['pdb_lower', 'chain_guess'],
                      right_on=['PDB', 'CHAIN'],
                      how='inner')
    
    merged = merged.drop_duplicates(subset=['scop_id'])
    uniprots = merged['SP_PRIMARY'].unique()
    print(f"Found {len(uniprots)} unique UniProt IDs to download.")
    
    print(f"Downloading to '{OUTPUT_DIR}'...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(download_af_worker, uniprots),
                           total=len(uniprots), unit="file"))
    
    success_uids = set([u for u, success in zip(uniprots, results) if success])
    print(f"Downloaded {len(success_uids)} files.")
    
    final_map = merged[merged['SP_PRIMARY'].isin(success_uids)].copy()
    final_map.rename(columns={'SP_PRIMARY': 'uniprot_id'}, inplace=True)
    final_map.to_csv('mappingPDBtoUniProt.csv', index=False)
    print("Saved to mappingPDBtoUniProt.csv")

if __name__ == "__main__":
    main()