import pandas as pd
import os
import sys

astral_url = "https://scop.berkeley.edu/downloads/scopeseq-2.08/astral-scopedom-seqres-gd-all-2.08-stable.fa"
fasta_file = "astral-all.fa"

print(f"Downloading full ASTRAL dataset from {astral_url}...")
!wget --no-check-certificate {astral_url} -O {fasta_file}

def load_scop_sequences(fasta_path, target_ids):
    sequences = {}
    target_set = set(target_ids)
    current_id = None
    
    print(f"\nParsing {fasta_path}...")
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Header format: >d1ux8a_ ...
                current_id = line.split()[0][1:]
            else:
                if current_id in target_set:
                    sequences[current_id] = line
    return sequences

# Loading the dataset
print("\nLoading sampled dataset...")
df = pd.read_csv('sampled_dataset.csv')
scop_ids = df['scop_id'].tolist()

# Extract sequences from the FULL file
seq_dict = load_scop_sequences(fasta_file, scop_ids)

# Map and Check
df['sequence'] = df['scop_id'].map(seq_dict)
missing = df['sequence'].isna().sum()

print(f"\nResults:")
print(f"- Total Domains: {len(df)}")
print(f"- Missing Sequences: {missing}")

if missing == 0:
    print("SUCCESS: Full coverage achieved!")
    df.to_csv('sampled_dataset_with_seqs.csv', index=False)
    print("Saved to sampled_dataset_with_seqs.csv")
else:
    print(f"WARNING: Still missing {missing} sequences. Check file version.")