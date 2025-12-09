import pandas as pd
import os

def create_final_dataset_fixed():
    df = pd.read_csv('data/scop/sampled_dataset.csv')

    sequences = {}
    target_ids = set(df['scop_id'].tolist())
    
    current_id = None
    current_seq_lines = []
    
    # 1. READ AND ACCUMULATE
    try:
        with open('data/scop/astral-all.fa', 'r', encoding='latin-1') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Save the previous domain if we were working on one
                    if current_id and current_id in target_ids:
                        sequences[current_id] = "".join(current_seq_lines)
                    
                    # Start new domain
                    parts = line.split()
                    current_id = parts[0][1:] # remove '>'
                    current_seq_lines = []
                else:
                    # Append sequence line
                    if current_id in target_ids:
                        current_seq_lines.append(line)
            
            # Don't forget the very last entry in the file!
            if current_id and current_id in target_ids:
                sequences[current_id] = "".join(current_seq_lines)
                
    except Exception as e:
        print(f"Error reading fasta file: {e}")
        return

    # 2. MAP AND VALIDATE
    print("Mapping sequences...")
    df['sequence'] = df['scop_id'].map(sequences)

    # Check lengths of first 5 to verify fix
    print("\nVerifying Sequence Lengths (First 5):")
    for i in range(5):
        seq = df['sequence'].iloc[i]
        if isinstance(seq, str):
            print(f"ID: {df['scop_id'].iloc[i]}, Length: {len(seq)}")

    # 3. SAVE
    df_clean = df.dropna(subset=['sequence'])
    output_filename = 'sampled_dataset_with_seqs.csv'
    df_clean.to_csv(output_filename, index=False)
    
    print(f"\nSaved correct dataset to {output_filename}")
    print(f"Final Count: {len(df_clean)}")

create_final_dataset_fixed()