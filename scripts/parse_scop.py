import pandas as pd

def parse_scop_classification(scop_file):
    data = []
    
    with open(scop_file, 'r') as f:
        for line in f:
            # Skip comments
            if line.startswith('#'):
                continue
            
            parts = line.strip().split()
            # Ensure line has enough columns
            if len(parts) < 6:
                continue
            
            scop_id = parts[0]
            
            # Correction: Hierarchy is in index 3, not 2
            hierarchy = parts[3]  # e.g., "a.1.1.1"
            scop_class = hierarchy.split('.')[0]
            
            # Optional: You can get PDB and Chain from columns 1 and 2 directly
            # if you prefer relying on the file columns over the ID string.
            pdb_id = parts[1]
            chain_info = parts[2]
            
            data.append({
                'scop_id': scop_id,
                'pdb_id': pdb_id,
                'scop_class': scop_class,
                'hierarchy': hierarchy
            })
    
    return pd.DataFrame(data)

# Parse
df = parse_scop_classification('data/scop/classifications.txt')

print(f"Total SCOP domains: {len(df)}")
print("\nClass distribution:")
print(df['scop_class'].value_counts().sort_index())

# Save
df.to_csv('data/scop/parsed_classifications.csv', index=False)