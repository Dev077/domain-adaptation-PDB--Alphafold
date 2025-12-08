# scripts/01_create_stratified_dataset.py
import pandas as pd
import numpy as np
import os

def create_stratified_dataset(input_csv, output_csv, samples_per_class=2000):
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)

    # 1. Filter to the 4 Major Classes
    target_classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    df = df[df['scop_class'].isin(target_classes)].copy()
    
    # 2. Extract Superfamily (e.g., "a.1.1.1" -> "a.1.1")
    # This is the key to avoiding data leakage!
    df['superfamily'] = df['hierarchy'].apply(lambda x: '.'.join(x.split('.')[:3]))
    
    print(f"Pool size: {len(df)} domains across {df['superfamily'].nunique()} superfamilies.")

    sampled_dfs = []
    np.random.seed(42)  # Reproducibility

    for cls in target_classes:
        cls_df = df[df['scop_class'] == cls]
        superfamilies = cls_df['superfamily'].unique()
        
        # Strategy: Pick 1 distinct domain from each Superfamily first
        if len(superfamilies) >= samples_per_class:
            # If we have loads of superfamilies, pick 2000 distinct ones
            selected_sfs = np.random.choice(superfamilies, samples_per_class, replace=False)
            sampled = cls_df[cls_df['superfamily'].isin(selected_sfs)].groupby('superfamily').sample(1)
        else:
            # If we have fewer (e.g., 1000 superfamilies), take 1 from each, then random fill
            first_round = cls_df.groupby('superfamily').sample(1)
            remaining_needed = samples_per_class - len(first_round)
            
            remaining_pool = cls_df.drop(first_round.index)
            if len(remaining_pool) >= remaining_needed:
                second_round = remaining_pool.sample(n=remaining_needed, random_state=42)
                sampled = pd.concat([first_round, second_round])
            else:
                sampled = cls_df # Take all if not enough

        sampled_dfs.append(sampled)
        print(f"Class {cls}: Selected {len(sampled)} domains.")

    # 3. Save
    final_df = pd.concat(sampled_dfs, ignore_index=True)
    final_df.to_csv(output_csv, index=False)
    print(f"\nSaved stratified dataset to {output_csv} ({len(final_df)} rows).")

if __name__ == "__main__":
    # Ensure directory exists
    os.makedirs('data/scop', exist_ok=True)
    # Use the clean file you made earlier
    create_stratified_dataset('data/scop/parsed_classifications.csv', 
                              'data/scop/sampled_dataset.csv')