import numpy as np
import matplotlib.pyplot as plt

# Load data
pdb_maps = np.load('data/features/pdb_maps.npy')
af_maps = np.load('data/features/af_maps.npy')
labels = np.load('data/features/labels.npy')
ids = np.load('data/features/ids.npy')

print("=== SHAPE CHECK ===")
print(f"PDB maps: {pdb_maps.shape}")
print(f"AF maps: {af_maps.shape}")
print(f"Labels: {labels.shape}")
print(f"IDs: {ids.shape}")

print("\n=== VALUE CHECK ===")
print(f"PDB maps - min: {pdb_maps.min()}, max: {pdb_maps.max()}")
print(f"AF maps - min: {af_maps.min()}, max: {af_maps.max()}")
print(f"Are they binary? PDB unique values: {np.unique(pdb_maps)}")

print("\n=== DOMAIN SHIFT CHECK ===")
diff = np.abs(pdb_maps - af_maps).mean()
print(f"Mean difference between PDB and AF: {diff:.4f}")

print("\n=== SAMPLE VISUALIZATION ===")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i in range(4):
    idx = i * 1000  # Sample from different parts
    
    axes[0, i].imshow(pdb_maps[idx], cmap='binary')
    axes[0, i].set_title(f'PDB - {ids[idx]}\nClass {labels[idx]}')
    axes[0, i].axis('off')
    
    axes[1, i].imshow(af_maps[idx], cmap='binary')
    axes[1, i].set_title(f'AlphaFold - {ids[idx]}')
    axes[1, i].axis('off')

plt.suptitle('PDB (top) vs AlphaFold (bottom) Contact Maps')
plt.tight_layout()
plt.show()


# # Pick a random index
# idx = 0 

# print(f"Checking Protein: {ids[idx]}")
# print(f"Class Label: {labels[idx]}")

# # Plot
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# # PDB (Real)
# axes[0].imshow(X_pdb[idx], cmap='viridis', origin='lower')
# axes[0].set_title(f"Real (PDB)\n{ids[idx]}")

# # AlphaFold (Sim)
# axes[1].imshow(X_af[idx], cmap='viridis', origin='lower')
# axes[1].set_title(f"Sim (AlphaFold)\n{ids[idx]}")

# # Difference (The Domain Gap!)
# diff = np.abs(X_pdb[idx] - X_af[idx])
# axes[2].imshow(diff, cmap='inferno', origin='lower')
# axes[2].set_title("Difference (Sim - Real)")

# plt.show()