import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
def run_pca_and_plot(vectors, labels, title, save_path, n_components=2, standardize=True):
    """
    Run PCA on a list of 1D vectors and save a scatter plot.

    Parameters
    ----------
    vectors : list of np.ndarray
        Each entry is a 1D feature vector (same length).
    labels : list of str
        Text labels for each sample.
    title : str
        Figure title.
    save_path : Path or str
        Where to save the PCA figure.
    n_components : int
        Number of PCs to compute (2 for 2D plot, 3 for 3D plot)
    standardize : bool
        If True, standardize features (recommended). If False, only center.
    """
    if len(vectors) < 2:
        print("  [PCA] Not enough samples to run PCA (need >= 2). Skipping.")
        return

    # Stack into matrix (n_samples, n_features)
    X = np.vstack(vectors)
    
    # Step 1: Preprocess
    if standardize:
        # Center AND scale (match sklearn StandardScaler)
        X_mean = X.mean(axis=0, keepdims=True)
        X_std = X.std(axis=0, keepdims=True)
        X_std[X_std == 0] = 1  # Avoid division by zero
        Xc = (X - X_mean) / X_std
    else:
        # Only center (your original approach)
        X_mean = X.mean(axis=0, keepdims=True)
        Xc = X - X_mean

    # Step 2: SVD decomposition
    try:
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError as e:
        print(f"  [PCA] SVD failed: {e}. Skipping PCA plot.")
        return

    # Step 3: PC scores (projection onto PCs)
    scores = U * S  # Shape: (n_samples, n_components)
    # Equivalent to: scores = Xc @ Vt.T
    
    # Step 4: Explained variance
    total_variance = np.sum(S**2)
    explained_variance_ratio = (S**2) / total_variance
    
    print(f"  [PCA] Explained variance:")
    for i in range(min(n_components, len(S))):
        print(f"    PC{i+1}: {explained_variance_ratio[i]:.4f} ({explained_variance_ratio[i]*100:.2f}%)")
    print(f"  [PCA] Total (PC1-{n_components}): {explained_variance_ratio[:n_components].sum():.4f}")
    
    # Step 5: Plot
    if n_components == 2:
        # 2D plot
        pc1 = scores[:, 0]
        pc2 = scores[:, 1] if scores.shape[1] > 1 else np.zeros_like(pc1)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(pc1, pc2, alpha=0.8, s=50)

        for x, y, lab in zip(pc1, pc2, labels):
            ax.text(x, y, lab, fontsize=7, ha='center', va='center')

        ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)' if len(S) > 1 else 'PC2')
        ax.set_title(title)
        ax.axhline(0, color='grey', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axvline(0, color='grey', linewidth=0.5, linestyle='--', alpha=0.5)
        
    elif n_components == 3:
        # 3D plot
        from mpl_toolkits.mplot3d import Axes3D
        pc1 = scores[:, 0]
        pc2 = scores[:, 1] if scores.shape[1] > 1 else np.zeros_like(pc1)
        pc3 = scores[:, 2] if scores.shape[1] > 2 else np.zeros_like(pc1)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pc1, pc2, pc3, alpha=0.8, s=50)
        
        for x, y, z, lab in zip(pc1, pc2, pc3, labels):
            ax.text(x, y, z, lab, fontsize=7)
        
        ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)' if len(S) > 1 else 'PC2')
        ax.set_zlabel(f'PC3 ({explained_variance_ratio[2]*100:.1f}%)' if len(S) > 2 else 'PC3')
        ax.set_title(title)
    
    fig.tight_layout()
    save_path = Path(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [PCA] Saved PCA plot to: {save_path}")