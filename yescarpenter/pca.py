# yescarpenter/pca.py

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def _varimax(loadings, max_iter=500, tol=1e-8):
    """Orthogonal varimax rotation matrix R for a (n_variables, n_components) loading matrix."""
    p, k = loadings.shape
    rotation = np.eye(k)
    if k < 2:
        return rotation
    last = 0.0
    for _ in range(max_iter):
        rotated = loadings @ rotation
        u, s, vt = np.linalg.svd(
            loadings.T @ (rotated ** 3 - rotated @ np.diag((rotated ** 2).sum(axis=0)) / p)
        )
        rotation = u @ vt
        if s.sum() < last * (1 + tol):
            break
        last = s.sum()
    return rotation

def _standardized_pca(data, n_components):
    if isinstance(data, pd.DataFrame):
        data = data.values
    data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(data)
    return data, pca, scores

def perform_pca(data, n_components):
    """
    Perform PCA on z-scored data, then apply a varimax rotation.

    Loadings, scores and explained variance all describe the same rotated components.

    Args:
        data (pd.DataFrame or np.ndarray): The input data (observations x variables).
        n_components (int): Number of components to keep.

    Returns:
        loadings (np.ndarray): Rotated loadings (variable-component correlations), (n_variables, n_components).
        explained_variance (np.ndarray): Proportion of total variance carried by each rotated component, sorted in descending order.
        components (np.ndarray): Unit-variance scores on the rotated components, (n_observations, n_components).
    """
    data, pca, scores = _standardized_pca(data, n_components)
    n_obs, n_vars = data.shape

    # PCA loadings: eigenvectors scaled by the square root of the eigenvalues (ddof=0, as in StandardScaler)
    eigenvalues = pca.explained_variance_ * (n_obs - 1) / n_obs
    loadings = pca.components_.T * np.sqrt(eigenvalues)

    rotation = _varimax(loadings)
    loadings = loadings @ rotation
    scores = (scores / np.sqrt(eigenvalues)) @ rotation  # unit-variance scores, so data ~ scores @ loadings.T

    explained_variance = (loadings ** 2).sum(axis=0) / n_vars

    # Rotation does not preserve the ordering of the components, so sort them
    order = np.argsort(-explained_variance)
    return loadings[:, order], explained_variance[order], scores[:, order]

# scree plot
def scree_plot(explained_variance, n_components):
    plt.plot(np.arange(1, n_components + 1), explained_variance, 'o-', color='black')
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance')
    plt.title('Scree plot')
    plt.grid()
    plt.show()

def create_scree_plot(data, max_components):
    # A scree plot shows the unrotated eigenvalue spectrum, so it uses plain PCA
    _, pca, _ = _standardized_pca(data, max_components)
    explained_variance = pca.explained_variance_ratio_
    scree_plot(explained_variance, max_components)

def pc_plot(loadings, df):

    # Create a DataFrame for PCA loadings
    loading_df = pd.DataFrame(loadings, 
                            columns=[f"PC{i+1}" for i in range(loadings.shape[1])], 
                            index=df.columns)

    # Convert to long format for easier plotting
    loading_long = loading_df.reset_index().melt(id_vars="index", 
                                                    var_name="Principal Component", 
                                                    value_name="Loading Strength")
    loading_long.rename(columns={"index": "Trait"}, inplace=True)

    # Set up the plot
    sns.set(style="whitegrid")
    n_components = loading_df.shape[1]
    fig, axes = plt.subplots(1, n_components, figsize=(15, 6), sharey=True, squeeze=False)
    axes = axes.ravel()

    # Create a consistent color palette for the normalized strength
    norm = plt.Normalize(loading_long['Loading Strength'].min(), loading_long['Loading Strength'].max())
    cmap = plt.cm.coolwarm

    # Revised plotting loop
    for i, ax in enumerate(axes):
        pc = f"PC{i+1}"
        pc_data = loading_long[loading_long["Principal Component"] == pc]
        
        # Plot the bars without applying colors yet
        barplot = sns.barplot(
            data=pc_data,
            x="Loading Strength",
            y="Trait",
            ax=ax
        )
        
        # Apply custom colors to each bar based on normalized strength
        for j, bar in enumerate(barplot.patches):
            strength = pc_data.iloc[j]['Loading Strength']
            bar.set_color(cmap(norm(strength)))
        
        ax.set_title(pc, fontsize=20)  # Larger title font size
        ax.tick_params(axis='x', labelsize=16)  # Larger x-tick label size
        ax.tick_params(axis='y', labelsize=16)  # Larger y-tick label size
        
        # Adjust x and y labels directly
        ax.set_xlabel("Loading Strength", fontsize=18)  # Set x-axis label with font size
        ax.set_ylabel("Traits", fontsize=18)            # Set y-axis label with font size

    # Add a color bar for the shared color scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.05, pad=0.1, label='Loading Strength')

    plt.show()

    