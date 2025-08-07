import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist, pdist
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm

def clean_data_df(df):
    """Remove rows with NaN or inf values in DataFrame."""
    return df.replace([np.inf, -np.inf], np.nan).dropna()

def clean_data_np(data):
    """Remove NaN or inf values in numpy array."""
    data = np.where(np.isfinite(data), data, np.nan)
    return data[~np.isnan(data)]

def get_triangular_matrix(full_rdm):
    """Convert a full RDM to a triangular matrix (flattened upper triangle)."""
    return full_rdm[np.triu_indices(full_rdm.shape[0], k=1)]

def calculate_r_squared_loss(full_model, reduced_model):
    """Calculate R-squared loss between full and reduced models."""
    return full_model.rsquared - reduced_model.rsquared

def standardize_rdms(rdm_dict):
    """
    Standardize each RDM to zero mean and unit variance using upper triangular values.
    
    Parameters:
    -----------
    rdm_dict : dict
        Dictionary of {rdm_name: rdm_matrix}
        
    Returns:
    --------
    standardized : dict
        Dictionary of {rdm_name: standardized_flattened_rdm}
    """
    standardized = {}
    for name, rdm in rdm_dict.items():
        # Extract upper triangular part
        flat = get_triangular_matrix(rdm)
        
        # Handle case where std is zero (constant values)
        std = np.std(flat)
        if std == 0 or np.isclose(std, 0):
            # If all values are the same, return zeros (or the original values)
            flat_std = np.zeros_like(flat)
        else:
            # Standardize to zero mean and unit variance
            flat_std = (flat - np.mean(flat)) / std
            
        standardized[name] = flat_std
    return standardized

def construct_RDM(data, n_target, method = "euclidean", draw = True):
    '''
    Input:
        data: n x m matrix (DataFrame or numpy array), where n is the number of target and m is the number of features
        n_target: the number of target
        method: the method to calculate the distance matrix
            euclidean: Euclidean distance
            cityblock: Manhattan distance
            spearman: Spearman correlation
            cosine: Cosine distance
        draw: whether to draw the heatmap of the RDM
    Usage:
        construct_RDM(data, n_target, method = "euclidean")
    '''
    import numpy as np
    import pandas as pd

    # check if the method is supported
    method = method.lower()
    if method not in ["euclidean", "cityblock", "spearman", "cosine"]:
        raise ValueError(f"Unsupported method: {method}. Supported methods are: euclidean, cityblock, spearman, cosine.")

    # Unify data format to numpy array
    if isinstance(data, pd.DataFrame):
        data = data.values
    elif isinstance(data, (list, tuple)):
        data = np.array(data)
    else:
        data = np.asarray(data)
    
    # Convert to 2D array if needed
    if data.ndim == 1:
        data = data.reshape(-1, 1)  # make it (n_samples, 1) if it's a flat vector
    elif data.ndim > 2:
        raise ValueError(f"Data must be 1D or 2D, got {data.ndim}D array")

    # Remove rows with NaN or inf values
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        # Find rows with any NaN or inf values
        invalid_rows = np.any(np.isnan(data) | np.isinf(data), axis=1)
        data = data[~invalid_rows]
        print(f"Removed {np.sum(invalid_rows)} rows containing NaN or inf values")

    # Ensure shape is (n_target, features)
    if data.shape[0] == n_target:
        pass
    elif data.shape[1] == n_target:
        data = data.T
    else:
        raise ValueError(
            f'The input data does not have {n_target} non-NaN observations. '
            f'After cleaning, it has {data.shape[0]} rows and {data.shape[1]} columns.'
        )

    # Calculate RDM based on method
    if method == "spearman":
        if data.shape[1] < 2:
            raise ValueError("Spearman correlation requires at least 2 features (columns), but data has only 1 column.")
        corr_matrix, _ = spearmanr(data, axis=1, nan_policy='omit')
        # Handle case where spearmanr returns a scalar for single feature
        if np.isscalar(corr_matrix):
            corr_matrix = np.array([[1.0]])
        rdm = 1 - corr_matrix
    elif method == "cityblock":
        rdm = cdist(data, data, metric='cityblock')
    elif method == "cosine":
        rdm = cdist(data, data, metric='cosine')
    elif method == "euclidean":
        rdm = cdist(data, data, metric='euclidean')
    else:
        raise ValueError(f"Unsupported method: {method}")

    if draw:
        draw_heatmap(rdm, title=f"RDM ({method})", cmap="viridis", cbar=True)

    return rdm

def draw_heatmap(rdm, title = None, cmap = "viridis", cbar = True):
    '''
    Draw a heatmap of the RDM.
    Input:
        rdm: n x n matrix
        title: title of the heatmap
        cmap: colormap of the heatmap
        cbar: whether to show the colorbar
    '''
    # draw the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(rdm, cmap=cmap, cbar=cbar, square=True, annot=False)
    if title is not None:
        plt.title(title)
    plt.show()


def shuffle_rdm(rdm, random_state=None):
    """
    Shuffle the rows and columns of a square RDM to create a null distribution.
    
    Parameters:
    -----------
    rdm : np.ndarray
        Square distance matrix (RDM).
    random_state : int or None
        Random seed for reproducibility.
        
    Returns:
    --------
    shuffled_rdm : np.ndarray
        Shuffled RDM with the same shape as the input.
    """
    n = rdm.shape[0]
    if random_state is not None:
        np.random.seed(random_state)

    permuted_indices = np.random.permutation(n)
    shuffled_rdm = rdm[permuted_indices, :][:, permuted_indices]
    return shuffled_rdm

def mantel_permutation(matrix1, matrix2, n_permutations=1000, random_state=None):
    """
    Perform Mantel permutations to calculate Spearman correlations.

    Parameters:
    -----------
    matrix1 : np.ndarray
        First distance matrix (square, symmetric).
    matrix2 : np.ndarray
        Second distance matrix (square, symmetric, same size as matrix1).
    n_permutations : int
        Number of permutations.
    random_state : int or None
        Random seed for reproducibility.

    Returns:
    --------
    permuted_correlations : np.ndarray
        Array of permuted Spearman correlation values.
    observed_correlation : float
        Observed Spearman correlation between original matrices.
    p_value : float
        P-value representing significance of the observed correlation.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Get the upper triangular indices, flatten, and standardize the matrices
    vecs = standardize_rdms({'rdm1': matrix1, 'rdm2':matrix2})
    vec1 = vecs['rdm1']
    vec2 = vecs['rdm2']
    print(f"rdm1: {vec1} \n rdm2: {vec2}")

    observed_correlation, _ = spearmanr(vec1, vec2)

    # Perform permutations
    permuted_correlations = np.zeros(n_permutations)
    n = matrix1.shape[0]

    for i in range(n_permutations):
        permuted_matrix2 = shuffle_rdm(matrix2, random_state=random_state)
        perm_vec2 = standardize_rdms({'rdm2': permuted_matrix2})['rdm2']
        permuted_correlations[i], _ = spearmanr(vec1, perm_vec2)

    # Calculate p-value
    p_value = (np.sum(permuted_correlations >= observed_correlation) + 1) / (n_permutations + 1)

    return permuted_correlations, observed_correlation, p_value

def do_RSA(matrix1, matrix2, n_permutations=1000, random_state=None, plot_histogram=True):
    """
    Perform Mantel permutations to calculate Spearman correlations.

    Parameters:
    -----------
    matrix1 : np.ndarray
        First distance matrix (square, symmetric).
    matrix2 : np.ndarray
        Second distance matrix (square, symmetric, same size as matrix1).
    n_permutations : int
        Number of permutations.
    random_state : int or None
        Random seed for reproducibility.
    plot_histogram : bool
        Whether to plot the histogram of permutation results.

    Returns:
    --------
    permuted_correlations : np.ndarray
        Array of permuted Spearman correlation values.
    observed_correlation : float
        Observed Spearman correlation between original matrices.
    p_value : float
        P-value representing significance of the observed correlation.
    """
    assert matrix1.shape == matrix2.shape, "Matrices must have the same dimensions"
    assert matrix1.shape[0] == matrix1.shape[1], "Matrices must be square"

    permuted_correlations, observed_correlation, p_value = mantel_permutation(
        matrix1, matrix2, n_permutations=n_permutations, random_state=random_state
    )

    # draw the histogram
    permutation_histogram(observed_correlation, permuted_correlations, p_value)
    
    return permuted_correlations, observed_correlation, p_value

def permutation_histogram(r, perm_r, perm_p = None):
    '''
    Plot the histogram of permutation results on a created figure.
    '''
    # Filter out NaN or infinite values
    perm_r = perm_r[np.isfinite(perm_r)]
    
    if len(perm_r) == 0:
        raise ValueError("All values in perm_r are NaN or infinite. Cannot plot histogram.")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(perm_r, bins=50, color='gray')
    ax.axvline(r, color='red', alpha=0.5)
    ax.text(r-0.01, 5, f'Observed r = {r:.2}', color='red', fontsize=16)
    if perm_p is not None:
        # add the p-value to the plot on the top right corner
        # Add the p-value to the top right corner
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.text(xlim[1] * 0.95, ylim[1] * 0.95, f'p = {perm_p:.3}', 
                color='red', fontsize=16, ha='right', va='top')
    ax.set_xlabel('Permutation r', fontsize=20)
    ax.set_ylabel('Frequency', fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    # ax.set_title('Permutation distribution', fontsize=22)
    plt.show()

def maximal_permutation_test(data, iv_single, iv_multiplecomp, n_perm = 1000, method = "euclidean"):
    '''
    This fuction is used to address multiple comparison, \
        which provides an alternative of Bonferroni correction.
    
    - data: For IS-RSA, each row is a subject, while each column is a variable. \
        For example, if you have 20 subjects and 5 variables, the shape of data is (20, 5).
    - iv_single: the independent variable that will be shuffled and compare across iv_multiplecomp
    - iv_multiplecomp: the independent variable that are inter-related and elicit the multiple comparison problem
    - n_perm: number of permutation
    - method: the method to calculate the distance matrix
    '''

    # construct the RDMs
    # convert the data into array
    ivSarray = data[iv_single].values.reshape(-1, 1)
    rdmS = construct_RDM(ivSarray, data.shape[0], method=method)

    # remove the upper triangle
    rdmS_f = standardize_rdms({'rdmS': rdmS})['rdmS']

    # observed_r: dictionary to store the observed correlation
    observed_r = {}
    for ivM in iv_multiplecomp:
        # Construct the RDM for the independent variable component ivM.
        ivM_array = data[ivM].values.reshape(-1, 1)
        rdmM = construct_RDM(ivM_array, data.shape[0], method=method)
        rdmM_f = standardize_rdms({'rdmM': rdmM})['rdmM']

        # Compute the observed correlation between rdmS_f and rdmM_f.
        r, _ = spearmanr(rdmS_f, rdmM_f)
        observed_r[ivM] = r

    perm_r = np.zeros(n_perm)    
    for iperm in range(n_perm):
        # define the max r
        max_r_null = -np.inf

        for ivM in iv_multiplecomp:
            # calculate the psudo correlation
            r, _, _ = spearmanr(shuffle_rdm(rdmS_f), rdmM_f)

            # update the max r
            if r > max_r_null:
                max_r_null = r
            
        perm_r[iperm] = max_r_null
            
    # calculate the p-value
    perm_p = float(np.sum(perm_r > observed_r[ivM]) / n_perm)
    print(f"p = {np.sum(perm_r > observed_r[ivM])} / {n_perm}")

    return [perm_r, perm_p, observed_r]

def align_data(*data_inputs):
    """
    Align multiple datasets (NumPy arrays or Pandas DataFrames) based on shared row identifiers 
    and valid (non-NA) rows.

    Each input should be a dictionary with keys:
        - 'data': a numpy array or pandas DataFrame
        - 'order': optional list/array of row identifiers (must match rows in 'data')

    Returns:
        A list of aligned data objects (same type as original input), filtered to rows with:
            - shared row identifiers
            - no missing values across all datasets
    """
    aligned_data = []
    index_lists = []

    # Step 1: Create row index (from 'order' or just index range)
    for i, item in enumerate(data_inputs):
        data = item['data']
        order = item.get('order', None)
        n_rows = data.shape[0]

        if order is None:
            raise ValueError(f"Missing 'order' for input {i}. Please provide a list of row identifiers.")
        else:
            if len(order) != n_rows:
                raise ValueError(f"Length of 'order' does not match number of rows in data input {i}.")
            index = np.array(order)

        print(f"Index length: {len(index)}")
        index_lists.append(pd.Index(index))

    # Step 2: Find shared row identifiers
    shared_index = index_lists[0]
    for idx in index_lists[1:]:
        shared_index = shared_index.intersection(idx)

    shared_index = shared_index.sort_values()

    # Step 3: Align each dataset to the shared index
    for i, item in enumerate(data_inputs):
        data = item['data']
        index = index_lists[i]

        if isinstance(data, pd.DataFrame):
            data.index = index
            aligned = data.loc[shared_index].copy()
            aligned_data.append(aligned)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            elif data.ndim > 2:
                raise ValueError(f"Unexpected number of dimensions for input {i}: {data.ndim}")
            # use boolean indexing on array
            sort_idx = index.get_indexer(shared_index)
            aligned = data[sort_idx]
            # make sure the aligned data is 2D
            aligned_data.append(aligned)
        else:
            raise TypeError(f"Unsupported data type for input {i}: {type(data)}")

    # Step 4: Remove rows with any missing values across datasets
    keep_mask = np.ones(len(shared_index), dtype=bool)

    for d in aligned_data:
        if isinstance(d, pd.DataFrame):
            keep_mask &= ~d.isnull().any(axis=1).to_numpy()
        else:  # numpy array
            keep_mask &= ~np.isnan(d).any(axis=1)

    for i in range(len(aligned_data)):
        if isinstance(aligned_data[i], pd.DataFrame):
            aligned_data[i] = aligned_data[i].iloc[keep_mask].reset_index(drop=True)
            print(f"The {i}th aligned data is a DataFrame with shape: {aligned_data[i].shape}")
        else: # for numpy array
            aligned_data[i] = aligned_data[i][keep_mask]
            print(f"The {i}th aligned data is a NumPy array with shape: {aligned_data[i].shape}")

    return aligned_data

def variance_partitioning(DV_rdms, rdm_dict, plot_title='RDMs Contributions', print_results=False):
    """
    Performs regression analysis to compare the contributions of multiple RDMs to dependent variable RDMs.

    Parameters:
    -----------
    DV_rdms : dict
        Dictionary of dependent variable RDMs {dv_name: rdm_matrix}
    rdm_dict : dict
        Dictionary of predictor RDMs {rdm_name: rdm_matrix}
    plot_title : str, default='RDMs Contributions'
        Title for the contribution plot
    print_results : bool, default=False
        Whether to print detailed regression summaries
        
    Returns:
    --------
    results_df : pandas.DataFrame
        DataFrame with R-squared results and contributions
    """
    # Check for empty inputs
    if not DV_rdms:
        raise ValueError("DV_rdms cannot be empty")
    if not rdm_dict:
        raise ValueError("rdm_dict cannot be empty")
        
    # Check for empty RDMs within dictionaries
    for name, rdm in DV_rdms.items():
        if rdm is None or (hasattr(rdm, 'size') and rdm.size == 0):
            raise ValueError(f"DV RDM '{name}' is empty or None")
            
    for name, rdm in rdm_dict.items():
        if rdm is None or (hasattr(rdm, 'size') and rdm.size == 0):
            raise ValueError(f"Predictor RDM '{name}' is empty or None")

    # Standardize all RDMs before regression
    rdm_dict = standardize_rdms(rdm_dict)
    DV_rdms = standardize_rdms(DV_rdms)

    rdm_names = list(rdm_dict.keys())
    X = pd.DataFrame(rdm_dict)
    results = []
    column_names = (['DV', 'Full R-squared'] + 
                   [f'{name} Exclusive Contribution' for name in rdm_names] + 
                   [f'{name} P-value' for name in rdm_names])

    for dv_name, dv_vector in DV_rdms.items():
        df = pd.concat([pd.Series(dv_vector, name='dvRDM'), X], axis=1)
        df = clean_data_df(df)

        # Full model
        full_model = sm.OLS(df['dvRDM'], sm.add_constant(df[rdm_names])).fit()
        if print_results:
            print(f"\n============== DV: {dv_name} ==============")
            print(full_model.summary())

        row_result = [dv_name, full_model.rsquared]

        # Exclusive contribution of each predictor
        for excl_name in rdm_names:
            reduced_names = [name for name in rdm_names if name != excl_name]
            reduced_model = sm.OLS(df['dvRDM'], sm.add_constant(df[reduced_names])).fit()
            loss = calculate_r_squared_loss(full_model, reduced_model)
            row_result.append(loss)
            if print_results:
                print(f"\n{excl_name} Exclusive Contribution: {loss:.4f}")

        # Add p-values for each predictor from the full model
        for name in rdm_names:
            p_value = full_model.pvalues[name]
            row_result.append(p_value)
            if print_results:
                print(f"\n{name} P-value: {p_value:.4f}")

        results.append(row_result)

    # Results DataFrame
    results_df = pd.DataFrame(results, columns=column_names).set_index('DV')
    results_df['Overlapped'] = (results_df['Full R-squared'] - 
                               results_df[[f'{name} Exclusive Contribution' for name in rdm_names]].sum(axis=1))

    # Plotting
    _plot_variance_contributions(results_df, rdm_names, plot_title)

    return results_df

def _plot_variance_contributions(results_df, rdm_names, plot_title):
    """
    Helper function to plot variance contribution results.
    """
    bottom = np.zeros(len(results_df))
    x = np.arange(len(results_df))

    # Define consistent colors for RDMs
    rdm_colors = {
        'Conceptual': '#1f77b4',  # Blue
        'Physical': '#2ca02c',    # Green
        'Overlapped': '#CCCCCC',  # Gray
        'Static Facial': '#81c784',
    }

    # Prepare color cycle for additional RDMs
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    used_colors = set(rdm_colors.values())
    color_iter = (c for c in color_cycle if c.lower() not in [col.lower() for col in used_colors])

    # Assign colors to all RDMs
    assigned_colors = {}
    for name in rdm_names:
        if name in rdm_colors:
            assigned_colors[name] = rdm_colors[name]
        else:
            assigned_colors[name] = next(color_iter, '#1f77b4')  # Default to blue if cycle exhausted

    # Plot exclusive contributions
    for name in rdm_names:
        contrib = results_df[f'{name} Exclusive Contribution']
        plt.bar(x, contrib, label=f'{name} Features', 
                color=assigned_colors[name], bottom=bottom)
        bottom += contrib.values

    # Plot overlapped contribution
    plt.bar(x, results_df['Overlapped'], label='Overlapped', 
            color=assigned_colors.get('Overlapped', '#CCCCCC'), 
            alpha=0.6, bottom=bottom)

    # Formatting
    plt.xticks(x, results_df.index, rotation=90, fontsize=16)
    plt.ylabel('R-squared', fontsize=16)
    plt.title(plot_title, fontsize=16)
    plt.ylim(0, results_df['Full R-squared'].max() + 0.1)
    plt.legend()
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()

    plt.show()
