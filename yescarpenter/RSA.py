import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist, pdist
import itertools
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm

def clean_data_df(df):
    return df.replace([np.inf, -np.inf], np.nan).dropna()

def clean_data_np(data):
    data = np.asarray(data)
    finite = np.isfinite(data)
    if data.ndim > 1:
        finite = finite.reshape(data.shape[0], -1).all(axis=1)
    return data[finite]

def get_triangular_matrix(full_rdm):
    return full_rdm[np.triu_indices(full_rdm.shape[0], k=1)]

def calculate_r_squared_loss(full_model, reduced_model):
    return full_model.rsquared - reduced_model.rsquared

def standardize_rdms(rdm_dict):
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

def construct_RDM(data, n_target = None, method = "euclidean", draw = True, target_axis = 0):
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
    
    if target_axis not in (0, 1):
        raise ValueError(f"target_axis must be 0 (targets are rows) or 1 (targets are columns), got {target_axis!r}")

    # Convert to 2D array (targets x features)
    if data.ndim == 1:
        data = data.reshape(-1, 1)  # make it (n_samples, 1) if it's a flat vector
    elif data.ndim == 2:
        if target_axis == 1:
            data = data.T
    else:
        raise ValueError(f"Data must be 1D or 2D, got {data.ndim}D array")

    # Remove targets (rows) with NaN or inf values
    if not np.all(np.isfinite(data)):
        invalid_rows = ~np.isfinite(data).all(axis=1)
        data = data[~invalid_rows]
        print(f"Removed {np.sum(invalid_rows)} targets (rows) containing NaN or inf values")

    # Validate the number of (valid) targets
    if n_target is not None and data.shape[0] != n_target:
        raise ValueError(
            f"Expected {n_target} valid targets along axis {target_axis}, but found {data.shape[0]} "
            f"after removing non-finite targets. Check n_target and target_axis."
        )

    # Calculate RDM based on method
    if method == "spearman":
        if data.shape[1] < 2:
            raise ValueError("Spearman correlation requires at least 2 features (columns), but data has only 1 column.")
        corr_result, _ = spearmanr(data, axis=1, nan_policy='omit')
        # Handle case where spearmanr returns a scalar for single feature
        if np.isscalar(corr_result):
            corr_matrix = np.array([[1.0]])
        else:
            corr_matrix = np.asarray(corr_result)  # type: ignore[arg-type]
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
    # draw the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(rdm, cmap=cmap, cbar=cbar, square=True, annot=False)
    if title is not None:
        plt.title(title)
    plt.show()


def shuffle_rdm(rdm, random_state=None, rng = None):
    n = rdm.shape[0]
    if rng is None:
        rng = np.random.default_rng(random_state)

    permuted_indices = rng.permutation(n)
    shuffled_rdm = rdm[permuted_indices, :][:, permuted_indices]
    return shuffled_rdm

def mantel_permutation(matrix1, matrix2, n_permutations=1000, random_state=None):

    # Get the upper triangular indices, flatten, and standardize the matrices
    vecs = standardize_rdms({'rdm1': matrix1, 'rdm2':matrix2})
    vec1 = vecs['rdm1']
    vec2 = vecs['rdm2']
    # print(f"rdm1: {vec1} \n rdm2: {vec2}")

    observed_corr_result, _ = spearmanr(vec1, vec2)
    observed_correlation = float(observed_corr_result)  # type: ignore[arg-type]

    # Perform permutations
    permuted_correlations = np.zeros(n_permutations)
    rng = np.random.default_rng(random_state)
    for i in range(n_permutations):
        permuted_matrix2 = shuffle_rdm(matrix2, rng=rng)
        perm_vec2 = standardize_rdms({'rdm2': permuted_matrix2})['rdm2']
        perm_corr, _ = spearmanr(vec1, perm_vec2)
        permuted_correlations[i] = float(perm_corr)  # type: ignore[arg-type]

    # Calculate p-value
    p_value = (np.sum(permuted_correlations >= observed_correlation) + 1) / (n_permutations + 1)

    return permuted_correlations, observed_correlation, p_value

def do_RSA(matrix1, matrix2, n_permutations=1000, random_state=None, plot_histogram=True):
    assert matrix1.shape == matrix2.shape, "Matrices must have the same dimensions"
    assert matrix1.shape[0] == matrix1.shape[1], "Matrices must be square"

    permuted_correlations, observed_correlation, p_value = mantel_permutation(
        matrix1, matrix2, n_permutations=n_permutations, random_state=random_state
    )

    # draw the histogram
    if plot_histogram:
        permutation_histogram(observed_correlation, permuted_correlations, p_value)
    
    return permuted_correlations, observed_correlation, p_value

def permutation_histogram(r, perm_r, perm_p = None):
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

def maximal_permutation_test(data, iv_single, iv_multiplecomp, n_perm = 1000, method = "euclidean", random_state = None):

    # construct the RDM of iv_single (kept as a square matrix so that it can be shuffled)
    ivSarray = data[iv_single].values.reshape(-1, 1)
    rdmS = construct_RDM(ivSarray, data.shape[0], method=method, draw=False)
    rdmS_f = standardize_rdms({'rdmS': rdmS})['rdmS']

    # construct the (flattened) RDM of each iv, and the observed correlation with rdmS_f
    rdmM_f_dict = {}
    observed_r = {}
    for ivM in iv_multiplecomp:
        ivM_array = data[ivM].values.reshape(-1, 1)
        rdmM = construct_RDM(ivM_array, data.shape[0], method=method, draw=False)
        rdmM_f_dict[ivM] = standardize_rdms({'rdmM': rdmM})['rdmM']

        r_result, _ = spearmanr(rdmS_f, rdmM_f_dict[ivM])
        observed_r[ivM] = float(r_result)  # type: ignore[arg-type]

    perm_r = np.zeros(n_perm)
    rng = np.random.default_rng(random_state)
    for iperm in range(n_perm):
        # shuffle rdmS once, shared by all the comparisons in this permutation
        perm_rdmS_f = standardize_rdms({'rdmS': shuffle_rdm(rdmS, rng=rng)})['rdmS']

        max_r_null = -np.inf
        for ivM in iv_multiplecomp:
            r_result, _ = spearmanr(perm_rdmS_f, rdmM_f_dict[ivM])
            r = float(r_result)  # type: ignore[arg-type]

            # update the max r
            if r > max_r_null:
                max_r_null = r

        perm_r[iperm] = max_r_null

    # adjusted p-value for each iv, aligned with mantel_permutation
    perm_p = {}
    for ivM in iv_multiplecomp:
        perm_p[ivM] = float((np.sum(perm_r >= observed_r[ivM]) + 1) / (n_perm + 1))
        print(f"{ivM}: r = {observed_r[ivM]:.3f}, adjusted p = {perm_p[ivM]:.4f}")

    return [perm_r, perm_p, observed_r]

def align_data(*data_inputs):
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
            # relabel a copy so the caller's DataFrame keeps its original index
            aligned = data.set_axis(index, axis=0).loc[shared_index].copy()
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

def variance_partitioning(DV_rdms, rdm_dict, plot_title='RDMs Contributions', print_results=False, colors=None):
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
    _plot_variance_contributions(results_df, rdm_names, plot_title, colors)

    return results_df

def _plot_variance_contributions(results_df, rdm_names, plot_title, colors=None):
    bottom = np.zeros(len(results_df))
    x = np.arange(len(results_df))

    colors = dict(colors) if colors else {}
    overlap_color = colors.get('Overlapped', '#CCCCCC')

    # Fall back to the default color cycle for predictors without a supplied color.
    # to_hex normalizes tuple/hex/named colors so they can be compared safely.
    user_hex = {mcolors.to_hex(c).lower() for c in colors.values()}
    user_hex.add(mcolors.to_hex(overlap_color).lower())
    cycle = [mcolors.to_hex(c) for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
    free_colors = [c for c in cycle if c.lower() not in user_hex] or cycle
    color_iter = itertools.cycle(free_colors)

    assigned_colors = {name: colors[name] if name in colors else next(color_iter)
                       for name in rdm_names}

    # Plot exclusive contributions
    for name in rdm_names:
        contrib = results_df[f'{name} Exclusive Contribution']
        plt.bar(x, contrib, label=f'{name} Features', 
                color=assigned_colors[name], bottom=bottom)
        bottom += contrib.values

    # Plot overlapped contribution
    plt.bar(x, results_df['Overlapped'], label='Overlapped', 
            color=overlap_color, 
            alpha=0.6, bottom=bottom)

    # Formatting
    plt.xticks(x, results_df.index, rotation=90, fontsize=16)
    plt.ylabel('R-squared', fontsize=16)
    plt.title(plot_title, fontsize=16)
    plt.ylim(0, results_df['Full R-squared'].max() + 0.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()

    plt.show()
