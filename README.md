# yescarpenter
This library provides some frequent used functions for YESlab members and other researchers, including data processing and analyses

## Installation

### From PyPI
Users can install it using pip:

```bash
pip install yescarpenter
```

## Functions

### perform_pca
This function leverages scikit-learn to perform a tailored PCA analysis (e.g., with rotation to maximize variance)

Usage:

```python
import pandas as pd
from yescarpenter import perform_pca

# Create a sample DataFrame
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4],
    'feature2': [2, 3, 4, 5],
    'feature3': [3, 4, 5, 6]
})

# Perform PCA with 2 components
loadings, explained_variance, components = perform_pca(data, n_components=2)

print("Loadings:\n", loadings)
print("Explained Variance:\n", explained_variance)
print("Components:\n", components)
```

### create_scree_plot

This function creates a scree plot to visualize the explained variance of each principal component.

```python
scree_plot(explained_variance, n_components)
```

### pc_plot
Create a plot to visualize the PCA loadings.

```python
pc_plot(loadings, df)
```

## RSA Functions

### clean_data_df
This function removes rows containing NaN or infinite values from a pandas DataFrame, ensuring clean data for analysis.

**def clean_data_df(df):**

    """
    Remove rows with NaN or inf values in DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame to clean
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned DataFrame with NaN and infinite values removed
    """

### clean_data_np
This function removes rows containing NaN or infinite values from a numpy array, ensuring clean data for analysis.

**def clean_data_np(data):**

    """
    Remove rows with NaN or inf values from numpy array.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input array to clean
        
    Returns:
    --------
    numpy.ndarray
        Cleaned array. For 1D input, non-finite elements are dropped; for 2D+
        input, rows containing any NaN or infinite value are dropped.
    """

### get_triangular_matrix
This function extracts the upper triangular part of a square matrix (excluding the diagonal), commonly used in RDM analysis.

**def get_triangular_matrix(full_rdm):**

    """
    Convert a full RDM to a triangular matrix (flattened upper triangle).
    
    Parameters:
    -----------
    full_rdm : numpy.ndarray
        Square matrix (RDM) to extract upper triangle from
        
    Returns:
    --------
    numpy.ndarray
        1D array containing upper triangular values (excluding diagonal)
    """

### standardize_rdms
This function standardizes RDMs to zero mean and unit variance using their upper triangular values, preparing them for statistical analysis.

**def standardize_rdms(rdm_dict):**

    """
    Standardize each RDM to zero mean and unit variance using upper triangular values.
    
    Parameters:
    -----------
    rdm_dict : dict
        Dictionary of {rdm_name: rdm_matrix}
        
    Returns:
    --------
    dict
        Dictionary of {rdm_name: standardized_flattened_rdm}
    """

### construct_RDM
This function constructs a Representational Dissimilarity Matrix (RDM) from data using various distance metrics.

**def construct_RDM(data, n_target=None, method="euclidean", draw=True, target_axis=0):**

    """
    Construct a Representational Dissimilarity Matrix (RDM) from data.
    
    Parameters:
    -----------
    data : array-like
        2D input data. A 1D input is treated as n targets with a single feature.
    n_target : int, optional
        Expected number of (valid) targets/stimuli. Only used for validation:
        a ValueError is raised if it does not match the data.
    method : str, default="euclidean"
        Distance metric to use ('euclidean', 'cityblock', 'cosine', 'spearman')
    draw : bool, default=True
        Whether to display a heatmap of the resulting RDM
    target_axis : {0, 1}, default=0
        Axis of `data` that holds the targets. 0: rows are targets (targets x features);
        1: columns are targets (features x targets). Ignored for 1D input.
        
    Returns:
    --------
    numpy.ndarray
        Square RDM matrix of shape (n_target, n_target)
    """

Usage:
```python
# Example usage
import numpy as np
from yescarpenter import construct_RDM
data = np.random.rand(10, 5)  # 10 pictures, 5 ratings
n_target = 10
rdm = construct_RDM(data, n_target, method="euclidean")

# If the targets are the columns (features x targets), say so explicitly
rdm = construct_RDM(data.T, n_target, method="euclidean", target_axis=1)
```

### draw_heatmap
This function creates a heatmap visualization of an RDM or any square matrix.

**def draw_heatmap(rdm, title=None, cmap="viridis", cbar=True):**

    """
    Draw a heatmap of the RDM.
    
    Parameters:
    -----------
    rdm : numpy.ndarray
        Square matrix to visualize
    title : str, optional
        Title for the heatmap
    cmap : str, default="viridis"
        Colormap for the heatmap
    cbar : bool, default=True
        Whether to show the colorbar
        
    Returns:
    --------
    None
        Displays the heatmap plot
    """

### shuffle_rdm
This function randomly permutes the rows and columns of an RDM while maintaining its structure, used for permutation testing.

**def shuffle_rdm(rdm, random_state=None):**

    """
    Randomly shuffle the rows and columns of an RDM.
    
    Parameters:
    -----------
    rdm : numpy.ndarray
        Square RDM matrix to shuffle
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    numpy.ndarray
        Shuffled RDM matrix
    """

### mantel_permutation
This function performs Mantel permutation tests to assess the statistical significance of correlations between two RDMs.

**def mantel_permutation(matrix1, matrix2, n_permutations=1000, random_state=None):**

    """
    Perform Mantel permutations to calculate Spearman correlations.
    
    Parameters:
    -----------
    matrix1 : numpy.ndarray
        First distance matrix (square, symmetric)
    matrix2 : numpy.ndarray
        Second distance matrix (square, symmetric, same size as matrix1)
    n_permutations : int, default=1000
        Number of permutations for statistical testing
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    permuted_correlations : numpy.ndarray
        Array of permuted Spearman correlation values
    observed_correlation : float
        Observed Spearman correlation between original matrices
    p_value : float
        One-tailed (right-tail) permutation p-value; see the note below
    """

**Assumption: the p-value is one-tailed.** It is computed as `p = (sum(permuted_correlations >= observed_correlation) + 1) / (n_permutations + 1)`, i.e. the proportion of permutations (counting the observed one) whose correlation is at least as large as the observed correlation. This tests only for a *positive* correspondence between the two RDMs. A strong negative correlation yields a p-value close to 1, not a small one. No two-tailed option is currently exposed. If you need one, compare `abs(permuted_correlations) >= abs(observed_correlation)` using the returned `permuted_correlations` and `observed_correlation`.


### do_RSA
This function calculates the Spearman correlation between two RDMs and performs Mantel permutation testing to assess statistical significance.

**def do_RSA(matrix1, matrix2, n_permutations=1000, random_state=None, plot_histogram=True):**

    """
    Perform Representational Similarity Analysis (RSA) with permutation testing.
    
    Parameters:
    -----------
    matrix1 : numpy.ndarray
        First distance matrix (square, symmetric)
    matrix2 : numpy.ndarray
        Second distance matrix (square, symmetric, same size as matrix1)
    n_permutations : int, default=1000
        Number of permutations for statistical testing
    random_state : int, optional
        Random seed for reproducibility
    plot_histogram : bool, default=True
        Whether to display the permutation histogram
        
    Returns:
    --------
    permuted_correlations : numpy.ndarray
        Array of permuted Spearman correlation values
    observed_correlation : float
        Observed Spearman correlation between original matrices
    p_value : float
        One-tailed (right-tail) permutation p-value; see the note below
    """

**Note:** as in `mantel_permutation`, the returned `p_value` is one-tailed (right-tail) and only tests for a positive correspondence between the RDMs. See the assumption above.


Usage:

```python
do_RSA(rdm1, rdm2, n_permutations=1000, random_state=None)
```

### permutation_histogram
This function plots the histogram of null distribution from permutation testing, with the observed value and p-value marked.

**def permutation_histogram(r, perm_r, perm_p=None):**

    """
    Plot the histogram of permutation results.
    
    Parameters:
    -----------
    r : float
        Observed correlation value
    perm_r : numpy.ndarray
        Array of permuted correlation values (null distribution)
    perm_p : float, optional
        P-value to display on the plot
        
    Returns:
    --------
    None
        Displays the histogram plot
    """

Usage:

```python
permutation_histogram(r, perm_r)
```
- `r`: The observed value.
- `perm_r`: The null distribution, which consists of the iterated surrogated values

### maximal_permutation_test
This function addresses multiple comparison problems by using the maximal statistic approach, providing an alternative to Bonferroni correction.

**def maximal_permutation_test(data, iv_single, iv_multiplecomp, n_perm=1000, method="euclidean", random_state=None):**

    """
    Perform maximal permutation test for multiple comparison correction.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data where each row is a subject and each column is a variable
    iv_single : str
        Column name of the independent variable to be permuted
    iv_multiplecomp : list
        List of column names for variables involved in multiple comparisons
    n_perm : int, default=1000
        Number of permutations
    method : str, default="euclidean"
        Distance metric for RDM construction
    random_state : int or None
        Random seed for reproducibility
        
    Returns:
    --------
    perm_r : numpy.ndarray
        Array of maximal permuted correlation values
    perm_p : dict
        Adjusted p-value for each variable in iv_multiplecomp
    observed_r : dict
        Dictionary of observed correlations for each comparison
    """

Usage:

```python
[perm_r, perm_p, observed_r] = maximal_permutation_test(data, iv_single, iv_multiplecomp, nperm)
```
- data: For IS-RSA, each row is a subject, while each column is a variable. For example, if you have 20 subjects and 5 variables, the shape of data is (20, 5).
- iv_single: the independent variable that will be shuffled and compare across iv_multiplecomp
- iv_multiplecomp: the independent variable that are inter-related and elicit the multiple comparison problem
- n_perm: number of permutation

### align_data
This function aligns different sources of data based on shared identifiers, handling missing values and ensuring consistent ordering across datasets.

**def align_data(*data_inputs):**

    """
    Align multiple datasets based on shared row identifiers and valid (non-NA) rows.
    
    Parameters:
    -----------
    *data_inputs : dict
        Variable number of dictionaries, each containing:
        - 'data': numpy array or pandas DataFrame
        - 'order': list/array of row identifiers matching the data rows (required)
        
    Returns:
    --------
    list
        List of aligned data objects (same type as input), filtered to shared identifiers
    """

Usage:

```python
aligned_cong_fec, aligned_vgg, aligned_sem = align_data(
    {'data': cong_fec['Vote share percentage'].values, 'order': cong_fec['Image_name'].values},
    {'data': response, 'order': vgglist['image_name'].values},
    {'data': sememb, 'order': sem_imgname}
)

print(aligned_cong_fec.shape, aligned_vgg.shape, aligned_sem.shape)
```

- You can put as many data as you want. Each input has to be a dictionary with two keys: 'data' and 'order'.

### variance_partitioning
This function performs variance partitioning analysis, which is useful for understanding how much variance in a dependent variable can be explained by one or more predictors.

**def variance_partitioning(DV_rdms, rdm_dict, plot_title='RDMs Contributions', print_results=False):**

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

### calculate_r_squared_loss
This utility function calculates the difference in R-squared values between full and reduced regression models, used in variance partitioning.

**def calculate_r_squared_loss(full_model, reduced_model):**

    """
    Calculate R-squared loss between full and reduced models.
    
    Parameters:
    -----------
    full_model : statsmodels regression model
        The full regression model
    reduced_model : statsmodels regression model
        The reduced regression model (with one predictor removed)
        
    Returns:
    --------
    float
        Difference in R-squared values (exclusive contribution)
    """