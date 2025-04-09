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

### construct_RDM
For IS-RSA. Construct the Representational Dissimilarity Matrix(RDM) from the data.

Usage:

```python
construct_RDM(data, n_target, method = "cityblock")
```

- `data`: The input data for RDM construction.
- `method`: cityblock, or spearman

### do_rsa

Calculate the Spearman correlation between two RDMs(lower triangle) and do permutation

Usage:

```python        
do_RSA(rdm1, rdm2, n_perm=1000)
```

### permutation_histogram

Plot the histogram of null distribution, with the observed value and p-value marked.

Usage:

```python
permutation_histogram(r, perm_r)
```
- `r`: The observed value.
- `perm_r`: The null distribution, which consists of the iterated surrogated values

### maximal_permutation_test
This fuction is used to address multiple comparison, which provides an alternative of Bonferroni correction.


Usage:

```python
[perm_r, perm_p, observed_r] = maximal_permutation_test(data, iv_single, iv_multiplecomp, nperm)
```
- data: For IS-RSA, each row is a subject, while each column is a variable. \
        For example, if you have 20 subjects and 5 variables, the shape of data is (20, 5).
    - iv_single: the independent variable that will be shuffled and compare across iv_multiplecomp
    - iv_multiplecomp: the independent variable that are inter-related and elicit the multiple comparison problem
    - n_perm: number of permutation