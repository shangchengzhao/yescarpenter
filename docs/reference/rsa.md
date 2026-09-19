# RSA and permutation tests

```python
from yescarpenter import (do_RSA, mantel_permutation, permutation_histogram,
                          maximal_permutation_test)
```

These functions test whether two RDMs are more similar than chance. The null
distribution is built by shuffling the targets of one RDM (a Mantel test), which keeps
the dependence between the cells of a distance matrix intact.

!!! note "The p-value is one-tailed"
    `mantel_permutation`, `do_RSA` and `maximal_permutation_test` all compute

    ```
    p = (sum(permuted_correlations >= observed_correlation) + 1) / (n_permutations + 1)
    ```

    This tests only for a *positive* correspondence between the RDMs. A strong
    negative correlation gives a p-value near 1, not a small one. There is no
    two-tailed option. To get one, use the returned values:

    ```python
    p_two_tailed = (np.sum(np.abs(perm_r) >= abs(observed_r)) + 1) / (len(perm_r) + 1)
    ```

## do_RSA

```python
do_RSA(matrix1, matrix2, n_permutations=1000, random_state=None, plot_histogram=True)
```

Correlates two RDMs (Spearman, on their upper triangles) and tests the correlation with
a Mantel permutation test. This is the main entry point for a two-RDM analysis.

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `matrix1` | `numpy.ndarray` | | Square, symmetric distance matrix. |
| `matrix2` | `numpy.ndarray` | | Square, symmetric, same shape as `matrix1`. |
| `n_permutations` | `int` | `1000` | Number of shuffles. |
| `random_state` | `int`, optional | `None` | Seed for reproducibility. |
| `plot_histogram` | `bool` | `True` | Display the null distribution with [`permutation_histogram`](#permutation_histogram). |

**Returns** a tuple `(permuted_correlations, observed_correlation, p_value)`:

| Name | Type | Description |
| --- | --- | --- |
| `permuted_correlations` | `numpy.ndarray`, shape `(n_permutations,)` | The null distribution. |
| `observed_correlation` | `float` | Spearman correlation of the two RDMs. |
| `p_value` | `float` | One-tailed permutation p-value (see above). |

Raises `AssertionError` if the matrices differ in shape or are not square.

```python
perm_r, observed_r, p = do_RSA(rdm1, rdm2, n_permutations=1000, random_state=42)
```

## mantel_permutation

```python
mantel_permutation(matrix1, matrix2, n_permutations=1000, random_state=None)
```

The computation behind `do_RSA`, without the shape checks and without the plot. It
takes the same parameters (minus `plot_histogram`) and returns the same three values.
In each permutation only `matrix2` is shuffled; `matrix1` stays fixed.

## permutation_histogram

```python
permutation_histogram(r, perm_r, perm_p=None)
```

Plots the null distribution as a histogram, with the observed value marked by a red
line.

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `r` | `float` | | Observed correlation. |
| `perm_r` | `numpy.ndarray` | | Null distribution. Non-finite values are ignored. |
| `perm_p` | `float`, optional | `None` | If given, printed in the top-right corner of the plot. |

Returns `None`. Raises `ValueError` if `perm_r` contains no finite values.

## maximal_permutation_test

```python
maximal_permutation_test(data, iv_single, iv_multiplecomp,
                         n_perm=1000, method="euclidean", random_state=None)
```

Tests one variable against several related variables while controlling for the
multiple comparisons, as an alternative to a Bonferroni correction. It is designed for
individual-differences RSA, where each row of `data` is a subject and each column is a
variable.

For every variable, an RDM is built across subjects. In each permutation the RDM of
`iv_single` is shuffled **once**, and that same shuffled RDM is correlated with the RDM of
every variable in `iv_multiplecomp`. The largest of those correlations goes into the
null distribution. Each variable's observed correlation is then compared with this
distribution of maxima, which yields an adjusted p-value.

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `data` | `pandas.DataFrame` | | One row per subject, one column per variable. For 20 subjects and 5 variables the shape is `(20, 5)`. |
| `iv_single` | `str` | | Column name of the variable that is shuffled and compared against all the others. |
| `iv_multiplecomp` | `list` of `str` | | Column names of the variables that are compared with `iv_single` and that cause the multiple-comparison problem. |
| `n_perm` | `int` | `1000` | Number of permutations. |
| `method` | `str` | `"euclidean"` | Distance metric used to build each variable's RDM. |
| `random_state` | `int`, optional | `None` | Seed for reproducibility. |

**Returns** a list `[perm_r, perm_p, observed_r]`:

| Name | Type | Description |
| --- | --- | --- |
| `perm_r` | `numpy.ndarray`, shape `(n_perm,)` | Null distribution of the maximal correlation. |
| `perm_p` | `dict` | Adjusted one-tailed p-value for each variable in `iv_multiplecomp`. |
| `observed_r` | `dict` | Observed Spearman correlation with `iv_single` for each variable in `iv_multiplecomp`. |

**Behavior**

- Each variable is one column, so each RDM is built from a single feature. Use
  `"euclidean"` or `"cityblock"`. `"spearman"` raises an error because it needs at
  least two features, and `"cosine"` on single values is degenerate.
- The function prints `r` and the adjusted p-value for every variable.
- An RDM heatmap is drawn for each variable, because the RDMs are built with
  `construct_RDM`'s default `draw=True`.

```python
perm_r, perm_p, observed_r = maximal_permutation_test(
    data, iv_single="age", iv_multiplecomp=["iq", "score"], n_perm=1000, random_state=0
)
```
