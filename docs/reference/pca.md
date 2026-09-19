# PCA

```python
from yescarpenter import perform_pca, scree_plot, create_scree_plot, pc_plot
```

## perform_pca

```python
perform_pca(data, n_components)
```

Standardizes the data, runs PCA, and applies a varimax rotation. Loadings, scores and
explained variance all describe the same rotated components.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `data` | `pandas.DataFrame` or `numpy.ndarray` | Observations in rows, variables in columns. Columns are z-scored internally. Must not contain missing values. |
| `n_components` | `int` | Number of components to keep. |

**Returns** a tuple `(loadings, explained_variance, components)`:

| Name | Shape | Description |
| --- | --- | --- |
| `loadings` | `(n_variables, n_components)` | Varimax-rotated loadings (correlations between variables and components). |
| `explained_variance` | `(n_components,)` | Proportion of total variance carried by each rotated component, sorted in descending order. |
| `components` | `(n_observations, n_components)` | Unit-variance scores of each observation on the rotated components. |

!!! note "Rotation and variance"
    A varimax rotation redistributes variance between components but keeps the total,
    so the ratios here differ from those of the unrotated PCA. `explained_variance`
    sums to the same value as the unrotated ratios for the same `n_components`, and
    with all components it sums to 1. `create_scree_plot` deliberately plots the
    unrotated PCA spectrum, because that is what a scree plot is for.

**Example**

```python
import pandas as pd
from yescarpenter import perform_pca

data = pd.DataFrame({
    "feature1": [1, 2, 3, 4, 5],
    "feature2": [2, 1, 4, 3, 6],
    "feature3": [5, 3, 4, 1, 2],
})
loadings, explained_variance, components = perform_pca(data, n_components=2)
```

## scree_plot

```python
scree_plot(explained_variance, n_components)
```

Plots explained variance against component number, for choosing how many components
to keep.

| Name | Type | Description |
| --- | --- | --- |
| `explained_variance` | array-like | Explained variance ratios, for example the second output of `perform_pca`. |
| `n_components` | `int` | Number of values to plot. Must equal `len(explained_variance)`. |

Returns `None`; displays the figure.

## create_scree_plot

```python
create_scree_plot(data, max_components)
```

Convenience wrapper: runs `perform_pca(data, max_components)` and passes the explained
variance to `scree_plot`. Use it to inspect the elbow before deciding on
`n_components`.

| Name | Type | Description |
| --- | --- | --- |
| `data` | `pandas.DataFrame` or `numpy.ndarray` | Same input as `perform_pca`. |
| `max_components` | `int` | Number of components to fit and plot. Usually the number of variables. |

Returns `None`; displays the figure.

## pc_plot

```python
pc_plot(loadings, df)
```

Draws one horizontal bar chart per component, showing each variable's loading. Bars
share one diverging color scale, so colors are comparable across panels.

| Name | Type | Description |
| --- | --- | --- |
| `loadings` | `numpy.ndarray` | Loadings of shape `(n_variables, n_components)`, for example the first output of `perform_pca`. |
| `df` | `pandas.DataFrame` | The original data. Its column names label the bars, in the same order as the rows of `loadings`. |

Returns `None`; displays the figure.

!!! warning "Limitations"
    - The bars are labeled with the column names of `df`, and the figure size is fixed
      (15 x 6 inches), so many components or many variables get crowded.
    - The call `sns.set(style="whitegrid")` changes seaborn and matplotlib styling for
      the rest of the session.
