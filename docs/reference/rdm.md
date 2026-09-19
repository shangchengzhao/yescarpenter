# RDM construction

```python
from yescarpenter import (construct_RDM, draw_heatmap, get_triangular_matrix,
                          standardize_rdms, shuffle_rdm)
```

A representational dissimilarity matrix (RDM) is a square, symmetric matrix whose entry
`(i, j)` is the distance between target `i` and target `j`, so the diagonal is zero.
"Target" means whatever the rows represent: stimuli, subjects, conditions.

## construct_RDM

```python
construct_RDM(data, n_target=None, method="euclidean", draw=True, target_axis=0)
```

Builds an RDM from a table of measurements, one row (or column) per target.

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `data` | `DataFrame`, `ndarray`, list or tuple | | 2D data. A 1D input is treated as `n` targets with a single feature. |
| `n_target` | `int`, optional | `None` | Expected number of targets. Used only for validation: a `ValueError` is raised if it differs from the number of targets found, **after** non-finite targets are removed. |
| `method` | `str` | `"euclidean"` | Distance metric, case-insensitive. One of the four below. |
| `draw` | `bool` | `True` | Display a heatmap of the RDM. |
| `target_axis` | `0` or `1` | `0` | Which axis of `data` holds the targets. `0`: rows are targets (targets x features). `1`: columns are targets (features x targets). Ignored for 1D input. |

**Methods**

| `method` | Dissimilarity |
| --- | --- |
| `"euclidean"` | Euclidean distance |
| `"cityblock"` | Manhattan distance |
| `"cosine"` | Cosine distance |
| `"spearman"` | `1 - ρ`, where ρ is the Spearman correlation between two targets' feature vectors. Needs at least two features. |

**Returns** a `numpy.ndarray` of shape `(n_target, n_target)`.

**Behavior**

- Targets containing `NaN` or `inf` are dropped, and the number removed is printed.
  Because the matrix shrinks, its indices no longer line up with the original rows; use
  [`align_data`](data-utils.md#align_data) beforehand if you need to keep track of identity.
- Unsupported `method`, `target_axis` outside `{0, 1}`, or data with more than two
  dimensions raise `ValueError`.

**Example**

```python
import numpy as np
from yescarpenter import construct_RDM

data = np.random.rand(10, 5)   # 10 pictures x 5 ratings
rdm = construct_RDM(data, n_target=10, method="euclidean")

# Targets in columns (features x targets): say so explicitly.
rdm = construct_RDM(data.T, n_target=10, method="euclidean", target_axis=1)
```

## draw_heatmap

```python
draw_heatmap(rdm, title=None, cmap="viridis", cbar=True)
```

Displays a square heatmap of an RDM, or of any square matrix.

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `rdm` | `numpy.ndarray` | | Matrix to draw. |
| `title` | `str`, optional | `None` | Figure title. |
| `cmap` | `str` | `"viridis"` | Any matplotlib colormap name. |
| `cbar` | `bool` | `True` | Show the color bar. |

Returns `None`.

## get_triangular_matrix

```python
get_triangular_matrix(full_rdm)
```

Returns the upper triangle of a square matrix, excluding the diagonal, as a 1D array
in row-major order. An `n x n` RDM gives `n * (n - 1) / 2` values. Because RDMs are
symmetric, these values contain all the information in the matrix.

## standardize_rdms

```python
standardize_rdms(rdm_dict)
```

Z-scores each RDM using its upper-triangle values.

| Name | Type | Description |
| --- | --- | --- |
| `rdm_dict` | `dict` | `{name: square_matrix}`. |

**Returns** a `dict` `{name: vector}`. Note that the values are the **flattened**
upper-triangle vectors, not matrices. If an RDM is constant (standard deviation of
zero), its vector is all zeros.

## shuffle_rdm

```python
shuffle_rdm(rdm, random_state=None, rng=None)
```

Applies one random permutation to both the rows and the columns of a square matrix.
The result is still a valid RDM with the same set of distances, but with the targets'
identities scrambled. This is the building block of the permutation tests.

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `rdm` | `numpy.ndarray` | | Square matrix. |
| `random_state` | `int`, optional | `None` | Seed for reproducibility. |
| `rng` | `numpy.random.Generator`, optional | `None` | An existing generator. When given, `random_state` is ignored. Pass one generator through repeated calls to get different shuffles from a single seed. |

**Returns** the shuffled `numpy.ndarray`, same shape as the input.
