# Data utilities

```python
from yescarpenter import align_data, clean_data_df, clean_data_np
```

## align_data

```python
align_data(*data_inputs)
```

Puts several datasets on the same footing: it keeps only the rows whose identifier
occurs in **every** dataset, puts them in the same order, and drops rows that have a
missing value in **any** dataset.

Each positional argument is a dictionary with two keys:

| Key | Description |
| --- | --- |
| `'data'` | A `numpy.ndarray` (1D or 2D) or a `pandas.DataFrame`. |
| `'order'` | List or array of row identifiers, one per row of `data`. Required. |

**Returns** a list of aligned datasets, in the order the inputs were given. Rows are
sorted by identifier, so row `i` refers to the same identifier in every output.

**Behavior**

- Any number of datasets may be passed.
- A 1D array is returned as a 2D column of shape `(n, 1)`.
- A DataFrame is returned as a copy with its index reset to `0..n-1`; the identifiers
  are not kept as the index. Your original DataFrame is not modified.
- Progress is printed: the length of each input and the shape of each output.
- Raises `ValueError` if `'order'` is missing, its length differs from the number of rows,
  or an array has more than two dimensions. Raises `TypeError` for any other data type.

**Example**

```python
aligned_votes, aligned_response, aligned_semantic = align_data(
    {"data": votes["Vote share percentage"].values, "order": votes["Image_name"].values},
    {"data": response,                              "order": response_names},
    {"data": semantic_embeddings,                   "order": semantic_names},
)
print(aligned_votes.shape, aligned_response.shape, aligned_semantic.shape)
```

## clean_data_df

```python
clean_data_df(df)
```

Returns a copy of a `pandas.DataFrame` with every row that contains `NaN` or `±inf`
removed. The index is not reset.

## clean_data_np

```python
clean_data_np(data)
```

Removes non-finite values from a NumPy array (or anything `numpy.asarray` accepts).

- **1D input:** the individual `NaN` / `±inf` elements are dropped.
- **2D or higher:** any row (first axis) that contains a non-finite value is dropped;
  the remaining axes keep their shape.

Returns a `numpy.ndarray`.
