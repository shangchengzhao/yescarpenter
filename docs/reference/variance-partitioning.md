# Variance partitioning

```python
from yescarpenter import variance_partitioning, calculate_r_squared_loss
```

## variance_partitioning

```python
variance_partitioning(DV_rdms, rdm_dict, plot_title="RDMs Contributions",
                      print_results=False, colors=None)
```

Asks how much of each dependent-variable RDM is explained by a set of predictor RDMs,
and how much of that is unique to each predictor.

For each dependent RDM the function:

1. z-scores the upper triangle of every RDM (dependent and predictors);
2. fits an ordinary least squares regression of the dependent RDM on all predictors
   (the *full model*), dropping any cell with a missing value;
3. refits the model once per predictor with that predictor left out;
4. takes the drop in R² as that predictor's **exclusive contribution**;
5. reports what remains, `Full R² - sum(exclusive contributions)`, as the **overlapped**
   variance that the predictors share.

A stacked bar chart (one bar per dependent RDM) is then displayed.

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `DV_rdms` | `dict` | | `{dv_name: square_matrix}`. One analysis per entry. |
| `rdm_dict` | `dict` | | `{predictor_name: square_matrix}`. All matrices must have the same size as those in `DV_rdms`. |
| `plot_title` | `str` | `"RDMs Contributions"` | Title of the bar chart. |
| `print_results` | `bool` | `False` | Print the full statsmodels summary and each contribution. |
| `colors` | `dict`, optional | `None` | `{predictor_name: color}` for the bars. Any matplotlib color spec works. The key `"Overlapped"` sets the color of the overlapped segment (default gray). Predictors without an entry take colors from the matplotlib default cycle, skipping colors you already used. |

**Returns** a `pandas.DataFrame` with one row per dependent RDM, indexed by `DV`:

| Column | Description |
| --- | --- |
| `Full R-squared` | R² of the model containing every predictor. |
| `<name> Exclusive Contribution` | R² lost when `<name>` is removed. One column per predictor. |
| `<name> P-value` | p-value of `<name>`'s coefficient in the full model. One column per predictor. |
| `Overlapped` | `Full R-squared` minus the sum of the exclusive contributions. |

Raises `ValueError` if either dictionary is empty or contains an empty or `None` matrix.

!!! Note: Descriptive, not inferential
    The result of this analysis is more descriptive than a strict statistical test. The
    R² values and exclusive contributions summarize how the predictor RDMs account for the
    dependent RDM in your data; they do not come with a formal test of significance.

!!! Caution when interpreting the p-values
    The p-values come from ordinary regression, which treats the cells of an RDM as
    independent observations. They are not, because every target contributes to many
    cells. Treat them as descriptive; use a permutation test such as
    [`do_RSA`](rsa.md#do_rsa) for inference on a single RDM pair.

**Example**

```python
results = variance_partitioning(
    DV_rdms={"neural": rdm_neural},
    rdm_dict={"visual": rdm_visual, "semantic": rdm_semantic},
    plot_title="Neural RDM",
    colors={"visual": "#1f77b4", "semantic": "#ff7f0e", "Overlapped": "lightgray"},
)
print(results.loc["neural", ["Full R-squared", "Overlapped"]])
```

## calculate_r_squared_loss

```python
calculate_r_squared_loss(full_model, reduced_model)
```

Returns `full_model.rsquared - reduced_model.rsquared`, the exclusive contribution of the
predictor that is missing from `reduced_model`. This is the helper used in step 4 above.

| Name | Type | Description |
| --- | --- | --- |
| `full_model` | fitted statsmodels regression result | Model with all predictors. |
| `reduced_model` | fitted statsmodels regression result | Same model with one predictor removed. |

Returns a `float`.
