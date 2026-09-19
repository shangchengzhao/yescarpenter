# Walkthrough

Two short end-to-end examples. Each block can be pasted into a notebook as is.

## 1. PCA on a table of traits

```python
import numpy as np
import pandas as pd
from yescarpenter import create_scree_plot, perform_pca, pc_plot, scree_plot

rng = np.random.default_rng(0)
traits = pd.DataFrame(rng.normal(size=(100, 6)),
                      columns=["warm", "kind", "smart", "skilled", "bold", "loud"])

# 1. How many components? Look for the elbow.
create_scree_plot(traits, max_components=5)

# 2. Fit the chosen number of components.
loadings, explained_variance, scores = perform_pca(traits, n_components=2)

# 3. Inspect the loadings, one panel per component.
pc_plot(loadings, traits)
```

`loadings` has shape `(n_variables, n_components)` and `scores` has shape
`(n_observations, n_components)`. See [PCA](reference/pca.md) for what each output
contains.

## 2. Representational similarity analysis

The question: do two sets of measurements organise the same 20 items in a similar way?

```python
import numpy as np
from yescarpenter import construct_RDM, do_RSA

rng = np.random.default_rng(1)
ratings = rng.normal(size=(20, 8))                 # 20 items x 8 ratings
model = ratings + rng.normal(scale=0.5, size=(20, 8))  # a noisy copy, 20 items x 8 features

# 1. One RDM per data source (items are rows).
rdm_ratings = construct_RDM(ratings, method="euclidean", draw=False)
rdm_model = construct_RDM(model, method="cosine", draw=False)

# 2. Correlate the two RDMs and test with a Mantel permutation test.
perm_r, observed_r, p = do_RSA(rdm_ratings, rdm_model,
                               n_permutations=1000, random_state=42)
print(f"r = {observed_r:.3f}, p = {p:.4f}")
```

`do_RSA` also draws the null distribution with the observed correlation marked. Pass
`plot_histogram=False` to suppress it.

### Aligning data first

When the two sources cover different items, or contain gaps, align them before building
RDMs. [`align_data`](reference/data-utils.md#align_data) keeps only the identifiers present in every
source and drops rows with missing values:

```python
from yescarpenter import align_data

ratings_aligned, model_aligned = align_data(
    {"data": ratings, "order": item_ids_ratings},
    {"data": model,   "order": item_ids_model},
)
```

### Comparing many variables against one

If one variable is compared against several related ones, the p-values need a multiple
comparison correction.
[`maximal_permutation_test`](reference/rsa.md#maximal_permutation_test) does this with the
maximal-statistic approach. [`variance_partitioning`](reference/variance-partitioning.md) instead asks how much of
one RDM each of several predictor RDMs explains uniquely.
