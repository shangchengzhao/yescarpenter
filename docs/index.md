# yescarpenter

yescarpenter collects the analysis functions that YESlab members use most often, so
they can be imported instead of copied between notebooks. It currently covers two areas:

| Area | What it does | Functions |
| --- | --- | --- |
| [PCA](reference/pca.md) | Varimax-rotated PCA with scree and loading plots | `perform_pca`, `scree_plot`, `create_scree_plot`, `pc_plot` |
| [Representational similarity analysis](reference/rdm.md) | Build representational dissimilarity matrices (RDMs), correlate them, and test the correlation with permutations | `construct_RDM`, `do_RSA`, `mantel_permutation`, `maximal_permutation_test`, `variance_partitioning`, and helpers |

Every function is importable from the top-level package:

```python
from yescarpenter import perform_pca, construct_RDM, do_RSA
```

## Where to go next

- [Installation](installation.md): install from PyPI or from source.
- [Walkthrough](walkthrough.md): a PCA example and an RSA example from start to finish.
- Reference: one page per group of functions, with parameters, return values and caveats.

## Conventions used across the library

- **Plotting functions call `plt.show()`.** In a script this blocks until the window is
  closed. In a notebook the figure appears inline. Functions that draw by default
  (`construct_RDM`, `do_RSA`) accept a flag to turn the figure off.
- **Randomness is controlled by `random_state`.** Pass an integer to make permutation
  results reproducible.
- **RDMs are square, symmetric distance matrices** with one row and column per target
  (stimulus, subject, and so on).
