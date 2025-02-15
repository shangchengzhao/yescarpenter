# yescarpenter
This library provides some frequent used functions for YESlab members and other researchers, including data processing and analyses

## Installation

### From PyPI
Once the package is uploaded to PyPI, users can install it using pip:

```bash
pip install yescarpenter
```

## Functions

### perform_pca
This function leverages scikit-learn along with other popular data science libraries.

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