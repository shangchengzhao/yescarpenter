# Installation

yescarpenter needs Python 3.6 or newer.

## From PyPI

```bash
pip install yescarpenter
```

## From source

```bash
git clone https://github.com/shangchengzhao/yescarpenter.git
cd yescarpenter
pip install -e .
```

The `-e` flag installs in editable mode, so changes to the source take effect without
reinstalling.

## Dependencies

These are installed automatically:

| Package | Used for |
| --- | --- |
| numpy, pandas | array and table handling |
| scipy | distance metrics and Spearman correlation |
| scikit-learn | PCA |
| statsmodels | regression in variance partitioning |
| matplotlib, seaborn | plots |

## Check the install

```python
import yescarpenter
print(yescarpenter.__all__)
```

## Building this documentation

The documentation is built with [MkDocs](https://www.mkdocs.org/) and the Material theme.

```bash
pip install -r docs/requirements.txt
mkdocs serve      # live preview at http://127.0.0.1:8000
mkdocs build      # static site written to site/
```
