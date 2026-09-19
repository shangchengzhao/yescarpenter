import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from yescarpenter.pca import create_scree_plot, pc_plot, perform_pca, scree_plot


def _two_factor_data(n=300, seed=0):
    """Six variables: a1-a3 driven by one latent factor, b1-b3 by an independent one."""
    rng = np.random.default_rng(seed)
    f1, f2 = rng.normal(size=(2, n))
    cols = {f'a{i + 1}': 0.9 * f1 + 0.3 * rng.normal(size=n) for i in range(3)}
    cols.update({f'b{i + 1}': 0.9 * f2 + 0.3 * rng.normal(size=n) for i in range(3)})
    return pd.DataFrame(cols)


class TestPerformPCA:
    def setup_method(self):
        self.df = _two_factor_data()

    def test_output_shapes(self):
        loadings, ev, comps = perform_pca(self.df, 2)
        assert loadings.shape == (6, 2)
        assert ev.shape == (2,)
        assert comps.shape == (300, 2)

    def test_dataframe_and_array_give_same_result(self):
        for a, b in zip(perform_pca(self.df, 2), perform_pca(self.df.values, 2)):
            np.testing.assert_allclose(a, b)

    def test_explained_variance_is_sorted_and_bounded(self):
        _, ev, _ = perform_pca(self.df, 4)
        assert np.all(np.diff(ev) <= 1e-12)
        assert np.all(ev > 0)
        assert ev.sum() <= 1 + 1e-12

    def test_explained_variance_sums_to_one_with_all_components(self):
        _, ev, _ = perform_pca(self.df, 6)
        assert ev.sum() == pytest.approx(1.0)

    def test_explained_variance_matches_the_returned_loadings(self):
        loadings, ev, _ = perform_pca(self.df, 3)
        np.testing.assert_allclose(ev, (loadings ** 2).sum(axis=0) / 6)

    def test_scores_are_unit_variance_and_uncorrelated(self):
        _, _, comps = perform_pca(self.df, 3)
        np.testing.assert_allclose(np.cov(comps.T, ddof=0), np.eye(3), atol=1e-6)

    def test_total_variance_equals_plain_pca_total(self):
        # rotation redistributes variance between components but not the total
        z = (self.df - self.df.mean()) / self.df.std(ddof=0)
        eig = np.sort(np.linalg.eigvalsh(np.cov(z.values.T, ddof=0)))[::-1]
        _, ev, _ = perform_pca(self.df, 3)
        assert ev.sum() == pytest.approx(eig[:3].sum() / eig.sum())

    def test_loadings_are_orthogonally_rotated_pca_loadings(self):
        loadings, _, _ = perform_pca(self.df, 6)
        # full-rank: loadings @ loadings.T reproduces the correlation matrix
        z = (self.df - self.df.mean()) / self.df.std(ddof=0)
        np.testing.assert_allclose(loadings @ loadings.T, np.corrcoef(z.values.T), atol=1e-8)

    def test_scores_reconstruct_standardized_data(self):
        loadings, _, comps = perform_pca(self.df, 6)
        z = (self.df - self.df.mean()) / self.df.std(ddof=0)
        np.testing.assert_allclose(comps @ loadings.T, z.values, atol=1e-8)

    def test_loadings_are_correlations_with_the_scores(self):
        loadings, _, comps = perform_pca(self.df, 2)
        z = ((self.df - self.df.mean()) / self.df.std(ddof=0)).values
        corr = np.array([[np.corrcoef(z[:, j], comps[:, k])[0, 1] for k in range(2)] for j in range(6)])
        np.testing.assert_allclose(corr, loadings, atol=1e-6)

    def test_two_factors_capture_most_variance(self):
        _, ev, _ = perform_pca(self.df, 2)
        assert ev.sum() > 0.85

    def test_varimax_loadings_separate_the_two_variable_groups(self):
        loadings, _, _ = perform_pca(self.df, 2)
        loadings = np.abs(loadings)
        a_factor = int(np.argmax(loadings[:3].mean(axis=0)))
        b_factor = 1 - a_factor
        assert loadings[:3, a_factor].min() > 0.8
        assert loadings[3:, b_factor].min() > 0.8
        assert loadings[:3, b_factor].max() < 0.25
        assert loadings[3:, a_factor].max() < 0.25

    def test_invariant_to_column_scale_and_shift(self):
        # data are standardized internally
        scaled = self.df * np.array([1, 10, 100, 0.1, 5, 2]) + 50
        for a, b in zip(perform_pca(self.df, 2), perform_pca(scaled, 2)):
            np.testing.assert_allclose(np.abs(a), np.abs(b), atol=1e-6)

    def test_does_not_mutate_input(self):
        before = self.df.copy()
        perform_pca(self.df, 2)
        pd.testing.assert_frame_equal(self.df, before)

        arr = self.df.values.copy()
        arr_before = arr.copy()
        perform_pca(arr, 2)
        np.testing.assert_array_equal(arr, arr_before)

    def test_single_component(self):
        loadings, ev, comps = perform_pca(self.df, 1)
        assert loadings.shape == (6, 1)
        assert ev.shape == (1,)
        assert comps.shape == (300, 1)

    def test_too_many_components_raises(self):
        with pytest.raises(ValueError):
            perform_pca(self.df, 7)

    def test_nan_input_raises(self):
        bad = self.df.copy()
        bad.iloc[0, 0] = np.nan
        with pytest.raises(ValueError):
            perform_pca(bad, 2)


class TestScreePlot:
    def test_scree_plot_draws_one_point_per_component(self):
        ev = np.array([0.5, 0.3, 0.2])
        scree_plot(ev, 3)
        ax = plt.gca()
        line = ax.lines[0]
        np.testing.assert_array_equal(line.get_xdata(), [1, 2, 3])
        np.testing.assert_array_equal(line.get_ydata(), ev)
        assert ax.get_xlabel() == 'Number of components'
        assert ax.get_ylabel() == 'Explained variance'
        assert ax.get_title() == 'Scree plot'

    def test_create_scree_plot_plots_unrotated_eigenvalue_spectrum(self):
        df = _two_factor_data()
        create_scree_plot(df, 4)
        z = (df - df.mean()) / df.std(ddof=0)
        eig = np.sort(np.linalg.eigvalsh(np.cov(z.values.T, ddof=0)))[::-1]
        np.testing.assert_allclose(plt.gca().lines[0].get_ydata(), (eig / eig.sum())[:4], atol=1e-6)


class TestPCPlot:
    def setup_method(self):
        self.df = _two_factor_data()
        self.loadings, _, _ = perform_pca(self.df, 2)

    def test_one_panel_per_component_with_trait_labels(self):
        pc_plot(self.loadings, self.df)
        fig = plt.gcf()
        # the colorbar adds one extra axes
        panels = [ax for ax in fig.axes if ax.get_title() in ('PC1', 'PC2')]
        assert [ax.get_title() for ax in panels] == ['PC1', 'PC2']
        labels = [t.get_text() for t in panels[0].get_yticklabels()]
        assert labels == list(self.df.columns)

    def test_bars_use_the_loading_values(self):
        pc_plot(self.loadings, self.df)
        panel = [ax for ax in plt.gcf().axes if ax.get_title() == 'PC1'][0]
        widths = sorted(p.get_width() for p in panel.patches)
        np.testing.assert_allclose(widths, sorted(self.loadings[:, 0]))

    def test_three_components(self):
        loadings, _, _ = perform_pca(self.df, 3)
        pc_plot(loadings, self.df)
        titles = [ax.get_title() for ax in plt.gcf().axes]
        assert {'PC1', 'PC2', 'PC3'} <= set(titles)

    def test_single_component(self):
        # regression: plt.subplots(1, 1) returns a bare Axes, which pc_plot used to iterate over
        loadings, _, _ = perform_pca(self.df, 1)
        pc_plot(loadings, self.df)
        assert 'PC1' in [ax.get_title() for ax in plt.gcf().axes]
