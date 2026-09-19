import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from yescarpenter.RSA import (
    align_data,
    calculate_r_squared_loss,
    clean_data_df,
    clean_data_np,
    construct_RDM,
    get_triangular_matrix,
    mantel_permutation,
    maximal_permutation_test,
    permutation_histogram,
    shuffle_rdm,
)


def _rdm(values):
    """Euclidean RDM of a 1D vector of values."""
    return squareform(pdist(np.asarray(values, dtype=float).reshape(-1, 1)))


class TestCleanDataDf:
    def test_drops_nan_and_inf_rows(self):
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0, 4.0], 'b': [1.0, 2.0, np.inf, -np.inf]})
        cleaned = clean_data_df(df)
        assert cleaned.index.tolist() == [0]

    def test_clean_input_unchanged(self):
        df = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
        pd.testing.assert_frame_equal(clean_data_df(df), df)

    def test_does_not_modify_input(self):
        df = pd.DataFrame({'a': [1.0, np.inf]})
        clean_data_df(df)
        assert np.isinf(df.loc[1, 'a'])


class TestCleanDataNp:
    def test_1d_drops_individual_elements(self):
        out = clean_data_np(np.array([1.0, np.nan, 3.0, np.inf, -np.inf]))
        np.testing.assert_array_equal(out, [1.0, 3.0])

    def test_2d_drops_whole_rows_and_keeps_shape(self):
        data = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0], [7.0, np.inf]])
        out = clean_data_np(data)
        np.testing.assert_array_equal(out, [[1.0, 2.0], [5.0, 6.0]])
        assert out.ndim == 2 and out.shape[1] == 2

    def test_3d_drops_rows_along_first_axis(self):
        data = np.arange(24, dtype=float).reshape(4, 3, 2)
        data[2, 1, 0] = np.nan
        out = clean_data_np(data)
        assert out.shape == (3, 3, 2)
        np.testing.assert_array_equal(out, np.delete(data, 2, axis=0))

    def test_all_finite_is_unchanged(self):
        data = np.arange(6, dtype=float).reshape(3, 2)
        np.testing.assert_array_equal(clean_data_np(data), data)

    def test_all_rows_dropped_gives_empty_with_columns(self):
        out = clean_data_np(np.full((3, 2), np.nan))
        assert out.shape == (0, 2)

    def test_accepts_list_input(self):
        out = clean_data_np([[1, 2], [np.nan, 3]])
        np.testing.assert_array_equal(out, [[1, 2]])

    def test_integer_array(self):
        data = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_equal(clean_data_np(data), data)


class TestGetTriangularMatrix:
    def test_upper_triangle_row_major_order(self):
        m = np.array([[0, 1, 2, 3],
                      [1, 0, 4, 5],
                      [2, 4, 0, 6],
                      [3, 5, 6, 0]])
        np.testing.assert_array_equal(get_triangular_matrix(m), [1, 2, 3, 4, 5, 6])

    def test_length_is_n_choose_2(self):
        for n in (2, 5, 10):
            assert get_triangular_matrix(np.zeros((n, n))).shape == (n * (n - 1) // 2,)

    def test_excludes_diagonal(self):
        m = np.eye(4) * 99
        assert np.all(get_triangular_matrix(m) == 0)

    def test_matches_pdist_ordering(self):
        x = np.random.default_rng(0).normal(size=(6, 3))
        np.testing.assert_allclose(get_triangular_matrix(squareform(pdist(x))), pdist(x))


class TestCalculateRSquaredLoss:
    def test_difference_of_rsquared(self):
        full = SimpleNamespace(rsquared=0.8)
        reduced = SimpleNamespace(rsquared=0.5)
        assert calculate_r_squared_loss(full, reduced) == pytest.approx(0.3)

    def test_zero_when_equal(self):
        m = SimpleNamespace(rsquared=0.4)
        assert calculate_r_squared_loss(m, m) == 0


class TestShuffleRDM:
    def setup_method(self):
        self.rdm = _rdm([1, 4, 9, 16, 25, 36])

    def test_preserves_shape_symmetry_and_diagonal(self):
        out = shuffle_rdm(self.rdm, random_state=0)
        assert out.shape == self.rdm.shape
        np.testing.assert_allclose(out, out.T)
        np.testing.assert_array_equal(np.diag(out), 0)

    def test_preserves_multiset_of_distances(self):
        out = shuffle_rdm(self.rdm, random_state=0)
        np.testing.assert_allclose(np.sort(get_triangular_matrix(out)),
                                   np.sort(get_triangular_matrix(self.rdm)))

    def test_is_a_joint_row_column_permutation(self):
        out = shuffle_rdm(self.rdm, random_state=3)
        perm = np.random.default_rng(3).permutation(self.rdm.shape[0])
        np.testing.assert_array_equal(out, self.rdm[perm][:, perm])

    def test_same_seed_reproducible(self):
        np.testing.assert_array_equal(shuffle_rdm(self.rdm, random_state=7),
                                      shuffle_rdm(self.rdm, random_state=7))

    def test_different_seeds_differ(self):
        outs = [shuffle_rdm(self.rdm, random_state=s) for s in range(5)]
        assert any(not np.array_equal(outs[0], o) for o in outs[1:])

    def test_does_not_mutate_input(self):
        original = self.rdm.copy()
        shuffle_rdm(self.rdm, random_state=1)
        np.testing.assert_array_equal(self.rdm, original)

    def test_rng_argument_is_used_and_advances(self):
        rng = np.random.default_rng(5)
        first = shuffle_rdm(self.rdm, rng=rng)
        second = shuffle_rdm(self.rdm, rng=rng)
        expected_rng = np.random.default_rng(5)
        np.testing.assert_array_equal(first, shuffle_rdm(self.rdm, rng=expected_rng))
        np.testing.assert_array_equal(second, shuffle_rdm(self.rdm, rng=expected_rng))

    def test_rng_takes_precedence_over_random_state(self):
        a = shuffle_rdm(self.rdm, random_state=1, rng=np.random.default_rng(2))
        b = shuffle_rdm(self.rdm, rng=np.random.default_rng(2))
        np.testing.assert_array_equal(a, b)


class TestMantelPermutation:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.x = rng.normal(size=12)
        self.rdm1 = _rdm(self.x)
        self.rdm_related = _rdm(self.x + rng.normal(scale=0.1, size=12))
        self.rdm_unrelated = _rdm(rng.normal(size=12))

    def test_output_types_and_lengths(self):
        perm, obs, p = mantel_permutation(self.rdm1, self.rdm_related, n_permutations=50, random_state=0)
        assert perm.shape == (50,)
        assert isinstance(obs, float)
        assert 0 < p <= 1

    def test_identical_matrices_have_r_of_one(self):
        _, obs, p = mantel_permutation(self.rdm1, self.rdm1, n_permutations=200, random_state=0)
        assert obs == pytest.approx(1.0)
        assert p < 0.05

    def test_observed_matches_spearman_of_upper_triangles(self):
        _, obs, _ = mantel_permutation(self.rdm1, self.rdm_related, n_permutations=5, random_state=0)
        expected = spearmanr(get_triangular_matrix(self.rdm1), get_triangular_matrix(self.rdm_related))[0]
        assert obs == pytest.approx(expected)

    def test_related_more_significant_than_unrelated(self):
        _, _, p_rel = mantel_permutation(self.rdm1, self.rdm_related, n_permutations=300, random_state=0)
        _, _, p_unrel = mantel_permutation(self.rdm1, self.rdm_unrelated, n_permutations=300, random_state=0)
        assert p_rel < p_unrel

    def test_same_seed_reproducible(self):
        a = mantel_permutation(self.rdm1, self.rdm_related, n_permutations=30, random_state=42)
        b = mantel_permutation(self.rdm1, self.rdm_related, n_permutations=30, random_state=42)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:] == b[1:]

    def test_p_value_formula(self):
        perm, obs, p = mantel_permutation(self.rdm1, self.rdm_related, n_permutations=40, random_state=1)
        assert p == pytest.approx((np.sum(perm >= obs) + 1) / 41)

    def test_p_value_never_zero(self):
        _, _, p = mantel_permutation(self.rdm1, self.rdm1, n_permutations=20, random_state=0)
        assert p >= 1 / 21

    def test_does_not_mutate_inputs(self):
        m1, m2 = self.rdm1.copy(), self.rdm_related.copy()
        mantel_permutation(self.rdm1, self.rdm_related, n_permutations=10, random_state=0)
        np.testing.assert_array_equal(self.rdm1, m1)
        np.testing.assert_array_equal(self.rdm_related, m2)


class TestPermutationHistogram:
    def test_draws_histogram(self):
        import matplotlib.pyplot as plt
        permutation_histogram(0.3, np.random.default_rng(0).normal(size=100), perm_p=0.04)
        ax = plt.gcf().axes[0]
        assert len(ax.patches) > 0
        assert any('p = ' in t.get_text() for t in ax.texts)

    def test_without_p_value(self):
        import matplotlib.pyplot as plt
        permutation_histogram(0.3, np.random.default_rng(0).normal(size=100))
        assert not any('p = ' in t.get_text() for t in plt.gcf().axes[0].texts)

    def test_ignores_non_finite_values(self):
        perm = np.array([0.1, np.nan, 0.2, np.inf, -0.1])
        permutation_histogram(0.3, perm)

    def test_all_non_finite_raises(self):
        with pytest.raises(ValueError, match="NaN or infinite"):
            permutation_histogram(0.3, np.array([np.nan, np.inf]))


class TestMaximalPermutationTest:
    def setup_method(self):
        rng = np.random.default_rng(0)
        n = 15
        base = rng.normal(size=n)
        self.data = pd.DataFrame({
            'single': base,
            'related': base + rng.normal(scale=0.2, size=n),
            'unrelated1': rng.normal(size=n),
            'unrelated2': rng.normal(size=n),
        })

    def _run(self, ivs, n_perm=60, seed=0, **kw):
        return maximal_permutation_test(self.data, 'single', ivs, n_perm=n_perm, random_state=seed, **kw)

    def test_return_structure(self):
        ivs = ['related', 'unrelated1']
        perm_r, perm_p, observed_r = self._run(ivs)
        assert perm_r.shape == (60,)
        assert set(perm_p) == set(ivs)
        assert set(observed_r) == set(ivs)

    def test_single_iv_multiplecomp_works(self):
        # regression: a single comparison used to raise IndexError
        perm_r, perm_p, observed_r = self._run(['related'])
        assert list(perm_p) == ['related']

    def test_p_values_in_valid_range(self):
        _, perm_p, _ = self._run(['related', 'unrelated1', 'unrelated2'], n_perm=50)
        for p in perm_p.values():
            assert 1 / 51 <= p <= 1

    def test_p_value_formula(self):
        ivs = ['related', 'unrelated1']
        perm_r, perm_p, observed_r = self._run(ivs, n_perm=40)
        for iv in ivs:
            assert perm_p[iv] == pytest.approx((np.sum(perm_r >= observed_r[iv]) + 1) / 41)

    def test_observed_r_matches_spearman_of_rdms(self):
        _, _, observed_r = self._run(['related', 'unrelated1'], n_perm=5)
        rdm_s = get_triangular_matrix(_rdm(self.data['single']))
        for iv in observed_r:
            expected = spearmanr(rdm_s, get_triangular_matrix(_rdm(self.data[iv])))[0]
            assert observed_r[iv] == pytest.approx(expected)

    def test_related_iv_has_higher_r_and_lower_p(self):
        _, perm_p, observed_r = self._run(['related', 'unrelated1', 'unrelated2'], n_perm=200)
        assert observed_r['related'] > max(observed_r['unrelated1'], observed_r['unrelated2'])
        assert perm_p['related'] < 0.05
        assert perm_p['related'] < min(perm_p['unrelated1'], perm_p['unrelated2'])

    def test_same_seed_reproducible(self):
        a = self._run(['related', 'unrelated1'], seed=11)
        b = self._run(['related', 'unrelated1'], seed=11)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1] == b[1]
        assert a[2] == b[2]

    def test_different_seeds_change_null(self):
        a = self._run(['related', 'unrelated1'], seed=1)
        b = self._run(['related', 'unrelated1'], seed=2)
        assert not np.array_equal(a[0], b[0])

    def test_null_is_max_over_ivs(self):
        # with the same seed, the same permutations are drawn, so the maximal null
        # must be >= the null of any subset of the ivs
        full, _, _ = self._run(['related', 'unrelated1', 'unrelated2'], n_perm=40, seed=3)
        subset, _, _ = self._run(['unrelated1'], n_perm=40, seed=3)
        assert np.all(full >= subset - 1e-12)

    def test_single_iv_null_matches_mantel_permutation(self):
        perm_r, perm_p, observed_r = self._run(['related'], n_perm=40, seed=5)
        # mantel_permutation shuffles its second matrix, so pass 'single' second
        mantel_perm, mantel_obs, mantel_p = mantel_permutation(
            _rdm(self.data['related']), _rdm(self.data['single']), n_permutations=40, random_state=5)
        assert observed_r['related'] == pytest.approx(mantel_obs)
        assert perm_p['related'] == pytest.approx(mantel_p)
        np.testing.assert_allclose(perm_r, mantel_perm, atol=1e-12)

    def test_adjusted_p_not_smaller_than_unadjusted(self):
        ivs = ['related', 'unrelated1', 'unrelated2']
        _, perm_p, _ = self._run(ivs, n_perm=100, seed=4)
        for iv in ivs:
            _, _, p_single = mantel_permutation(
                _rdm(self.data['single']), _rdm(self.data[iv]), n_permutations=100, random_state=4)
            assert perm_p[iv] >= p_single - 1e-12

    def test_method_is_passed_to_rdm_construction(self):
        # for a single column cityblock == euclidean; an invalid method must raise
        a = self._run(['related'], n_perm=10, method='euclidean')
        b = self._run(['related'], n_perm=10, method='cityblock')
        np.testing.assert_allclose(a[0], b[0])
        with pytest.raises(ValueError):
            self._run(['related'], n_perm=2, method='not_a_metric')

    def test_does_not_mutate_data(self):
        before = self.data.copy()
        self._run(['related', 'unrelated1'], n_perm=5)
        pd.testing.assert_frame_equal(self.data, before)

    def test_missing_column_raises(self):
        with pytest.raises(KeyError):
            self._run(['nonexistent'])


class TestAlignData:
    def test_intersects_and_sorts_shared_ids(self):
        df1 = pd.DataFrame({'x': [1.0, 2.0, 3.0]})
        df2 = pd.DataFrame({'y': [10.0, 20.0, 30.0]})
        out1, out2 = align_data({'data': df1, 'order': ['c', 'a', 'b']},
                                {'data': df2, 'order': ['b', 'c', 'd']})
        # shared ids sorted: b, c
        assert out1['x'].tolist() == [3.0, 1.0]
        assert out2['y'].tolist() == [10.0, 20.0]

    def test_numpy_arrays(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        b = np.array([[10.0], [20.0], [30.0]])
        out_a, out_b = align_data({'data': a, 'order': [3, 1, 2]},
                                  {'data': b, 'order': [2, 3, 1]})
        # sorted shared ids: 1, 2, 3
        np.testing.assert_array_equal(out_a, [[3, 4], [5, 6], [1, 2]])
        np.testing.assert_array_equal(out_b, [[30], [10], [20]])
        assert isinstance(out_a, np.ndarray)

    def test_mixed_dataframe_and_numpy(self):
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0]})
        arr = np.array([[10.0], [20.0], [30.0]])
        out_df, out_arr = align_data({'data': df, 'order': [3, 2, 1]},
                                     {'data': arr, 'order': [1, 2, 3]})
        assert isinstance(out_df, pd.DataFrame)
        assert isinstance(out_arr, np.ndarray)
        assert out_df['x'].tolist() == [3.0, 2.0, 1.0]
        np.testing.assert_array_equal(out_arr, [[10.0], [20.0], [30.0]])

    def test_1d_array_becomes_2d_column(self):
        out, = align_data({'data': np.array([1.0, 2.0, 3.0]), 'order': [1, 2, 3]})
        assert out.shape == (3, 1)

    def test_drops_rows_with_nan_in_any_dataset(self):
        df = pd.DataFrame({'x': [1.0, np.nan, 3.0, 4.0]})
        arr = np.array([[1.0], [2.0], [np.nan], [4.0]])
        out_df, out_arr = align_data({'data': df, 'order': [1, 2, 3, 4]},
                                     {'data': arr, 'order': [1, 2, 3, 4]})
        assert out_df['x'].tolist() == [1.0, 4.0]
        np.testing.assert_array_equal(out_arr, [[1.0], [4.0]])

    def test_dataframe_index_is_reset(self):
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0]}, index=[7, 8, 9])
        out, = align_data({'data': df, 'order': ['a', 'b', 'c']})
        assert out.index.tolist() == [0, 1, 2]

    def test_does_not_mutate_input_dataframe(self):
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0]}, index=[7, 8, 9])
        original = df.copy()
        align_data({'data': df, 'order': ['c', 'b', 'a']},
                   {'data': pd.DataFrame({'y': [1.0, 2.0]}), 'order': ['a', 'b']})
        pd.testing.assert_frame_equal(df, original)

    def test_does_not_mutate_input_array(self):
        arr = np.array([[1.0], [np.nan], [3.0]])
        original = arr.copy()
        align_data({'data': arr, 'order': [1, 2, 3]})
        np.testing.assert_array_equal(arr, original)

    def test_no_shared_ids_gives_empty_outputs(self):
        out1, out2 = align_data({'data': pd.DataFrame({'x': [1.0, 2.0]}), 'order': ['a', 'b']},
                                {'data': np.array([[1.0], [2.0]]), 'order': ['c', 'd']})
        assert out1.shape[0] == 0
        assert out2.shape[0] == 0

    def test_three_inputs(self):
        outs = align_data(
            {'data': np.array([[1.0], [2.0], [3.0]]), 'order': [1, 2, 3]},
            {'data': np.array([[1.0], [2.0]]), 'order': [2, 3]},
            {'data': np.array([[1.0], [2.0]]), 'order': [3, 4]},
        )
        assert len(outs) == 3
        assert all(o.shape[0] == 1 for o in outs)  # only id 3 is shared
        np.testing.assert_array_equal(outs[0], [[3.0]])

    def test_missing_order_raises(self):
        with pytest.raises(ValueError, match="Missing 'order'"):
            align_data({'data': np.zeros((3, 1))})

    def test_order_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length of 'order'"):
            align_data({'data': np.zeros((3, 1)), 'order': [1, 2]})

    def test_3d_array_raises(self):
        with pytest.raises(ValueError, match="dimensions"):
            align_data({'data': np.zeros((2, 2, 2)), 'order': [1, 2]})

    def test_unsupported_type_raises(self):
        # a list has no .shape, so use an object that does but is neither ndarray nor DataFrame
        class Fake:
            shape = (2, 1)
        with pytest.raises(TypeError, match="Unsupported data type"):
            align_data({'data': Fake(), 'order': [1, 2]})
