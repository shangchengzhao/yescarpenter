import numpy as np
from scipy import stats
import pandas as pd

def onesample_ttest(sample, popmean):
    '''
    Perform a one-sample t-test on a sample.
    Results:
    - Degrees of freedom
    - t-statistic
    - p-value
    - mean
    - 95% confidence interval for the mean
    - Cohen's d (standardized mean difference)
    '''
    # Calculate the differences
    n = len(sample)
    df = n - 1  # degrees of freedom

    # Perform paired t-test
    t_stat, p_value = stats.ttest_1samp(sample, popmean)

    # Calculate mean and standard deviation of the differences
    mean_diff = np.mean(sample) - popmean
    std_diff = np.std(sample, ddof=1)  # sample standard deviation
    se_diff = std_diff / np.sqrt(n)  # standard error

    # Calculate 95% confidence interval for the mean difference
    t_crit = stats.t.ppf(1 - 0.025, df)  # two-tailed, so use 97.5th percentile
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    # Calculate Cohen's d for paired samples (standardized mean difference)
    cohens_d = mean_diff / std_diff

    # Output results
    print("Degrees of freedom:", df)
    print("Paired t-test: t-statistic = {:.4f}, p-value = {:.4f}".format(t_stat, p_value))
    print("Mean difference: {:.4f}".format(mean_diff))
    print("95% Confidence Interval for the mean difference: ({:.4f}, {:.4f})".format(ci_lower, ci_upper))
    print("Cohen's d:", cohens_d)

    result = pd.DataFrame(
        columns=['Degrees of freedom', 't-statistic', 'p-value', 'Mean difference', '95% CI', 'Cohen\'s d'])
    result.loc[0] = [df, t_stat, p_value, mean_diff, (ci_lower, ci_upper), cohens_d]
    return result


def paired_ttest(before, after):
    '''
    Perform a paired t-test on two sets of measurements.
    Results:
    - Degrees of freedom
    - t-statistic
    - p-value
    - mean difference
    - 95% confidence interval for the mean difference
    - Cohen's d (standardized mean difference)
    '''
    
    # Calculate the differences
    diff = after - before
    n = len(diff)
    df = n - 1  # degrees of freedom

    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(before, after)

    # Calculate mean and standard deviation of the differences
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)  # sample standard deviation
    se_diff = std_diff / np.sqrt(n)  # standard error

    # Calculate 95% confidence interval for the mean difference
    t_crit = stats.t.ppf(1 - 0.025, df)  # two-tailed, so use 97.5th percentile
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    # Calculate Cohen's d for paired samples (standardized mean difference)
    cohens_d = mean_diff / std_diff

    # Output results
    print("Degrees of freedom:", df)
    print("Paired t-test: t-statistic = {:.4f}, p-value = {:.4f}".format(t_stat, p_value))
    print("Mean difference: {:.4f}".format(mean_diff))
    print("95% Confidence Interval for the mean difference: ({:.4f}, {:.4f})".format(ci_lower, ci_upper))
    print("Cohen's d:", cohens_d)

    result = pd.DataFrame(
        columns=['Degrees of freedom', 't-statistic', 'p-value', 'Mean difference', '95% CI', 'Cohen\'s d'])
    result.loc[0] = [df, t_stat, p_value, mean_diff, (ci_lower, ci_upper), cohens_d]
    return result
