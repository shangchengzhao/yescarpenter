def onesample_ttest(var1, var2, group1_name="Group 1", group2_name="Group 2"):
    """
    Perform one-sample t-test analysis on two groups.

    Parameters:
    var1: array-like, first group data
    var2: array-like, second group data
    group1_name: str, name for first group (default: "Group 1")
    group2_name: str, name for second group (default: "Group 2")
    """
    import pandas as pd
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    from scipy.stats import probplot
    
    # Convert to pandas Series if needed
    if not isinstance(var1, pd.Series):
        var1 = pd.Series(var1)
    if not isinstance(var2, pd.Series):
        var2 = pd.Series(var2)
    
    print(f"\n{group1_name} size: {len(var1)}")
    print(f"{group2_name} size: {len(var2)}")

    # Check for normality using Shapiro-Wilk test
    print("\n" + "="*50)
    print("NORMALITY TESTS")
    print("="*50)

    shapiro_group1 = stats.shapiro(var1)
    print(f"{group1_name} - Shapiro-Wilk test:")
    print(f"  Statistic: {shapiro_group1.statistic:.4f}")
    print(f"  p-value: {shapiro_group1.pvalue:.4f}")

    shapiro_group2 = stats.shapiro(var2)
    print(f"{group2_name} - Shapiro-Wilk test:")
    print(f"  Statistic: {shapiro_group2.statistic:.4f}")
    print(f"  p-value: {shapiro_group2.pvalue:.4f}")

    # Test for equal variances (Levene's test)
    print("\nLevene's test for equal variances:")
    levene_test = stats.levene(var1, var2)
    print(f"  Statistic: {levene_test.statistic:.4f}")
    print(f"  p-value: {levene_test.pvalue:.4f}")

    # Conduct independent samples t-test
    print("\n" + "="*50)
    print("INDEPENDENT SAMPLES T-TEST RESULTS")
    print("="*50)

    # Default assumes equal variances
    ttest_equal_var = stats.ttest_ind(var1, var2, equal_var=True)
    print(f"Equal variances assumed:")
    print(f"  Statistic: {ttest_equal_var.statistic:.4f}")
    print(f"  p-value: {ttest_equal_var.pvalue:.4f}")

    # Welch's t-test (unequal variances)
    ttest_unequal_var = stats.ttest_ind(var1, var2, equal_var=False)
    print(f"\nWelch's t-test (unequal variances):")
    print(f"  Statistic: {ttest_unequal_var.statistic:.4f}")
    print(f"  p-value: {ttest_unequal_var.pvalue:.4f}")

    # Calculate descriptive statistics
    print("\n" + "="*50)
    print("DESCRIPTIVE STATISTICS")
    print("="*50)

    print(f"{group1_name}:")
    print(f"  Mean: {var1.mean():.4f}")
    print(f"  SD: {var1.std(ddof=1):.4f}")
    print(f"  Median: {var1.median():.4f}")
    print(f"  Min: {var1.min():.4f}")
    print(f"  Max: {var1.max():.4f}")

    print(f"\n{group2_name}:")
    print(f"  Mean: {var2.mean():.4f}")
    print(f"  SD: {var2.std(ddof=1):.4f}")
    print(f"  Median: {var2.median():.4f}")
    print(f"  Min: {var2.min():.4f}")
    print(f"  Max: {var2.max():.4f}")

    print(f"\nDifference in means: {var1.mean() - var2.mean():.4f}")

    # Calculate effect size (Cohen's d)
    print("\n" + "="*50)
    print("EFFECT SIZE")
    print("="*50)

    # Cohen's d for independent t-test
    pooled_sd = np.sqrt(((len(var1) - 1) * var1.var(ddof=1) + 
                         (len(var2) - 1) * var2.var(ddof=1)) / 
                        (len(var1) + len(var2) - 2))
    cohens_d = (var1.mean() - var2.mean()) / pooled_sd
    print(f"Cohen's d: {cohens_d:.4f}")

    # Interpretation of Cohen's d
    def interpret_cohens_d(d):
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    print(f"Effect size interpretation: {interpret_cohens_d(cohens_d)}")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Box plot
    groups_data = [var1, var2]
    axes[0, 0].boxplot(groups_data, labels=[group1_name, group2_name])
    axes[0, 0].set_title('Box Plot Comparison')
    axes[0, 0].set_ylabel('Values')

    # Histogram
    axes[0, 1].hist(var1, alpha=0.7, label=group1_name, bins=8, color='blue')
    axes[0, 1].hist(var2, alpha=0.7, label=group2_name, bins=8, color='red')
    axes[0, 1].set_title('Distribution Comparison')
    axes[0, 1].set_xlabel('Values')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()

    # Strip plot with means
    axes[1, 0].scatter(np.ones(len(var1)), var1, alpha=0.7, label=group1_name, color='blue')
    axes[1, 0].scatter(np.ones(len(var2)) * 2, var2, alpha=0.7, label=group2_name, color='red')
    axes[1, 0].hlines(var1.mean(), 0.8, 1.2, colors='blue', linestyles='solid', linewidth=3, label=f'{group1_name} Mean')
    axes[1, 0].hlines(var2.mean(), 1.8, 2.2, colors='red', linestyles='solid', linewidth=3, label=f'{group2_name} Mean')
    axes[1, 0].set_xlim(0.5, 2.5)
    axes[1, 0].set_xticks([1, 2])
    axes[1, 0].set_xticklabels([group1_name, group2_name])
    axes[1, 0].set_ylabel('Values')
    axes[1, 0].set_title('Individual Values with Means')
    axes[1, 0].legend()

    # Q-Q plots for normality check
    probplot(var1, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title(f'Q-Q Plot: {group1_name}')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    # Create separate Q-Q plot for group 2
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    probplot(var2, dist="norm", plot=ax)
    ax.set_title(f'Q-Q Plot: {group2_name}')
    ax.grid(True)
    plt.show()

    # Summary and recommendations
    print("\n" + "="*50)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*50)

    # Check assumptions
    print("Assumption checks:")
    if shapiro_group1.pvalue > 0.05 and shapiro_group2.pvalue > 0.05:
        print("✓ Both groups appear normally distributed (p > 0.05)")
        normality_ok = True
    else:
        print("⚠ One or both groups may not be normally distributed (p ≤ 0.05)")
        normality_ok = False

    if levene_test.pvalue > 0.05:
        print("✓ Equal variances assumption met (p > 0.05)")
        equal_var_ok = True
    else:
        print("⚠ Equal variances assumption violated (p ≤ 0.05)")
        equal_var_ok = False

    # Recommend which test result to use
    print(f"\nRecommended test result:")
    if equal_var_ok:
        recommended_pvalue = ttest_equal_var.pvalue
        print(f"Use standard t-test result: p = {recommended_pvalue:.4f}")
    else:
        recommended_pvalue = ttest_unequal_var.pvalue
        print(f"Use Welch's t-test result: p = {recommended_pvalue:.4f}")

    if not normality_ok:
        print("Consider using Mann-Whitney U test due to normality violations.")

    # Final conclusion
    print(f"\nConclusion:")
    if recommended_pvalue < 0.05:
        print(f"✓ Significant difference found between groups (p = {recommended_pvalue:.4f})")
        if var1.mean() > var2.mean():
            print(f"  {group1_name} has significantly higher values than {group2_name}")
        else:
            print(f"  {group2_name} has significantly higher values than {group1_name}")
    else:
        print(f"✗ No significant difference found between groups (p = {recommended_pvalue:.4f})")

    print(f"Effect size: {interpret_cohens_d(cohens_d)} (Cohen's d = {cohens_d:.4f})")
    
    # Return results as dictionary
    return {
        'shapiro_group1': shapiro_group1,
        'shapiro_group2': shapiro_group2,
        'levene_test': levene_test,
        'ttest_equal_var': ttest_equal_var,
        'ttest_unequal_var': ttest_unequal_var,
        'cohens_d': cohens_d,
        'recommended_pvalue': recommended_pvalue,
        'descriptive_stats': {
            f'{group1_name}_mean': var1.mean(),
            f'{group1_name}_sd': var1.std(ddof=1),
            f'{group2_name}_mean': var2.mean(),
            f'{group2_name}_sd': var2.std(ddof=1)
        }
    }

def paired_ttest(pre, post, pre_name="Pre", post_name="Post"):
    """
    Perform paired samples t-test analysis on two related groups.
    
    Parameters:
    pre: array-like, pre-treatment/baseline data
    post: array-like, post-treatment/follow-up data
    pre_name: str, name for pre group (default: "Pre")
    post_name: str, name for post group (default: "Post")
    """
    import pandas as pd
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Convert to pandas Series if needed
    if not isinstance(pre, pd.Series):
        pre = pd.Series(pre)
    if not isinstance(post, pd.Series):
        post = pd.Series(post)
    
    print(f"Data summary:")
    combined_data = pd.DataFrame({pre_name: pre, post_name: post})
    print(combined_data.describe())
    print(f"\nFirst few rows:")
    print(combined_data.head())
    print(f"\nData shape: {combined_data.shape}")
    
    print(f"\n{pre_name} size: {len(pre)}")
    print(f"{post_name} size: {len(post)}")

    # Check for normality using Shapiro-Wilk test on difference scores
    print("\n" + "="*50)
    print("NORMALITY TESTS")
    print("="*50)

    diff_scores = pre - post
    shapiro_diff = stats.shapiro(diff_scores)
    print(f"Difference ({pre_name} - {post_name}) - Shapiro-Wilk test:")
    print(f"  Statistic: {shapiro_diff.statistic:.4f}")
    print(f"  p-value: {shapiro_diff.pvalue:.4f}")

    # Conduct paired t-test
    print("\n" + "="*50)
    print("PAIRED T-TEST RESULTS")
    print("="*50)

    paired_ttest = stats.ttest_rel(pre, post)
    print(f"Statistic: {paired_ttest.statistic:.4f}")
    print(f"p-value: {paired_ttest.pvalue:.4f}")

    # Calculate descriptive statistics
    print("\n" + "="*50)
    print("DESCRIPTIVE STATISTICS")
    print("="*50)

    print(f"{pre_name}:")
    print(f"  Mean: {pre.mean():.4f}")
    print(f"  SD: {pre.std(ddof=1):.4f}")
    print(f"  Median: {pre.median():.4f}")
    print(f"  Min: {pre.min():.4f}")
    print(f"  Max: {pre.max():.4f}")

    print(f"\n{post_name}:")
    print(f"  Mean: {post.mean():.4f}")
    print(f"  SD: {post.std(ddof=1):.4f}")
    print(f"  Median: {post.median():.4f}")
    print(f"  Min: {post.min():.4f}")
    print(f"  Max: {post.max():.4f}")

    print(f"\nDifference in means ({pre_name} - {post_name}): {pre.mean() - post.mean():.4f}")

    # Calculate effect size (Cohen's d)
    print("\n" + "="*50)
    print("EFFECT SIZES")
    print("="*50)

    # Cohen's d for paired t-test
    cohens_d_paired = diff_scores.mean() / diff_scores.std(ddof=1)
    print(f"Cohen's d (paired): {cohens_d_paired:.4f}")

    # Interpretation of Cohen's d
    def interpret_cohens_d(d):
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    print(f"Effect size interpretation (paired): {interpret_cohens_d(cohens_d_paired)}")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Box plot
    axes[0, 0].boxplot([pre, post], labels=[pre_name, post_name])
    axes[0, 0].set_title('Box Plot Comparison')
    axes[0, 0].set_ylabel('Values')

    # Histogram
    axes[0, 1].hist(pre, alpha=0.7, label=pre_name, bins=10, color='blue')
    axes[0, 1].hist(post, alpha=0.7, label=post_name, bins=10, color='red')
    axes[0, 1].set_title('Distribution Comparison')
    axes[0, 1].set_xlabel('Values')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()

    # Scatter plot (paired data)
    axes[1, 0].scatter(pre, post, alpha=0.7)
    axes[1, 0].plot([min(pre.min(), post.min()), 
                    max(pre.max(), post.max())], 
                   [min(pre.min(), post.min()), 
                    max(pre.max(), post.max())], 
                   'r--', alpha=0.8, label='Line of equality')
    axes[1, 0].set_xlabel(pre_name)
    axes[1, 0].set_ylabel(post_name)
    axes[1, 0].set_title(f'{pre_name} vs {post_name} Scatter Plot')
    axes[1, 0].legend()

    # Difference plot (for paired data)
    axes[1, 1].hist(diff_scores, bins=10, alpha=0.7, color='green')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.8, label='No difference')
    axes[1, 1].axvline(x=diff_scores.mean(), color='blue', linestyle='-', alpha=0.8, label='Mean difference')
    axes[1, 1].set_title(f'Difference Scores ({pre_name} - {post_name})')
    axes[1, 1].set_xlabel('Difference')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    # Summary recommendation
    print("\n" + "="*50)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*50)

    if shapiro_diff.pvalue > 0.05:
        print("✓ Difference scores appear normal (p > 0.05), t-test assumptions met.")
    else:
        print("⚠ Difference scores may not be normal (p ≤ 0.05).")
        print("  Consider using non-parametric tests (Wilcoxon signed-rank test).")

    print(f"\nBased on your data:")
    if paired_ttest.pvalue < 0.05:
        print(f"✓ Significant difference found (paired t-test p = {paired_ttest.pvalue:.4f})")
        if pre.mean() > post.mean():
            print(f"  {pre_name} values are significantly higher than {post_name} values")
        else:
            print(f"  {post_name} values are significantly higher than {pre_name} values")
    else:
        print(f"✗ No significant difference found (paired t-test p = {paired_ttest.pvalue:.4f})")

    print(f"Effect size: {interpret_cohens_d(cohens_d_paired)} (Cohen's d = {cohens_d_paired:.4f})")
    
    # Return results as dictionary
    return {
        'shapiro_diff': shapiro_diff,
        'paired_ttest': paired_ttest,
        'cohens_d': cohens_d_paired,
        'diff_scores': diff_scores,
        'descriptive_stats': {
            f'{pre_name}_mean': pre.mean(),
            f'{pre_name}_sd': pre.std(ddof=1),
            f'{post_name}_mean': post.mean(),
            f'{post_name}_sd': post.std(ddof=1),
            'mean_difference': pre.mean() - post.mean()
        }
    }



# previous code
# import numpy as np
# from scipy import stats
# import pandas as pd

# def onesample_ttest(sample, popmean):
#     '''
#     Perform a one-sample t-test on a sample.
#     Results:
#     - Degrees of freedom
#     - t-statistic
#     - p-value
#     - mean
#     - 95% confidence interval for the mean
#     - Cohen's d (standardized mean difference)
#     '''
#     # Calculate the differences
#     n = len(sample)
#     df = n - 1  # degrees of freedom

#     # Perform paired t-test
#     t_stat, p_value = stats.ttest_1samp(sample, popmean)

#     # Calculate mean and standard deviation of the differences
#     mean_diff = np.mean(sample) - popmean
#     std_diff = np.std(sample, ddof=1)  # sample standard deviation
#     se_diff = std_diff / np.sqrt(n)  # standard error

#     # Calculate 95% confidence interval for the mean difference
#     t_crit = stats.t.ppf(1 - 0.025, df)  # two-tailed, so use 97.5th percentile
#     ci_lower = mean_diff - t_crit * se_diff
#     ci_upper = mean_diff + t_crit * se_diff

#     # Calculate Cohen's d for paired samples (standardized mean difference)
#     cohens_d = mean_diff / std_diff

#     # Output results
#     print("Degrees of freedom:", df)
#     print("Paired t-test: t-statistic = {:.4f}, p-value = {:.4f}".format(t_stat, p_value))
#     print("Mean difference: {:.4f}".format(mean_diff))
#     print("95% Confidence Interval for the mean difference: ({:.4f}, {:.4f})".format(ci_lower, ci_upper))
#     print("Cohen's d:", cohens_d)

#     result = pd.DataFrame(
#         columns=['Degrees of freedom', 't-statistic', 'p-value', 'Mean difference', '95% CI', 'Cohen\'s d'])
#     result.loc[0] = [df, t_stat, p_value, mean_diff, (ci_lower, ci_upper), cohens_d]
#     return result


# def paired_ttest(before, after):
#     '''
#     Perform a paired t-test on two sets of measurements.
#     Results:
#     - Degrees of freedom
#     - t-statistic
#     - p-value
#     - mean difference
#     - 95% confidence interval for the mean difference
#     - Cohen's d (standardized mean difference)
#     '''
    
#     # Calculate the differences
#     diff = after - before
#     n = len(diff)
#     df = n - 1  # degrees of freedom

#     # Perform paired t-test
#     t_stat, p_value = stats.ttest_rel(before, after)

#     # Calculate mean and standard deviation of the differences
#     mean_diff = np.mean(diff)
#     std_diff = np.std(diff, ddof=1)  # sample standard deviation
#     se_diff = std_diff / np.sqrt(n)  # standard error

#     # Calculate 95% confidence interval for the mean difference
#     t_crit = stats.t.ppf(1 - 0.025, df)  # two-tailed, so use 97.5th percentile
#     ci_lower = mean_diff - t_crit * se_diff
#     ci_upper = mean_diff + t_crit * se_diff

#     # Calculate Cohen's d for paired samples (standardized mean difference)
#     cohens_d = mean_diff / std_diff

#     # Output results
#     print("Degrees of freedom:", df)
#     print("Paired t-test: t-statistic = {:.4f}, p-value = {:.4f}".format(t_stat, p_value))
#     print("Mean difference: {:.4f}".format(mean_diff))
#     print("95% Confidence Interval for the mean difference: ({:.4f}, {:.4f})".format(ci_lower, ci_upper))
#     print("Cohen's d:", cohens_d)

#     result = pd.DataFrame(
#         columns=['Degrees of freedom', 't-statistic', 'p-value', 'Mean difference', '95% CI', 'Cohen\'s d'])
#     result.loc[0] = [df, t_stat, p_value, mean_diff, (ci_lower, ci_upper), cohens_d]
#     return result
