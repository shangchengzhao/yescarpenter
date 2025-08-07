# yescarpenter/__init__.py

from .pca import perform_pca, scree_plot, create_scree_plot, pc_plot
from .RSA import (
    construct_RDM, draw_heatmap, 
    do_RSA, permutation_histogram, maximal_permutation_test, align_data,
    clean_data_df, clean_data_np, get_triangular_matrix, 
    calculate_r_squared_loss, standardize_rdms, shuffle_rdm, 
    mantel_permutation, variance_partitioning
)
from .ttest import paired_ttest, onesample_ttest

__all__ = [
    # PCA functions
    'perform_pca', 'scree_plot', 'create_scree_plot', 'pc_plot',
    
    # RSA core functions
    'construct_RDM', 'draw_heatmap', 'convert_RDM_to_vector', 
    'do_RSA', 'permutation_histogram', 'maximal_permutation_test', 
    'align_data',
    
    # RSA utility functions
    'clean_data_df', 'clean_data_np', 'get_triangular_matrix', 
    'calculate_r_squared_loss', 'standardize_rdms', 'shuffle_rdm', 
    'mantel_permutation',
    
    # Variance partitioning
    'variance_partitioning',

    # T-test functions
    'paired_ttest', 'onesample_ttest'
]
