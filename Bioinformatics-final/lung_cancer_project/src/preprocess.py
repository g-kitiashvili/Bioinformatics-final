"""
Preprocessing module for lung cancer classification.

Implements methodology from PMC7909418:
1. Log-CPM normalization (similar to edgeR/LIMMA approach)
2. Differential expression analysis for feature selection
3. Gene filtering based on fold change and p-value thresholds
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import Tuple, List




def calculate_fold_change(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate log2 fold change between two groups."""
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    # Since data is log2 transformed, fold change = difference of means
    return mean1 - mean2


def differential_expression_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 0.05,
    log2fc_threshold: float = 1.0
) -> pd.DataFrame:
    """
    Perform differential expression analysis between two groups.

    Similar to LIMMA analysis in the original paper:
    - Calculate log2 fold change
    - Perform statistical test (t-test)
    - Apply FDR correction
    - Filter by significance thresholds

    Args:
        X: Gene expression matrix (samples x genes)
        y: Labels (LUAD/LUSC)
        alpha: Significance threshold after FDR correction
        log2fc_threshold: Minimum absolute log2 fold change

    Returns:
        DataFrame with DEG statistics for all genes
    """
    print("Performing differential expression analysis...")

    labels = y.unique()
    group1_mask = y == labels[0]
    group2_mask = y == labels[1]

    results = []
    genes = X.columns.tolist()

    for gene in genes:
        group1_values = X.loc[group1_mask, gene].values
        group2_values = X.loc[group2_mask, gene].values

        log2fc = calculate_fold_change(group1_values, group2_values)

        t_stat, p_value = stats.ttest_ind(group1_values, group2_values)

        results.append({
            'gene': gene,
            'log2FoldChange': log2fc,
            'pvalue': p_value,
            't_statistic': t_stat,
            'mean_group1': np.mean(group1_values),
            'mean_group2': np.mean(group2_values)
        })

    deg_df = pd.DataFrame(results)

    # FDR correction (Benjamini-Hochberg)
    _, pvals_corrected, _, _ = multipletests(
        deg_df['pvalue'].fillna(1),
        alpha=alpha,
        method='fdr_bh'
    )
    deg_df['padj'] = pvals_corrected

    # Add absolute fold change
    deg_df['abs_log2FC'] = np.abs(deg_df['log2FoldChange'])

    # Mark significant genes
    deg_df['significant'] = (
        (deg_df['padj'] < alpha) &
        (deg_df['abs_log2FC'] > log2fc_threshold)
    )

    n_sig = deg_df['significant'].sum()
    print(f"  Significant DEGs (padj < {alpha}, |log2FC| > {log2fc_threshold}): {n_sig}")

    return deg_df


def select_top_genes(
    deg_df: pd.DataFrame,
    n_genes: int,
    by: str = 'padj'
) -> List[str]:
    """
    Select top N genes based on differential expression results.

    Following paper methodology: select top genes by significance,
    testing different thresholds (256, 512, 1024, 2048, 4096).

    Args:
        deg_df: DataFrame from differential_expression_analysis
        n_genes: Number of top genes to select
        by: Column to sort by ('padj' or 'abs_log2FC')

    Returns:
        List of top gene names
    """
    # Sort by significance (ascending p-value) and fold change (descending)
    sorted_df = deg_df.sort_values(
        by=['padj', 'abs_log2FC'],
        ascending=[True, False]
    )

    top_genes = sorted_df.head(n_genes)['gene'].tolist()
    return top_genes


def preprocess_for_classification(
    df: pd.DataFrame,
    n_top_genes: int = 1024,
    alpha: float = 0.05,
    log2fc_threshold: float = 1.0
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str]]:
    """
    Full preprocessing pipeline for classification.

    Args:
        df: Combined expression data with 'sample_id' and 'label' columns
        n_top_genes: Number of top DEGs to select
        alpha: Significance threshold
        log2fc_threshold: Fold change threshold

    Returns:
        X: Preprocessed feature matrix
        y: Labels
        deg_results: Full DEG analysis results
        selected_genes: List of selected gene names
    """
    print(f"\n{'='*60}")
    print("PREPROCESSING PIPELINE")
    print(f"{'='*60}")

    # Separate features and labels
    y = df['label']
    X = df.drop(columns=['sample_id', 'label'])

    print(f"Input: {X.shape[0]} samples, {X.shape[1]} genes")

    # Remove genes with zero variance
    variances = X.var()
    nonzero_var_genes = variances[variances > 0].index
    X = X[nonzero_var_genes]
    print(f"After variance filter: {X.shape[1]} genes")


    # Differential expression analysis
    deg_results = differential_expression_analysis(
        X, y, alpha=alpha, log2fc_threshold=log2fc_threshold
    )

    # Select top genes
    selected_genes = select_top_genes(deg_results, n_top_genes)
    X_selected = X[selected_genes]

    print(f"Selected top {n_top_genes} genes for classification")
    print(f"Final feature matrix: {X_selected.shape}")

    return X_selected, y, deg_results, selected_genes


if __name__ == '__main__':
    # Test preprocessing
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'tcga_lung_expression.csv')

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        X, y, deg_results, genes = preprocess_for_classification(df, n_top_genes=256)

        # Save DEG results
        results_dir = os.path.join(script_dir, '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        deg_results.to_csv(os.path.join(results_dir, 'deg_analysis.csv'), index=False)
        print(f"\nDEG results saved to results/deg_analysis.csv")
    else:
        print(f"Data file not found: {data_path}")
        print("Run download_data.py first.")
