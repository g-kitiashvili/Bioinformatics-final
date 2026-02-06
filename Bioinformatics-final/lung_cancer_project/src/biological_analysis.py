"""
Biological Analysis Module

Performs:
1. Gene Ontology (GO) Enrichment Analysis
2. KEGG Pathway Analysis
3. Visualization of biological findings

This adds biological depth to the project by interpreting
which biological processes/pathways distinguish LUAD from LUSC.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import gseapy as gp
    GSEAPY_AVAILABLE = True
except ImportError:
    GSEAPY_AVAILABLE = False
    print("Warning: gseapy not installed. Run: pip install gseapy")

def load_deg_results(results_dir):
    """Load differential expression analysis results."""
    deg_path = os.path.join(results_dir, 'deg_analysis.csv')
    if not os.path.exists(deg_path):
        raise FileNotFoundError(f"DEG results not found at {deg_path}")

    deg_df = pd.read_csv(deg_path)
    print(f"Loaded DEG results: {len(deg_df)} genes")
    return deg_df


def get_significant_genes(deg_df, n_top=500, direction='both'):
    """
    Get significant differentially expressed genes.

    Args:
        deg_df: DataFrame with DEG results
        n_top: Number of top genes to select
        direction: 'up' (LUAD > LUSC), 'down' (LUAD < LUSC), or 'both'

    Returns:
        List of gene symbols
    """
    # Sort by significance
    sorted_df = deg_df.sort_values('padj')

    if direction == 'up':
        filtered = sorted_df[sorted_df['log2FoldChange'] > 0]
    elif direction == 'down':
        filtered = sorted_df[sorted_df['log2FoldChange'] < 0]
    else:
        filtered = sorted_df

    genes = filtered.head(n_top)['gene'].tolist()
    print(f"Selected {len(genes)} {direction}-regulated genes")
    return genes


def run_go_enrichment(gene_list, organism='human', gene_sets=['GO_Biological_Process_2021']):
    """
    Perform Gene Ontology enrichment analysis.

    Args:
        gene_list: List of gene symbols
        organism: Species (default: human)
        gene_sets: GO databases to use

    Returns:
        DataFrame with enrichment results
    """
    if not GSEAPY_AVAILABLE:
        print("gseapy not available. Skipping GO enrichment.")
        return None

    print(f"\nRunning GO Enrichment Analysis on {len(gene_list)} genes...")

    try:
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gene_sets,
            organism=organism,
            outdir=None,
            cutoff=0.05
        )

        results = enr.results
        results = results[results['Adjusted P-value'] < 0.05]
        print(f"Found {len(results)} significant GO terms")
        return results

    except Exception as e:
        print(f"GO enrichment failed: {e}")
        return None


def run_kegg_pathway_analysis(gene_list, organism='human'):
    """
    Perform KEGG pathway enrichment analysis.

    Args:
        gene_list: List of gene symbols
        organism: Species

    Returns:
        DataFrame with pathway enrichment results
    """
    if not GSEAPY_AVAILABLE:
        print("gseapy not available. Skipping KEGG analysis.")
        return None

    print(f"\nRunning KEGG Pathway Analysis on {len(gene_list)} genes...")

    try:
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=['KEGG_2021_Human'],
            organism=organism,
            outdir=None,
            cutoff=0.05
        )

        results = enr.results
        results = results[results['Adjusted P-value'] < 0.05]
        print(f"Found {len(results)} significant KEGG pathways")
        return results

    except Exception as e:
        print(f"KEGG analysis failed: {e}")
        return None


def run_comprehensive_enrichment(gene_list, organism='human'):
    """
    Run enrichment against multiple databases.
    """
    if not GSEAPY_AVAILABLE:
        print("gseapy not available. Skipping enrichment.")
        return None

    databases = [
        'GO_Biological_Process_2021',
        'GO_Molecular_Function_2021',
        'GO_Cellular_Component_2021',
        'KEGG_2021_Human',
        'Reactome_2022',
        'WikiPathway_2023_Human'
    ]

    print(f"\nRunning comprehensive enrichment analysis...")

    try:
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=databases,
            organism=organism,
            outdir=None,
            cutoff=0.05
        )

        results = enr.results
        results = results[results['Adjusted P-value'] < 0.05]
        print(f"Found {len(results)} significant terms across all databases")
        return results

    except Exception as e:
        print(f"Comprehensive enrichment failed: {e}")
        return None


def plot_enrichment_results(results, title, output_path, top_n=15):
    """
    Create bar plot of enrichment results.
    """
    if results is None or len(results) == 0:
        print(f"No results to plot for {title}")
        return

    # Get top terms
    plot_data = results.head(top_n).copy()

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color by -log10(p-value)
    colors = plt.cm.Reds(plot_data['Adjusted P-value'].rank(pct=True))

    bars = ax.barh(
        range(len(plot_data)),
        -np.log10(plot_data['Adjusted P-value']),
        color=colors
    )

    # Labels
    ax.set_yticks(range(len(plot_data)))

    # Truncate long term names
    terms = plot_data['Term'].tolist()
    terms = [t[:50] + '...' if len(t) > 50 else t for t in terms]
    ax.set_yticklabels(terms)

    ax.set_xlabel('-log10(Adjusted P-value)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.invert_yaxis()

    # Add gene count annotations
    for i, (_, row) in enumerate(plot_data.iterrows()):
        gene_count = row.get('Overlap', '').split('/')[0] if 'Overlap' in row else ''
        ax.text(
            0.1, i, f"n={gene_count}",
            va='center', fontsize=9, color='white', fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_volcano(deg_df, output_path, fc_threshold=1, pval_threshold=0.05):
    """
    Create volcano plot showing differential expression.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate -log10(p-value)
    deg_df = deg_df.copy()
    deg_df['-log10_padj'] = -np.log10(deg_df['padj'].clip(lower=1e-300))

    # Categorize genes
    deg_df['category'] = 'Not Significant'
    deg_df.loc[
        (deg_df['padj'] < pval_threshold) & (deg_df['log2FoldChange'] > fc_threshold),
        'category'
    ] = 'Up in LUAD'
    deg_df.loc[
        (deg_df['padj'] < pval_threshold) & (deg_df['log2FoldChange'] < -fc_threshold),
        'category'
    ] = 'Up in LUSC'

    # Colors
    colors = {
        'Not Significant': '#CCCCCC',
        'Up in LUAD': '#E74C3C',
        'Up in LUSC': '#3498DB'
    }

    # Plot
    for category in ['Not Significant', 'Up in LUAD', 'Up in LUSC']:
        subset = deg_df[deg_df['category'] == category]
        ax.scatter(
            subset['log2FoldChange'],
            subset['-log10_padj'],
            c=colors[category],
            label=f"{category} (n={len(subset)})",
            alpha=0.6,
            s=20
        )

    # Add threshold lines
    ax.axhline(-np.log10(pval_threshold), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(fc_threshold, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(-fc_threshold, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('log2(Fold Change)', fontsize=12)
    ax.set_ylabel('-log10(Adjusted P-value)', fontsize=12)
    ax.set_title('Volcano Plot: LUAD vs LUSC Differential Expression', fontsize=14)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved volcano plot: {output_path}")


def create_enrichment_summary(go_results, kegg_results, output_path):
    """
    Create summary table of top enrichment results.
    """
    summary_data = []

    if go_results is not None and len(go_results) > 0:
        for _, row in go_results.head(10).iterrows():
            summary_data.append({
                'Database': 'GO',
                'Term': row['Term'],
                'P-value': row['Adjusted P-value'],
                'Genes': row.get('Overlap', 'N/A')
            })

    if kegg_results is not None and len(kegg_results) > 0:
        for _, row in kegg_results.head(10).iterrows():
            summary_data.append({
                'Database': 'KEGG',
                'Term': row['Term'],
                'P-value': row['Adjusted P-value'],
                'Genes': row.get('Overlap', 'N/A')
            })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path, index=False)
        print(f"Saved enrichment summary: {output_path}")
        return summary_df

    return None


def main():
    """Run biological analysis pipeline."""
    print("="*70)
    print("BIOLOGICAL ANALYSIS PIPELINE")
    print("="*70)

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(script_dir, '..')
    results_dir = os.path.join(project_dir, 'results')
    figures_dir = os.path.join(project_dir, 'figures')

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Load DEG results
    try:
        deg_df = load_deg_results(results_dir)
    except FileNotFoundError as e:
        print(e)
        print("Please run classify.py first to generate DEG results.")
        return

    # Create volcano plot
    print("\n1. Creating Volcano Plot...")
    plot_volcano(
        deg_df,
        os.path.join(figures_dir, 'volcano_plot.png')
    )

    # Get significant genes
    print("\n2. Selecting significant genes...")
    all_sig_genes = get_significant_genes(deg_df, n_top=500, direction='both')
    up_genes = get_significant_genes(deg_df, n_top=250, direction='up')
    down_genes = get_significant_genes(deg_df, n_top=250, direction='down')

    # GO Enrichment - All significant genes
    print("\n3. Running GO Enrichment (all significant genes)...")
    go_results_all = run_go_enrichment(all_sig_genes)

    if go_results_all is not None and len(go_results_all) > 0:
        go_results_all.to_csv(
            os.path.join(results_dir, 'go_enrichment_all.csv'),
            index=False
        )
        plot_enrichment_results(
            go_results_all,
            'GO Biological Process Enrichment (All DEGs)',
            os.path.join(figures_dir, 'go_enrichment_all.png')
        )

    # GO Enrichment - Up-regulated in LUAD
    print("\n4. Running GO Enrichment (up-regulated in LUAD)...")
    go_results_up = run_go_enrichment(up_genes)

    if go_results_up is not None and len(go_results_up) > 0:
        go_results_up.to_csv(
            os.path.join(results_dir, 'go_enrichment_up_luad.csv'),
            index=False
        )
        plot_enrichment_results(
            go_results_up,
            'GO Enrichment: Genes Up-regulated in LUAD',
            os.path.join(figures_dir, 'go_enrichment_up_luad.png')
        )

    # GO Enrichment - Up-regulated in LUSC
    print("\n5. Running GO Enrichment (up-regulated in LUSC)...")
    go_results_down = run_go_enrichment(down_genes)

    if go_results_down is not None and len(go_results_down) > 0:
        go_results_down.to_csv(
            os.path.join(results_dir, 'go_enrichment_up_lusc.csv'),
            index=False
        )
        plot_enrichment_results(
            go_results_down,
            'GO Enrichment: Genes Up-regulated in LUSC',
            os.path.join(figures_dir, 'go_enrichment_up_lusc.png')
        )

    # KEGG Pathway Analysis
    print("\n6. Running KEGG Pathway Analysis...")
    kegg_results = run_kegg_pathway_analysis(all_sig_genes)

    if kegg_results is not None and len(kegg_results) > 0:
        kegg_results.to_csv(
            os.path.join(results_dir, 'kegg_pathways.csv'),
            index=False
        )
        plot_enrichment_results(
            kegg_results,
            'KEGG Pathway Enrichment',
            os.path.join(figures_dir, 'kegg_pathways.png')
        )

    # Create summary
    print("\n7. Creating enrichment summary...")
    summary = create_enrichment_summary(
        go_results_all,
        kegg_results,
        os.path.join(results_dir, 'enrichment_summary.csv')
    )

    print("\n" + "="*70)
    print("BIOLOGICAL ANALYSIS COMPLETE")
    print("="*70)

    if summary is not None:
        print("\nTop Enriched Terms:")
        print(summary.to_string(index=False))

    print(f"\nResults saved to: {results_dir}")
    print(f"Figures saved to: {figures_dir}")


if __name__ == '__main__':
    main()
