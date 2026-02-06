"""
Download TCGA LUAD and LUSC gene expression data from UCSC Xena.

Data sources:
- LUAD: TCGA Lung Adenocarcinoma
- LUSC: TCGA Lung Squamous Cell Carcinoma

Following methodology from PMC7909418 (breast cancer classification),
applied to lung cancer subtypes.
"""

import os
import gzip
import requests
import pandas as pd
from io import BytesIO

# TCGA data URLs from UCSC Xena
DATA_URLS = {
    'LUAD': 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.LUAD.sampleMap%2FHiSeqV2.gz',
    'LUSC': 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.LUSC.sampleMap%2FHiSeqV2.gz'
}

def download_and_load(url: str, cancer_type: str) -> pd.DataFrame:
    """Download gzipped data and load into DataFrame."""
    print(f"Downloading {cancer_type} data...")

    response = requests.get(url, timeout=300)
    response.raise_for_status()

    # Decompress and read
    with gzip.open(BytesIO(response.content), 'rt') as f:
        df = pd.read_csv(f, sep='\t', index_col=0)

    print(f"  Downloaded: {df.shape[1]} samples, {df.shape[0]} genes")
    return df


def filter_tumor_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to keep only primary tumor samples.
    TCGA barcodes: samples ending with -01 are primary tumors.
    """
    tumor_cols = [col for col in df.columns if '-01' in col]
    filtered = df[tumor_cols]
    print(f"  After tumor filtering: {filtered.shape[1]} samples")
    return filtered


def combine_datasets(luad_df: pd.DataFrame, lusc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine LUAD and LUSC datasets.
    - Transpose so samples are rows, genes are columns
    - Add label column
    - Keep only common genes
    """
    # Find common genes
    common_genes = luad_df.index.intersection(lusc_df.index)
    print(f"Common genes between datasets: {len(common_genes)}")

    # Filter to common genes
    luad_common = luad_df.loc[common_genes]
    lusc_common = lusc_df.loc[common_genes]

    # Transpose (samples as rows)
    luad_t = luad_common.T
    lusc_t = lusc_common.T

    # Add labels
    luad_t['label'] = 'LUAD'
    lusc_t['label'] = 'LUSC'

    # Combine
    combined = pd.concat([luad_t, lusc_t], axis=0)
    combined = combined.reset_index().rename(columns={'index': 'sample_id'})

    print(f"Combined dataset: {combined.shape[0]} samples, {combined.shape[1]-2} genes")
    print(f"  LUAD samples: {(combined['label'] == 'LUAD').sum()}")
    print(f"  LUSC samples: {(combined['label'] == 'LUSC').sum()}")

    return combined


def main():
    """Main function to download and prepare data."""
    # Create data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Download both datasets
    luad_df = download_and_load(DATA_URLS['LUAD'], 'LUAD')
    lusc_df = download_and_load(DATA_URLS['LUSC'], 'LUSC')

    # Filter to tumor samples only
    luad_tumor = filter_tumor_samples(luad_df)
    lusc_tumor = filter_tumor_samples(lusc_df)

    # Combine datasets
    combined = combine_datasets(luad_tumor, lusc_tumor)

    # Save to CSV
    output_path = os.path.join(data_dir, 'tcga_lung_expression.csv')
    combined.to_csv(output_path, index=False)
    print(f"\nData saved to: {output_path}")

    return combined


if __name__ == '__main__':
    main()
