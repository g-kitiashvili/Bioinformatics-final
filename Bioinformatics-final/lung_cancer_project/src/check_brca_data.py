"""Quick script to inspect BRCA clinical data structure."""

import requests
import pandas as pd
from io import BytesIO

print("Downloading BRCA clinical data...")
url = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FBRCA_clinicalMatrix"
response = requests.get(url, timeout=120)
clinical = pd.read_csv(BytesIO(response.content), sep='\t', index_col=0)

print(f"Shape: {clinical.shape}")
print(f"\nAll columns ({len(clinical.columns)}):")
for col in sorted(clinical.columns):
    print(f"  {col}")

# Check receptor status columns specifically
print("\n" + "="*50)
print("Receptor status columns and their values:")
print("="*50)

receptor_cols = [
    'breast_carcinoma_estrogen_receptor_status',
    'breast_carcinoma_progesterone_receptor_status',
    'lab_proc_her2_neu_immunohistochemistry_receptor_status',
    'ER_Status_nature2012',
    'PR_Status_nature2012',
    'HER2_Final_Status_nature2012'
]

for col in receptor_cols:
    if col in clinical.columns:
        print(f"\n{col}:")
        print(clinical[col].value_counts(dropna=False))
