"""
Reproduce Original Paper (PMC7909418) Results

This script applies our pipeline to the original breast cancer dataset
(TNBC vs non-TNBC) to verify how closely we can reproduce the paper's results.

Original Paper Results:
- Best Accuracy: 90% (SVM with 256 genes)
- Dataset: TCGA BRCA (934 samples)
- Classifiers: SVM, KNN, Naive Bayes, Decision Tree
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import gzip
from io import BytesIO
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

GENE_THRESHOLDS = [256, 512, 1024, 2048, 4096]
TEST_SIZE = 0.10
CV_FOLDS = 10
RANDOM_STATE = 42


def download_brca_data():
    """Download TCGA BRCA gene expression data from UCSC Xena."""
    print("Downloading TCGA BRCA data from UCSC Xena...")

    # TCGA BRCA HiSeqV2 URL
    url = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FHiSeqV2.gz"

    response = requests.get(url, timeout=300)
    response.raise_for_status()

    with gzip.open(BytesIO(response.content), 'rt') as f:
        df = pd.read_csv(f, sep='\t', index_col=0)

    # Transpose: genes as columns, samples as rows
    df = df.T
    print(f"Downloaded: {df.shape[0]} samples, {df.shape[1]} genes")

    return df


def get_brca_subtypes():
    """
    Download BRCA clinical data to identify TNBC vs non-TNBC.

    TNBC (Triple-Negative Breast Cancer) is defined as:
    - ER negative (Estrogen Receptor)
    - PR negative (Progesterone Receptor)
    - HER2 negative
    """
    print("Downloading BRCA clinical/subtype data...")

    # Try to get PAM50 subtype data which includes Basal (often TNBC)
    # TCGA BRCA has PAM50 molecular subtypes
    url = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FBRCA_clinicalMatrix"

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    clinical = pd.read_csv(BytesIO(response.content), sep='\t', index_col=0)
    print(f"Clinical data: {clinical.shape[0]} samples")

    return clinical


def classify_tnbc(clinical_df, expression_samples):
    """
    Classify samples as TNBC or non-TNBC based on receptor status.

    The paper used ER, PR, HER2 status from clinical data.
    TNBC = ER-, PR-, HER2-

    Using TCGA BRCA columns:
    - breast_carcinoma_estrogen_receptor_status: Positive/Negative
    - breast_carcinoma_progesterone_receptor_status: Positive/Negative
    - lab_proc_her2_neu_immunohistochemistry_receptor_status: Positive/Negative/Equivocal
    """
    # Filter to samples in expression data
    common_samples = list(set(expression_samples) & set(clinical_df.index))
    print(f"Samples with both expression and clinical data: {len(common_samples)}")

    # Use the most complete receptor status columns
    er_col = 'breast_carcinoma_estrogen_receptor_status'
    pr_col = 'breast_carcinoma_progesterone_receptor_status'
    her2_col = 'lab_proc_her2_neu_immunohistochemistry_receptor_status'

    print(f"\nClassifying based on receptor status:")
    print(f"  ER column: {er_col}")
    print(f"  PR column: {pr_col}")
    print(f"  HER2 column: {her2_col}")
    print(f"  TNBC = ER Negative AND PR Negative AND HER2 Negative")

    labels = {}
    tnbc_count = 0
    non_tnbc_count = 0
    skipped = 0

    for sample in common_samples:
        er = clinical_df.loc[sample, er_col] if sample in clinical_df.index else None
        pr = clinical_df.loc[sample, pr_col] if sample in clinical_df.index else None
        her2 = clinical_df.loc[sample, her2_col] if sample in clinical_df.index else None

        # Check for Negative status (exact match)
        er_neg = str(er) == 'Negative'
        pr_neg = str(pr) == 'Negative'
        her2_neg = str(her2) == 'Negative'

        # Check for Positive status
        er_pos = str(er) == 'Positive'
        pr_pos = str(pr) == 'Positive'
        her2_pos = str(her2) == 'Positive'

        # TNBC = all three negative
        if er_neg and pr_neg and her2_neg:
            labels[sample] = 'TNBC'
            tnbc_count += 1
        elif er_pos or pr_pos or her2_pos:
            # At least one positive = non-TNBC
            labels[sample] = 'non-TNBC'
            non_tnbc_count += 1
        else:
            # Unknown/indeterminate/equivocal - skip
            skipped += 1

    print(f"\nClassification results:")
    print(f"  TNBC: {tnbc_count}")
    print(f"  non-TNBC: {non_tnbc_count}")
    print(f"  Skipped (missing/equivocal): {skipped}")

    return labels


def preprocess_brca(expression_df, labels_dict):
    """Preprocess BRCA data similar to lung cancer pipeline."""

    # Filter to labeled samples
    samples = [s for s in expression_df.index if s in labels_dict]
    df = expression_df.loc[samples].copy()
    labels = pd.Series({s: labels_dict[s] for s in samples})

    print(f"\nPreprocessing {len(df)} samples...")

    # Remove genes with zero variance
    gene_vars = df.var()
    df = df.loc[:, gene_vars > 0]
    print(f"After variance filter: {df.shape[1]} genes")

    return df, labels


def differential_expression_analysis(X, y, alpha=0.05, log2fc_threshold=1.0):
    """Perform DEG analysis between TNBC and non-TNBC."""
    print("\nPerforming differential expression analysis...")

    tnbc_mask = y == 'TNBC'
    non_tnbc_mask = y == 'non-TNBC'

    results = []
    for gene in X.columns:
        tnbc_expr = X.loc[tnbc_mask, gene].values
        non_tnbc_expr = X.loc[non_tnbc_mask, gene].values

        # t-test
        t_stat, p_val = stats.ttest_ind(tnbc_expr, non_tnbc_expr)

        # Log2 fold change
        mean_tnbc = np.mean(tnbc_expr)
        mean_non_tnbc = np.mean(non_tnbc_expr)

        # Avoid division by zero
        if mean_non_tnbc > 0 and mean_tnbc > 0:
            log2fc = mean_tnbc - mean_non_tnbc  # Already log2 transformed
        else:
            log2fc = 0

        results.append({
            'gene': gene,
            'log2FoldChange': log2fc,
            'pvalue': p_val,
            't_statistic': t_stat
        })

    results_df = pd.DataFrame(results)

    # FDR correction
    _, padj, _, _ = multipletests(results_df['pvalue'].fillna(1), method='fdr_bh')
    results_df['padj'] = padj

    # Count significant DEGs
    sig_mask = (results_df['padj'] < alpha) & (results_df['log2FoldChange'].abs() > log2fc_threshold)
    print(f"Significant DEGs (padj < {alpha}, |log2FC| > {log2fc_threshold}): {sig_mask.sum()}")

    return results_df


def run_classification(X_train_raw, X_test_raw, y_train, y_test, n_genes, deg_results, tune_params=False):
    """Run classification with specified number of genes on pre-split data."""

    # Select top genes by significance
    sorted_degs = deg_results.sort_values('padj')
    top_genes = sorted_degs.head(n_genes)['gene'].tolist()

    X_train_sel = X_train_raw[top_genes]
    X_test_sel = X_test_raw[top_genes]

    # Scale features (fit on training data only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_sel)
    X_test = scaler.transform(X_test_sel)

    # Classifiers (matching original paper)
    classifiers = {
        'SVM': SVC(probability=True, random_state=RANDOM_STATE),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
    }

    # Parameter grids for tuning
    param_grids = {
        'SVM': {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'linear']},
        'KNN': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']},
        'Naive Bayes': {'var_smoothing': [1e-9, 1e-8, 1e-7]},
        'Decision Tree': {'max_depth': [5, 10, 15, None], 'min_samples_split': [2, 5, 10]},
    }

    results = []

    for name, clf in classifiers.items():
        if tune_params:
            # GridSearchCV
            grid = GridSearchCV(clf, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train, y_train)
            best_clf = grid.best_estimator_
            cv_score = grid.best_score_
        else:
            # Use defaults (more similar to original paper)
            best_clf = clf
            best_clf.fit(X_train, y_train)
            cv_score = cross_val_score(best_clf, X_train, y_train, cv=CV_FOLDS).mean()

        # Predict
        y_pred = best_clf.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')

        if hasattr(best_clf, 'predict_proba'):
            y_prob = best_clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = None

        results.append({
            'Classifier': name,
            'N_Genes': n_genes,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUC': auc,
            'CV_Score': cv_score
        })

        print(f"  {name}: Accuracy={accuracy:.4f}, F1={f1:.4f}, CV={cv_score:.4f}")

    return pd.DataFrame(results)


def main():
    """Main function to reproduce original paper results."""
    start_time = datetime.now()

    print("="*70)
    print("REPRODUCING ORIGINAL PAPER (PMC7909418)")
    print("Breast Cancer Classification: TNBC vs non-TNBC")
    print("="*70)

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(script_dir, '..')
    results_dir = os.path.join(project_dir, 'results')
    figures_dir = os.path.join(project_dir, 'figures')

    # Step 1: Download data
    print("\n" + "="*50)
    print("STEP 1: Download TCGA BRCA Data")
    print("="*50)

    expression_df = download_brca_data()
    clinical_df = get_brca_subtypes()

    # Step 2: Classify TNBC vs non-TNBC
    print("\n" + "="*50)
    print("STEP 2: Classify TNBC vs non-TNBC")
    print("="*50)

    labels = classify_tnbc(clinical_df, expression_df.index.tolist())

    if labels is None or len(labels) < 100:
        print("ERROR: Could not classify sufficient samples as TNBC/non-TNBC")
        return

    # Step 3: Preprocess
    print("\n" + "="*50)
    print("STEP 3: Preprocess Data")
    print("="*50)

    X, y = preprocess_brca(expression_df, labels)

    print(f"\nFinal dataset:")
    print(f"  Samples: {len(X)}")
    print(f"  Genes: {X.shape[1]}")
    print(f"  TNBC: {(y == 'TNBC').sum()}")
    print(f"  non-TNBC: {(y == 'non-TNBC').sum()}")

    # Step 4: Split and Differential Expression
    print("\n" + "="*50)
    print("STEP 4: Train/Test Split and Differential Expression Analysis")
    print("="*50)

    # Encode labels and split BEFORE DEG analysis to prevent data leakage
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        stratify=y_encoded,
        random_state=RANDOM_STATE
    )

    print(f"Train set: {len(y_train)} samples")
    print(f"Test set: {len(y_test)} samples")

    # DEG analysis on TRAINING data only
    y_train_labels = pd.Series(le.inverse_transform(y_train), index=X_train_raw.index)
    deg_results = differential_expression_analysis(X_train_raw, y_train_labels)

    # Step 5: Classification
    print("\n" + "="*50)
    print("STEP 5: Classification (WITHOUT hyperparameter tuning)")
    print("        (To match original paper methodology)")
    print("="*50)

    all_results = []

    for n_genes in GENE_THRESHOLDS:
        print(f"\n--- Top {n_genes} genes ---")
        results = run_classification(X_train_raw, X_test_raw, y_train, y_test, n_genes, deg_results, tune_params=False)
        all_results.append(results)

    all_results_df = pd.concat(all_results, ignore_index=True)

    # Save results
    results_path = os.path.join(results_dir, 'original_paper_reproduction.csv')
    all_results_df.to_csv(results_path, index=False)

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print("\nAccuracy by Classifier and Gene Threshold:")
    pivot = all_results_df.pivot(index='N_Genes', columns='Classifier', values='Accuracy')
    print(pivot.round(4).to_string())

    # Best result
    best_idx = all_results_df['Accuracy'].idxmax()
    best = all_results_df.loc[best_idx]

    print(f"\n{'─'*50}")
    print("OUR BEST RESULT:")
    print(f"  Classifier: {best['Classifier']}")
    print(f"  N_Genes: {int(best['N_Genes'])}")
    print(f"  Accuracy: {best['Accuracy']:.4f} ({best['Accuracy']*100:.1f}%)")
    print(f"  F1 Score: {best['F1']:.4f}")

    print(f"\n{'─'*50}")
    print("ORIGINAL PAPER RESULT:")
    print(f"  Classifier: SVM")
    print(f"  N_Genes: 256")
    print(f"  Accuracy: 0.90 (90%)")

    print(f"\n{'─'*50}")
    print("COMPARISON:")

    # Find SVM with 256 genes
    svm_256 = all_results_df[(all_results_df['Classifier'] == 'SVM') &
                              (all_results_df['N_Genes'] == 256)]
    if len(svm_256) > 0:
        our_svm = svm_256.iloc[0]['Accuracy']
        print(f"  Our SVM (256 genes): {our_svm:.4f} ({our_svm*100:.1f}%)")
        print(f"  Paper SVM (256 genes): 0.90 (90%)")
        print(f"  Difference: {(our_svm - 0.90)*100:+.1f}%")

    # Now run WITH hyperparameter tuning
    print("\n" + "="*70)
    print("BONUS: Classification WITH Hyperparameter Tuning")
    print("       (Our enhancement over original paper)")
    print("="*70)

    tuned_results = []
    for n_genes in [256, 512, 1024]:  # Just key thresholds for speed
        print(f"\n--- Top {n_genes} genes (with tuning) ---")
        results = run_classification(X_train_raw, X_test_raw, y_train, y_test, n_genes, deg_results, tune_params=True)
        tuned_results.append(results)

    tuned_df = pd.concat(tuned_results, ignore_index=True)

    print("\n" + "─"*50)
    print("WITH TUNING - Best Result:")
    best_tuned_idx = tuned_df['Accuracy'].idxmax()
    best_tuned = tuned_df.loc[best_tuned_idx]
    print(f"  Classifier: {best_tuned['Classifier']}")
    print(f"  N_Genes: {int(best_tuned['N_Genes'])}")
    print(f"  Accuracy: {best_tuned['Accuracy']:.4f} ({best_tuned['Accuracy']*100:.1f}%)")

    # Save tuned results
    tuned_path = os.path.join(results_dir, 'original_paper_reproduction_tuned.csv')
    tuned_df.to_csv(tuned_path, index=False)

    end_time = datetime.now()
    print(f"\n{'='*70}")
    print(f"Total runtime: {end_time - start_time}")
    print(f"Results saved to: {results_dir}")


if __name__ == '__main__':
    main()
