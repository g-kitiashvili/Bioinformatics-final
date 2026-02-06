# Lung Cancer Subtype Classification (LUAD vs LUSC)

**Bioinformatics Final Project**

---

## Abstract

Lung cancer is the leading cause of cancer-related deaths worldwide, with Lung Adenocarcinoma (LUAD) and Lung Squamous Cell Carcinoma (LUSC) being the two most common subtypes of non-small cell lung cancer (NSCLC). Accurate classification of these subtypes is crucial for treatment planning, as they respond differently to targeted therapies. This project applies a machine learning methodology originally developed for breast cancer classification (PMC7909418) to the problem of distinguishing LUAD from LUSC using gene expression data from The Cancer Genome Atlas (TCGA).

We analyzed 1,017 tumor samples (515 LUAD, 502 LUSC) with expression data for 20,530 genes. Using differential expression analysis on training data only (to prevent data leakage), we identified 2,843 significantly differentially expressed genes between the two subtypes. We evaluated five machine learning classifiers (SVM, KNN, Naive Bayes, Decision Tree, and Random Forest) across multiple gene selection thresholds (256, 512, 1024 genes). Our best result achieved **98.04% accuracy** using Naive Bayes with the top 256 differentially expressed genes, outperforming the original breast cancer study (90% accuracy). On a properly held-out validation set (20% of data, unseen during feature selection and scaling), four classifiers achieved 94.61% accuracy. To validate our implementation, we also reproduced the original paper's breast cancer experiment and achieved 93.6% accuracy compared to their reported 90%. These results demonstrate that gene expression-based machine learning approaches can effectively classify lung cancer subtypes and may have potential clinical applications in diagnosis support.


---

## 1. Introduction

Lung cancer remains the most common cause of cancer death globally, accounting for approximately 1.8 million deaths annually. Non-small cell lung cancer (NSCLC) represents about 85% of all lung cancer cases and includes two major histological subtypes. **Lung Adenocarcinoma (LUAD)** is the most common subtype, comprising roughly 40% of lung cancers, typically arising in the outer regions of the lung and more common in non-smokers and women. **Lung Squamous Cell Carcinoma (LUSC)** is the second most common subtype at 25-30% of cases, usually found in the central part of the lung near the bronchi and strongly associated with smoking.

Distinguishing between LUAD and LUSC is clinically critical. LUAD patients may benefit from EGFR and ALK targeted therapies, while LUSC patients typically do not respond to these treatments. The anti-angiogenic drug bevacizumab is contraindicated in LUSC due to risk of fatal hemorrhage. The subtypes also show different responses to immune checkpoint inhibitors, and their survival rates and disease progression patterns differ significantly.

Gene expression profiling measures the activity levels of thousands of genes simultaneously, creating a "molecular fingerprint" of a tumor. The Cancer Genome Atlas (TCGA) project has generated comprehensive gene expression data for thousands of cancer samples, enabling large-scale computational studies. Machine learning algorithms have been successfully applied to cancer classification using such data. The study PMC7909418 (2021) classified breast cancer subtypes (TNBC vs non-TNBC) using SVM, KNN, Naive Bayes, and Decision Tree, achieving 90% accuracy with 256 genes.

While the methodology in PMC7909418 proved effective for breast cancer, its applicability to lung cancer subtypes has not been systematically evaluated. This project fills this gap by applying the same rigorous methodology to LUAD vs LUSC classification. Our primary hypothesis is that **the machine learning methodology used for breast cancer subtype classification can be successfully applied to classify lung cancer subtypes with comparable or better accuracy**.

---

## 2. Methodology

The reference paper (PMC7909418) established a methodology for breast cancer subtype classification using log-CPM normalized gene expression data from TCGA, differential expression analysis with t-test and FDR correction (|log2FC| > 1) for feature selection, gene thresholds of 256, 512, and 1024, four classifiers (SVM, KNN, Naive Bayes, Decision Tree), a 90/10 stratified train/test split, 10-fold cross-validation, and standard metrics including accuracy, precision, recall, specificity, and F1.

We implemented the paper's methodology in Python using scikit-learn and extended it with several enhancements: GridSearchCV hyperparameter optimization for all classifiers, Random Forest as an additional ensemble classifier, ROC curves with AUC analysis, feature importance identification, Gene Ontology enrichment analysis for biological interpretation, external validation on a properly held-out set, volcano plot visualization, and outlier detection analysis to test classifier robustness.

Our implementation differs from the original paper in three aspects. First, we use scipy's t-test for differential expression analysis rather than LIMMA (Linear Models for Microarray Data). LIMMA uses empirical Bayes methods to moderate variance estimates and is more sophisticated, but our simpler t-test approach still identifies biologically meaningful genes. Second, we added hyperparameter tuning via GridSearchCV which the paper does not mention, suggesting they used default parameters. Third, we enforce a strict data leakage prevention protocol: the train/test split is performed **before** differential expression analysis and feature scaling, ensuring that feature selection and normalization parameters are derived exclusively from training data.

Gene expression data comes from [UCSC Xena](https://xenabrowser.net/datapages/) (TCGA Hub), specifically TCGA LUAD HiSeqV2 and TCGA LUSC HiSeqV2. The data format is Log2(normalized_count + 1) transformed RSEM gene expression values.

---

## 3. Results

### Lung Cancer Classification

The lung cancer dataset contains 1,017 total samples (515 LUAD, 502 LUSC) with expression data for 20,530 genes. Differential expression analysis (performed on training data only) identified 2,843 significant DEGs (padj < 0.05, |log2FC| > 1).

| N_Genes | SVM | KNN | Naive Bayes | Decision Tree | Random Forest |
|---------|-----|-----|-------------|---------------|---------------|
| 256 | 96.08% | 97.06% | **98.04%** | 95.10% | 96.08% |
| 512 | 96.08% | 95.10% | 97.06% | 96.08% | 96.08% |
| 1024 | 97.06% | 97.06% | 96.08% | 94.12% | 97.06% |

The best result was achieved by **Naive Bayes with 256 genes**: 98.04% accuracy, 97.96% F1 score, 0.9892 AUC, and 94.97% cross-validation mean accuracy. On external validation using a properly held-out 20% of data (separated before feature selection and scaling), SVM, KNN, Naive Bayes, and Random Forest all achieved 94.61% accuracy with AUCs up to 0.987 (Random Forest).

### Original Paper Reproduction (Breast Cancer)

To validate that our implementation correctly reproduces the original methodology, we ran our pipeline on the same breast cancer dataset (TNBC vs non-TNBC) used in PMC7909418. The breast cancer dataset contains 1,082 samples with receptor status information (128 TNBC, 954 non-TNBC) and 20,244 genes. Differential expression analysis (on training data only) identified 2,318 significant DEGs.

| N_Genes | SVM | KNN | Naive Bayes | Decision Tree |
|---------|-----|-----|-------------|---------------|
| 256 | 93.58% | 93.58% | 91.74% | 88.99% |
| 512 | 94.50% | 94.50% | 90.83% | 85.32% |
| 1024 | 94.50% | 95.41% | 90.83% | 89.91% |
| 2048 | 94.50% | **96.33%** | 90.83% | 88.99% |
| 4096 | 94.50% | **96.33%** | 90.83% | 88.99% |

Our SVM with 256 genes achieved **93.58%** compared to the paper's reported **90%**, a difference of +3.6%. The best overall result was KNN with 2048 genes at 96.33%. This confirms our implementation correctly reproduces and slightly exceeds the original paper's results.

### Comparison Summary

| Metric | Original Paper | Our Reproduction | Our Lung Cancer |
|--------|---------------|------------------|-----------------|
| Cancer Type | Breast (TNBC vs non-TNBC) | Breast (TNBC vs non-TNBC) | Lung (LUAD vs LUSC) |
| Samples | 934 | 1,082 | 1,017 |
| Significant DEGs | 5,502 | 2,318 | 2,843 |
| Best Classifier | SVM | KNN | Naive Bayes |
| Best Accuracy | 90.0% | 96.3% | 98.04% |
| SVM (256 genes) | 90.0% | 93.6% | 96.08% |

The difference in DEG counts (5,502 vs 2,318) is attributable to the paper's use of LIMMA versus our t-test approach. Despite this difference, classification performance is comparable or better, indicating both methods identify sufficiently discriminative genes.

---

## 4. Discussion

Our best model achieved **98.04% accuracy**, significantly exceeding our 90% hypothesis threshold and outperforming the original breast cancer study. This indicates that LUAD and LUSC have highly distinct gene expression profiles and that the methodology transfers well from breast to lung cancer. The best results came from using **256 genes**, consistent with the original study, confirming that cancer subtypes can be distinguished using a relatively small gene signature suitable for clinical diagnostic panels.

Unlike the original study where SVM performed best, **Naive Bayes** achieved the highest accuracy for lung cancer. Notably, when we reproduced the breast cancer experiment, KNN outperformed SVM in our implementation as well, suggesting default hyperparameter choices in the original study may have favored SVM.

The high classification accuracy reflects known biological differences between LUAD and LUSC. LUAD originates from Type II pneumocytes and Clara cells in the peripheral lung, commonly harbors EGFR, KRAS, and ALK mutations, and is typically TTF-1 positive. LUSC originates from basal cells of bronchial epithelium in the central airways, commonly harbors TP53, PIK3CA, and FGFR1 mutations, and shows squamous differentiation with keratinization. Gene Ontology enrichment analysis confirmed this biological interpretation, with top enriched terms being epidermis development (p=1.97e-12), skin development (p=1.27e-07), and keratinocyte differentiation (p=2.57e-06), consistent with LUSC's squamous differentiation.

### Novel Contribution: Outlier Detection

A key question not addressed by the original paper: **What happens when non-target data is passed to the classifier?** We tested the trained classifier on breast cancer samples (different organ) and normal lung tissue (same organ, non-cancerous). The classifier rejected 98% of breast cancer samples as outliers, but **accepted 98% of normal lung tissue**.

The classifier **cannot distinguish cancer from non-cancer** in the same tissue. It only detects samples from **different organs** as outliers. This happens because gene selection optimizes for LUAD vs LUSC distinction, making these genes lung-specific rather than tumor-specific. Normal lung tissue expresses these genes similarly to lung tumors.

**Clinical Implication:** This classifier should **only** be used to classify samples that are **already confirmed to be lung cancer**. It cannot determine whether a sample is cancerous. See `docs/OUTLIER_DETECTION_JOURNEY.md` for complete documentation of this scientific process.

### Limitations

The study has several limitations: potential batch effects from multi-institutional TCGA data collection, differences between frozen tissue and clinical FFPE samples, binary classification that does not account for mixed histology or rare subtypes, need for validation on independent datasets such as GEO, use of t-test rather than LIMMA for differential expression, inability to distinguish tumor from normal tissue, and a modest drop in validation accuracy (94.6%) compared to test accuracy (98.0%) that may reflect slight overfitting to the test split despite proper data leakage prevention.

---

## 5. Conclusions

The hypothesis was confirmed: the methodology successfully transfers to lung cancer with even better performance. A 256-gene signature achieves near-perfect classification (98.04% test accuracy, 94.61% on a properly held-out validation set), validating that the PMC7909418 approach generalizes to other cancer types. All results were obtained with a strict data leakage prevention protocol (train/test split before DEG analysis and scaling). Our reproduction of the original breast cancer experiment (93.6% vs reported 90%) confirms the implementation is correct.

The novel finding that the classifier detects tissue type rather than cancer type has important implications for clinical deployment: this tool should only be used on samples already confirmed to be lung cancer, not for cancer screening. External validation on independent datasets (e.g., GEO) and prospective clinical studies are recommended as future work.

---

## 6. Project Structure and Usage

```
lung_cancer_project/
├── README.md
├── requirements.txt
├── data/
│   └── tcga_lung_expression.csv
├── src/
│   ├── download_data.py
│   ├── preprocess.py
│   ├── classify_enhanced.py
│   ├── biological_analysis.py
│   ├── external_validation.py
│   ├── outlier_detection.py
│   ├── reproduce_original_paper.py
│   └── run_full_pipeline.py
├── results/
│   └── *.csv
└── figures/
    └── *.png
```

**Installation:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Usage:**

```bash
cd src
python run_full_pipeline.py                    # Full analysis
python run_full_pipeline.py --skip-download    # Skip data download
python run_full_pipeline.py --quick            # Skip hyperparameter tuning
python reproduce_original_paper.py             # Reproduce breast cancer experiment
```

---

## References

1. Original methodology paper: [PMC7909418](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7909418/)
2. TCGA Data: [UCSC Xena Browser](https://xenabrowser.net/)
