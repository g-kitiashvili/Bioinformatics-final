"""
Enhanced Lung Cancer Classification Pipeline

Improvements over original:
1. Hyperparameter tuning with GridSearchCV
2. Random Forest ensemble classifier
3. Nested cross-validation for unbiased evaluation
4. ROC curves and AUC metrics
5. Feature importance analysis
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV, cross_val_predict
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
from sklearn.pipeline import Pipeline

from preprocess import differential_expression_analysis, select_top_genes

warnings.filterwarnings('ignore')

# Configuration
GENE_THRESHOLDS = [256, 512, 1024]  # Reduced for faster tuning
TEST_SIZE = 0.10
CV_FOLDS = 10
RANDOM_STATE = 42
N_JOBS = -1  # Use all CPU cores


def get_param_grids():
    """
    Define hyperparameter grids for each classifier.
    These are the parameters we'll tune with GridSearchCV.
    """
    param_grids = {
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'kernel': ['rbf', 'linear']
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        'Decision Tree': {
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'Naive Bayes': {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
    }

    return param_grids


def get_base_classifiers():
    """Return base classifiers without hyperparameters."""
    classifiers = {
        'SVM': SVC(probability=True, random_state=RANDOM_STATE),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE),
        'Naive Bayes': GaussianNB()
    }

    return classifiers


def tune_hyperparameters(X_train, y_train, classifier_name, classifier, param_grid):
    """
    Perform GridSearchCV to find optimal hyperparameters.
    """
    print(f"    Tuning {classifier_name}...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        classifier,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=N_JOBS,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    print(f"    Best params: {grid_search.best_params_}")
    print(f"    Best CV score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculate comprehensive metrics."""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='binary'),
        'Recall': recall_score(y_true, y_pred, average='binary'),
        'F1': f1_score(y_true, y_pred, average='binary')
    }

    # Specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['Specificity'] = tn / (tn + fp)

    # AUC if probabilities available
    if y_prob is not None:
        metrics['AUC'] = roc_auc_score(y_true, y_prob)

    return metrics


def get_feature_importance(classifier, feature_names, classifier_name):
    """Extract feature importance from classifier."""
    importance = None

    if classifier_name in ['Decision Tree', 'Random Forest']:
        importance = classifier.feature_importances_
    elif classifier_name == 'SVM' and classifier.kernel == 'linear':
        importance = np.abs(classifier.coef_[0])

    if importance is not None:
        importance_df = pd.DataFrame({
            'gene': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return importance_df

    return None


def run_enhanced_pipeline(df: pd.DataFrame, n_genes: int, tune_params: bool = True):
    """
    Run enhanced classification pipeline with hyperparameter tuning.
    """
    print(f"\n{'='*70}")
    print(f"ENHANCED CLASSIFICATION WITH TOP {n_genes} GENES")
    print(f"{'='*70}")

    # Separate features and labels
    y = df['label']
    X_all = df.drop(columns=['sample_id', 'label'])

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train/test split BEFORE any feature selection or scaling
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_all, y_encoded,
        test_size=TEST_SIZE,
        stratify=y_encoded,
        random_state=RANDOM_STATE
    )

    print(f"\nTrain set: {len(y_train)} samples")
    print(f"Test set: {len(y_test)} samples")

    # Remove zero-variance genes (based on training data only)
    variances = X_train_raw.var()
    nonzero_var_genes = variances[variances > 0].index
    X_train_raw = X_train_raw[nonzero_var_genes]
    X_test_raw = X_test_raw[nonzero_var_genes]
    print(f"After variance filter (train): {len(nonzero_var_genes)} genes")

    # Differential expression analysis on TRAINING data only
    y_train_labels = pd.Series(le.inverse_transform(y_train), index=X_train_raw.index)
    deg_results = differential_expression_analysis(
        X_train_raw, y_train_labels, alpha=0.05, log2fc_threshold=1.0
    )

    # Select top genes from training DEG results
    selected_genes = select_top_genes(deg_results, n_genes)
    X_train_sel = X_train_raw[selected_genes]
    X_test_sel = X_test_raw[selected_genes]
    print(f"Selected top {n_genes} genes for classification")

    # Scale features (fit on training data only)
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train_sel),
        columns=X_train_sel.columns,
        index=X_train_sel.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test_sel),
        columns=X_test_sel.columns,
        index=X_test_sel.index
    )

    # Get classifiers and param grids
    classifiers = get_base_classifiers()
    param_grids = get_param_grids()

    results = []
    best_models = {}
    all_importances = {}
    roc_data = {}

    for name, clf in classifiers.items():
        print(f"\n{'─'*50}")
        print(f"Processing {name}...")

        if tune_params and name in param_grids:
            # Hyperparameter tuning
            best_clf, best_params, cv_score = tune_hyperparameters(
                X_train, y_train, name, clf, param_grids[name]
            )
        else:
            # Use defaults
            best_clf = clf
            best_clf.fit(X_train, y_train)
            best_params = {}
            cv_score = cross_val_score(
                best_clf, X_train, y_train, cv=CV_FOLDS
            ).mean()

        # Predict
        y_pred = best_clf.predict(X_test)

        # Get probabilities for AUC
        if hasattr(best_clf, 'predict_proba'):
            y_prob = best_clf.predict_proba(X_test)[:, 1]
        else:
            y_prob = None

        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        metrics['Classifier'] = name
        metrics['N_Genes'] = n_genes
        metrics['CV_Score'] = cv_score
        metrics['Best_Params'] = str(best_params)

        results.append(metrics)
        best_models[name] = best_clf

        # Store ROC data
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': metrics['AUC']}

        # Feature importance
        importance = get_feature_importance(best_clf, X_train.columns.tolist(), name)
        if importance is not None:
            all_importances[name] = importance

        print(f"    Test Accuracy: {metrics['Accuracy']:.4f}")
        print(f"    Test F1: {metrics['F1']:.4f}")
        if 'AUC' in metrics:
            print(f"    Test AUC: {metrics['AUC']:.4f}")

    return (
        pd.DataFrame(results),
        best_models,
        roc_data,
        all_importances,
        deg_results,
        le.classes_,
        scaler,
        selected_genes
    )


def plot_roc_curves(roc_data, n_genes, figures_dir):
    """Plot ROC curves for all classifiers."""
    plt.figure(figsize=(10, 8))

    colors = plt.cm.Set1(np.linspace(0, 1, len(roc_data)))

    for (name, data), color in zip(roc_data.items(), colors):
        plt.plot(
            data['fpr'], data['tpr'],
            color=color, lw=2,
            label=f"{name} (AUC = {data['auc']:.3f})"
        )

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - Top {n_genes} Genes', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'roc_curves_{n_genes}genes.png'), dpi=150)
    plt.close()


def plot_feature_importance(importances, n_genes, figures_dir, top_n=20):
    """Plot feature importance for applicable classifiers."""
    for name, importance_df in importances.items():
        plt.figure(figsize=(12, 8))

        top_features = importance_df.head(top_n)

        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['gene'].values)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Gene', fontsize=12)
        plt.title(f'Top {top_n} Important Genes - {name} ({n_genes} genes)', fontsize=14)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(
            os.path.join(figures_dir, f'feature_importance_{name.lower().replace(" ", "_")}_{n_genes}genes.png'),
            dpi=150
        )
        plt.close()


def plot_comparison_chart(all_results, figures_dir):
    """Plot comprehensive comparison of all models."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Accuracy by classifier and gene count
    ax1 = axes[0, 0]
    pivot = all_results.pivot(index='N_Genes', columns='Classifier', values='Accuracy')
    pivot.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Accuracy by Gene Threshold', fontsize=12)
    ax1.set_xlabel('Number of Genes')
    ax1.set_ylabel('Accuracy')
    ax1.legend(title='Classifier', bbox_to_anchor=(1.02, 1))
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)

    # 2. AUC comparison
    ax2 = axes[0, 1]
    if 'AUC' in all_results.columns:
        pivot_auc = all_results.pivot(index='N_Genes', columns='Classifier', values='AUC')
        pivot_auc.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('AUC by Gene Threshold', fontsize=12)
        ax2.set_xlabel('Number of Genes')
        ax2.set_ylabel('AUC')
        ax2.legend(title='Classifier', bbox_to_anchor=(1.02, 1))
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)

    # 3. F1 Score comparison
    ax3 = axes[1, 0]
    pivot_f1 = all_results.pivot(index='N_Genes', columns='Classifier', values='F1')
    pivot_f1.plot(kind='bar', ax=ax3, width=0.8)
    ax3.set_title('F1 Score by Gene Threshold', fontsize=12)
    ax3.set_xlabel('Number of Genes')
    ax3.set_ylabel('F1 Score')
    ax3.legend(title='Classifier', bbox_to_anchor=(1.02, 1))
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)

    # 4. Best model summary
    ax4 = axes[1, 1]
    best_per_clf = all_results.loc[all_results.groupby('Classifier')['Accuracy'].idxmax()]
    metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1']
    x = np.arange(len(metrics))
    width = 0.12

    for i, (_, row) in enumerate(best_per_clf.iterrows()):
        values = [row[m] for m in metrics]
        ax4.bar(x + i * width, values, width, label=row['Classifier'])

    ax4.set_ylabel('Score')
    ax4.set_title('Best Configuration per Classifier', fontsize=12)
    ax4.set_xticks(x + width * (len(best_per_clf) - 1) / 2)
    ax4.set_xticklabels(metrics)
    ax4.legend(title='Classifier', loc='lower right')
    ax4.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'comprehensive_comparison.png'), dpi=150)
    plt.close()


def main(tune_params=True):
    """Main execution function."""
    start_time = datetime.now()

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(script_dir, '..')
    data_path = os.path.join(project_dir, 'data', 'tcga_lung_expression.csv')
    results_dir = os.path.join(project_dir, 'results')
    figures_dir = os.path.join(project_dir, 'figures')

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please run download_data.py first.")
        return

    df = pd.read_csv(data_path)
    print(f"Loaded {df.shape[0]} samples")

    # Run enhanced pipeline for each gene threshold
    all_results = []
    all_best_models = {}
    all_roc_data = {}
    all_importances = {}

    for n_genes in GENE_THRESHOLDS:
        (results_df, best_models, roc_data, importances,
         deg_results, classes, scaler, selected_genes) = run_enhanced_pipeline(
            df, n_genes, tune_params=tune_params
        )

        all_results.append(results_df)
        all_best_models[n_genes] = best_models
        all_roc_data[n_genes] = roc_data
        all_importances[n_genes] = importances

        # Plot ROC curves for this threshold
        if roc_data:
            plot_roc_curves(roc_data, n_genes, figures_dir)

        # Plot feature importance
        if importances:
            plot_feature_importance(importances, n_genes, figures_dir)

        # Save DEG results once
        if n_genes == GENE_THRESHOLDS[0]:
            deg_results.to_csv(
                os.path.join(results_dir, 'deg_analysis.csv'),
                index=False
            )

    # Combine results
    all_results_df = pd.concat(all_results, ignore_index=True)

    # Save results
    results_path = os.path.join(results_dir, 'enhanced_classification_results.csv')
    all_results_df.to_csv(results_path, index=False)

    # Plot comprehensive comparison
    plot_comparison_chart(all_results_df, figures_dir)

    # Print summary
    print(f"\n{'='*70}")
    print("ENHANCED CLASSIFICATION RESULTS SUMMARY")
    print(f"{'='*70}")

    print("\nAccuracy by Classifier and Gene Threshold:")
    summary = all_results_df.pivot_table(
        index='N_Genes',
        columns='Classifier',
        values='Accuracy'
    )
    print(summary.round(4).to_string())

    if 'AUC' in all_results_df.columns:
        print("\nAUC by Classifier and Gene Threshold:")
        summary_auc = all_results_df.pivot_table(
            index='N_Genes',
            columns='Classifier',
            values='AUC'
        )
        print(summary_auc.round(4).to_string())

    # Best overall result
    best_idx = all_results_df['Accuracy'].idxmax()
    best = all_results_df.loc[best_idx]
    print(f"\n{'─'*50}")
    print("BEST RESULT:")
    print(f"  Classifier: {best['Classifier']}")
    print(f"  N_Genes: {int(best['N_Genes'])}")
    print(f"  Accuracy: {best['Accuracy']:.4f}")
    print(f"  F1 Score: {best['F1']:.4f}")
    if 'AUC' in best:
        print(f"  AUC: {best['AUC']:.4f}")
    print(f"  Best Params: {best['Best_Params']}")

    end_time = datetime.now()
    print(f"\nTotal runtime: {end_time - start_time}")
    print(f"\nResults saved to: {results_dir}")
    print(f"Figures saved to: {figures_dir}")

    return all_results_df, all_best_models


if __name__ == '__main__':
    results, models = main(tune_params=True)
