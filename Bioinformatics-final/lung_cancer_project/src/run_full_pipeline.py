"""
Full Pipeline Runner

Executes the complete enhanced analysis pipeline:
1. Download TCGA data
2. Enhanced classification with hyperparameter tuning
3. Biological analysis
4. External validation
5. Outlier detection

Usage:
    python run_full_pipeline.py [--skip-download] [--quick]
"""

import os
import sys
import argparse
from datetime import datetime


def run_step(step_name, function, *args, **kwargs):
    """Run a pipeline step with timing and error handling."""
    print(f"\n{'='*70}")
    print(f"STEP: {step_name}")
    print(f"{'='*70}")

    start = datetime.now()
    try:
        result = function(*args, **kwargs)
        elapsed = datetime.now() - start
        print(f"\n✓ {step_name} completed in {elapsed}")
        return result
    except Exception as e:
        print(f"\n✗ {step_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Run full lung cancer classification pipeline')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip data download if already exists')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: skip hyperparameter tuning')
    parser.add_argument('--skip-biological', action='store_true',
                        help='Skip biological analysis')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip external validation')

    args = parser.parse_args()

    print("="*70)
    print("LUNG CANCER CLASSIFICATION - FULL PIPELINE")
    print("="*70)
    print(f"Started at: {datetime.now()}")
    print(f"Options: skip_download={args.skip_download}, quick={args.quick}")

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    data_file = os.path.join(data_dir, 'tcga_lung_expression.csv')

    # Step 1: Download data
    if not args.skip_download or not os.path.exists(data_file):
        from download_data import main as download_main
        run_step("Download TCGA Data", download_main)
    else:
        print("\n⏭ Skipping download (data exists)")

    # Step 2: Enhanced classification
    from classify_enhanced import main as classify_main
    results = run_step(
        "Enhanced Classification with Hyperparameter Tuning",
        classify_main,
        tune_params=not args.quick
    )

    # Step 3: Biological analysis
    if not args.skip_biological:
        from biological_analysis import main as bio_main
        run_step("Biological Analysis (GO, KEGG)", bio_main)
    else:
        print("\n⏭ Skipping biological analysis")

    # Step 4: External validation
    if not args.skip_validation:
        from external_validation import main as val_main
        run_step("External Validation", val_main)
    else:
        print("\n⏭ Skipping external validation")

    # Step 5: Outlier detection
    from outlier_detection import main as outlier_main
    run_step("Outlier Detection (Novel Contribution)", outlier_main)

    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Finished at: {datetime.now()}")

    print("\nGenerated outputs:")
    print("  - results/enhanced_classification_results.csv")
    print("  - results/deg_analysis.csv")
    print("  - results/go_enrichment_*.csv")
    print("  - results/external_validation_results.csv")
    print("  - results/outlier_detection_results.csv")
    print("  - figures/*.png")



if __name__ == '__main__':
    main()
