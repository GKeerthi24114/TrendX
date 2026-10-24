import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer

def run_pipeline():
    """
    End-to-End Data Preprocessing and Feature Engineering Pipeline
    for Early Trend Detection System.
    """
    print("=" * 70)
    print("STARTING DATA PREPROCESSING & TREND FEATURE ENGINEERING PIPELINE")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "outputs")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # ---------------------------------------------------------
    # STEP 1: LOAD & CLEAN GTRENDS DATA
    # ---------------------------------------------------------
    print("\n[STEP 1] Loading and cleaning gtrends.csv...")
    loader = DataLoader()
    raw_gtrends = loader.load_gtrends()
    preprocessor = DataPreprocessor(data_loader=loader)
    
    clean_gtrends = preprocessor.clean_gtrends(raw_gtrends)
    clean_gtrends_path = os.path.join(output_dir, "clean_gtrends.csv")
    clean_gtrends.to_csv(clean_gtrends_path, index=False)
    print(f" -> Saved cleaned wide gtrends data: {clean_gtrends_path}")
    print(f" -> Cleaned gtrends shape: {clean_gtrends.shape} (Dates: {clean_gtrends['date'].min().strftime('%Y-%m-%d')} to {clean_gtrends['date'].max().strftime('%Y-%m-%d')})")

    # ---------------------------------------------------------
    # STEP 2: CATEGORIZE TREND VARIABLES & METADATA
    # ---------------------------------------------------------
    print("\n[STEP 2] Categorizing trend variables and creating metadata...")
    trend_cols = [c for c in clean_gtrends.columns if c != 'date']
    meta_df = preprocessor.generate_trend_metadata(trend_cols)
    metadata_path = os.path.join(output_dir, "trend_metadata.csv")
    meta_df.to_csv(metadata_path, index=False)
    print(f" -> Saved trend metadata: {metadata_path}")
    type_counts = meta_df['trend_type'].value_counts().to_dict()
    print(f" -> Trend Variable Categorization: {type_counts}")

    # ---------------------------------------------------------
    # STEP 3: CONVERT TO LONG-FORMAT TIME-SERIES
    # ---------------------------------------------------------
    print("\n[STEP 3] Converting wide data to long-format time-series...")
    long_timeseries = preprocessor.transform_to_long_format(clean_gtrends, meta_df)
    timeseries_path = os.path.join(output_dir, "trend_timeseries.csv")
    long_timeseries.to_csv(timeseries_path, index=False)
    print(f" -> Saved long-format time-series: {timeseries_path}")
    print(f" -> Long-format shape: {long_timeseries.shape}")

    # ---------------------------------------------------------
    # STEP 4 & 5 & 6: GENERATE HISTORICAL FEATURES (ANTI-LEAKAGE)
    # ---------------------------------------------------------
    print("\n[STEP 4-6] Engineering 11 historical time-series features...")
    feature_engineer = FeatureEngineer(window_size_short=4, window_size_long=8)
    trend_features = feature_engineer.generate_all_features(clean_gtrends, meta_df)
    features_path = os.path.join(output_dir, "trend_features.csv")
    trend_features.to_csv(features_path, index=False)
    print(f" -> Saved final trend features dataset: {features_path}")
    print(f" -> Feature dataset shape: {trend_features.shape}")

    # ---------------------------------------------------------
    # STEP 7: QUALITY CHECKS & INITIAL ROWS VERIFICATION
    # ---------------------------------------------------------
    print("\n[STEP 7] Performing data quality and anti-leakage checks...")
    null_count = trend_features.isnull().sum().sum()
    inf_count = np.isinf(trend_features.select_dtypes(include=[np.number])).sum().sum()
    print(f" -> Total NaN values in feature dataset: {null_count}")
    print(f" -> Total Inf values in feature dataset: {inf_count}")
    
    # Check chronological ordering per trend
    is_sorted = True
    for name, group in trend_features.groupby('trend_name'):
        if not group['date'].is_monotonic_increasing:
            is_sorted = False
            break
    print(f" -> Chronological ordering per trend verified: {is_sorted}")

    # ---------------------------------------------------------
    # STEP 8: GENERATE VISUAL VALIDATION PLOTS
    # ---------------------------------------------------------
    print("\n[STEP 8] Generating visual validation plots...")
    sample_trends = ['miniskirt', 'yellow', 'cotton']
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    for t_name in sample_trends:
        subset = trend_features[trend_features['trend_name'] == t_name].copy()
        subset['date'] = pd.to_datetime(subset['date'])

        fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        fig.suptitle(f"Visual Feature Validation: '{t_name}' ({subset['trend_type'].iloc[0].upper()})", fontsize=14, fontweight='bold')

        # Subplot 1: Raw Trend & Moving Averages
        axes[0].plot(subset['date'], subset['current_value'], label='Raw Index (current_value)', color='#1f77b4', alpha=0.6, linewidth=1.5)
        axes[0].plot(subset['date'], subset['moving_average_4'], label='4-Wk MA (moving_average_4)', color='#ff7f0e', linewidth=2.0)
        axes[0].plot(subset['date'], subset['moving_average_8'], label='8-Wk MA (moving_average_8)', color='#2ca02c', linestyle='--', linewidth=2.0)
        axes[0].set_ylabel('Search Index')
        axes[0].legend(loc='upper left')
        axes[0].set_title('Raw Search Volume & Moving Averages')

        # Subplot 2: Percentage Change
        axes[1].plot(subset['date'], subset['percentage_change'], label='Percentage Change (%)', color='#d62728', linewidth=1.5)
        axes[1].axhline(0, color='black', linestyle=':', alpha=0.5)
        axes[1].set_ylabel('Weekly % Change')
        axes[1].legend(loc='upper left')
        axes[1].set_title('Weekly Percentage Change (%)')

        # Subplot 3: Trend Slope & Momentum
        axes[2].plot(subset['date'], subset['trend_slope'], label='Trend Slope (4-Wk OLS)', color='#9467bd', linewidth=1.8)
        axes[2].plot(subset['date'], subset['momentum'], label='Momentum (Acceleration)', color='#8c564b', linestyle=':', linewidth=1.5)
        axes[2].axhline(0, color='black', linestyle=':', alpha=0.5)
        axes[2].set_ylabel('Slope / Acceleration')
        axes[2].legend(loc='upper left')
        axes[2].set_title('Trend Slope & Momentum')

        # Subplot 4: Volatility & Persistence
        ax4_1 = axes[3]
        ax4_2 = ax4_1.twinx()
        l1 = ax4_1.plot(subset['date'], subset['volatility'], label='Volatility (Coef of Var)', color='#e377c2', linewidth=1.5)
        l2 = ax4_2.plot(subset['date'], subset['persistence'], label='Persistence (Consecutive Up Wks)', color='#7f7f7f', linestyle='-.', linewidth=1.5)
        ax4_1.set_ylabel('Volatility Ratio')
        ax4_2.set_ylabel('Persistence (Weeks)')
        
        # Combined legend for twin axes
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax4_1.legend(lines, labels, loc='upper left')
        ax4_1.set_title('Volatility & Trend Persistence')

        plt.xlabel('Date')
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, f"validation_{t_name}.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f" -> Saved validation plot: {plot_path}")

    # ---------------------------------------------------------
    # STEP 9: GENERATE FEATURE SUMMARY REPORT
    # ---------------------------------------------------------
    print("\n[STEP 9] Generating feature summary report...")
    num_trends = meta_df['trend_name'].nunique()
    num_time_points = clean_gtrends['date'].nunique()
    num_obs = len(trend_features)
    num_features = len(feature_engineer.feature_names)
    date_min = clean_gtrends['date'].min().strftime('%Y-%m-%d')
    date_max = clean_gtrends['date'].max().strftime('%Y-%m-%d')

    numeric_cols = feature_engineer.feature_names
    stats_df = trend_features[numeric_cols].describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]

    summary_dict = {
        "num_trend_variables": int(num_trends),
        "num_time_points": int(num_time_points),
        "num_generated_observations": int(num_obs),
        "num_generated_features": int(num_features),
        "missing_values_after_fe": int(null_count),
        "date_range": {
            "start_date": date_min,
            "end_date": date_max
        },
        "trend_categories_distribution": type_counts,
        "feature_descriptive_stats": stats_df.to_dict(orient='index')
    }

    summary_json_path = os.path.join(output_dir, "feature_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_dict, f, indent=4)
    print(f" -> Saved summary JSON: {summary_json_path}")

    summary_txt_path = os.path.join(output_dir, "feature_summary.txt")
    with open(summary_txt_path, "w") as f:
        f.write("========================================================================\n")
        f.write("EARLY TREND DETECTION - PREPROCESSING & FEATURE ENGINEERING SUMMARY\n")
        f.write("========================================================================\n\n")
        f.write(f"Number of Trend Variables    : {num_trends}\n")
        f.write(f"Number of Time Points (Weeks): {num_time_points}\n")
        f.write(f"Total Generated Observations : {num_obs}\n")
        f.write(f"Generated Feature Count      : {num_features}\n")
        f.write(f"Missing Values (Post-FE)     : {null_count}\n")
        f.write(f"Date Range                   : {date_min} to {date_max}\n")
        f.write(f"Trend Categories Breakup     : {type_counts}\n\n")
        f.write("------------------------------------------------------------------------\n")
        f.write("FEATURE DESCRIPTIVE STATISTICS:\n")
        f.write("------------------------------------------------------------------------\n")
        f.write(stats_df.to_string())
        f.write("\n========================================================================\n")
    print(f" -> Saved summary TXT: {summary_txt_path}")

    # ---------------------------------------------------------
    # STEP 10: PRINT MANUAL VERIFICATION EXAMPLE
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("MANUAL VERIFICATION EXAMPLE: 'miniskirt' (First 5 Weeks)")
    print("=" * 70)
    sample_verify = trend_features[trend_features['trend_name'] == 'miniskirt'].head(5)
    print(sample_verify.to_string(index=False))
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
