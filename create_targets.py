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
from src.target_creation import TargetCreator

def main():
    print("=" * 75)
    print("RUNNING FUTURE-BASED TARGET CREATION & ML DATASET GENERATION PIPELINE")
    print("=" * 75)

    output_dir = r"d:\DS PROJECT\outputs"
    plots_dir = os.path.join(output_dir, "plots", "target_analysis")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # ---------------------------------------------------------
    # STEP 1: LOAD INPUT DATASETS
    # ---------------------------------------------------------
    clean_gtrends_path = os.path.join(output_dir, "clean_gtrends.csv")
    trend_features_path = os.path.join(output_dir, "trend_features.csv")

    if not os.path.exists(clean_gtrends_path) or not os.path.exists(trend_features_path):
        print("[INFO] Running preprocessing pipeline to generate clean data & features...")
        loader = DataLoader()
        raw_gt = loader.load_gtrends()
        prep = DataPreprocessor(data_loader=loader)
        df_clean = prep.clean_gtrends(raw_gt)
        meta_df = prep.generate_trend_metadata([c for c in df_clean.columns if c != 'date'])
        fe = FeatureEngineer()
        df_features = fe.generate_all_features(df_clean, meta_df)
    else:
        df_clean = pd.read_csv(clean_gtrends_path)
        df_features = pd.read_csv(trend_features_path)

    # ---------------------------------------------------------
    # STEP 2: COMPUTE TARGET LABELS
    # ---------------------------------------------------------
    print("\n[STEP 2] Computing 4-period lookahead target labels...")
    tc = TargetCreator(forecast_horizon=4)
    df_ml, df_debug = tc.compute_targets(df_features, df_clean)

    # Save final ML dataset
    ml_dataset_path = os.path.join(output_dir, "ml_dataset.csv")
    df_ml.to_csv(ml_dataset_path, index=False)
    print(f" -> Saved final ML dataset: {ml_dataset_path}")
    print(f" -> ML Dataset Shape: {df_ml.shape}")

    # Save debug dataset for auditing
    debug_path = os.path.join(output_dir, "target_analysis_debug.csv")
    df_debug.to_csv(debug_path, index=False)
    print(f" -> Saved target analysis debug dataset: {debug_path}")

    # ---------------------------------------------------------
    # STEP 3: ANTI-LEAKAGE & QUALITY CHECKS
    # ---------------------------------------------------------
    print("\n[STEP 3] Performing Anti-Leakage and Data Quality Validation...")
    forbidden_cols = ['future_val_t4', 'future_max', 'future_mean', 'future_growth_pct', 'future_max_growth_pct', 'next_4_values']
    leakage_found = [c for c in forbidden_cols if c in df_ml.columns]
    print(f" -> Future column leakage check in ML dataset: {len(leakage_found)} detected (Must be 0).")
    
    null_count = df_ml.isnull().sum().sum()
    inf_count = np.isinf(df_ml.select_dtypes(include=[np.number])).sum().sum()
    print(f" -> Total NaN values in ML dataset: {null_count}")
    print(f" -> Total Inf values in ML dataset: {inf_count}")

    # Check row counts & dropped rows
    orig_rows = len(df_features)
    final_rows = len(df_ml)
    dropped_rows = orig_rows - final_rows
    print(f" -> Original feature rows : {orig_rows}")
    print(f" -> Invalid end rows dropped: {dropped_rows} (96 trends x 4 trailing weeks)")
    print(f" -> Final valid ML rows   : {final_rows}")

    # ---------------------------------------------------------
    # STEP 4: CLASS BALANCE ANALYSIS
    # ---------------------------------------------------------
    print("\n[STEP 4] Class Distribution Analysis:")
    counts = df_ml['target_label'].value_counts()
    percentages = df_ml['target_label'].value_counts(normalize=True) * 100
    class_summary_df = pd.DataFrame({'Count': counts, 'Percentage (%)': percentages.round(2)})
    print(class_summary_df.to_string())

    # ---------------------------------------------------------
    # STEP 5: TARGET ANALYSIS VISUALIZATIONS
    # ---------------------------------------------------------
    print("\n[STEP 5] Generating target analysis plots...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Plot 1: Target Class Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = ['#2ca02c', '#ff7f0e', '#d62728']
    sns.barplot(x=counts.index, y=counts.values, palette=palette, ax=ax)
    ax.set_title('Target Class Distribution', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Observations')
    ax.set_xlabel('Target Class Label')
    for i, (cnt, pct) in enumerate(zip(counts.values, percentages.values)):
        ax.text(i, cnt + 200, f"{cnt:,}\n({pct:.1f}%)", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    p1_path = os.path.join(plots_dir, "target_class_distribution.png")
    plt.savefig(p1_path, dpi=200)
    plt.close()
    print(f" -> Saved plot: {p1_path}")

    # Plot 2: Future Growth Distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    growth_clipped = np.clip(df_debug['future_growth_pct'], -100, 200)
    sns.histplot(growth_clipped, bins=50, kde=True, color='#1f77b4', ax=ax)
    ax.axvline(5.0, color='#ff7f0e', linestyle='--', label='Emerging Threshold (+5%)')
    ax.axvline(25.0, color='#d62728', linestyle='--', label='Strong Threshold (+25%)')
    ax.set_title('Distribution of 4-Week Future Growth Rate (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Future Growth Rate (%) [Clipped for Display: -100% to +200%]')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    p2_path = os.path.join(plots_dir, "future_growth_distribution.png")
    plt.savefig(p2_path, dpi=200)
    plt.close()
    print(f" -> Saved plot: {p2_path}")

    # Plot 3: Future Max Growth Distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    max_growth_clipped = np.clip(df_debug['future_max_growth_pct'], -100, 300)
    sns.histplot(max_growth_clipped, bins=50, kde=True, color='#9467bd', ax=ax)
    ax.axvline(15.0, color='#ff7f0e', linestyle='--', label='Emerging Max Threshold (+15%)')
    ax.axvline(35.0, color='#d62728', linestyle='--', label='Strong Max Threshold (+35%)')
    ax.set_title('Distribution of 4-Week Future Maximum Growth Rate (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Future Max Growth Rate (%) [Clipped for Display: -100% to +300%]')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    p3_path = os.path.join(plots_dir, "future_max_growth_distribution.png")
    plt.savefig(p3_path, dpi=200)
    plt.close()
    print(f" -> Saved plot: {p3_path}")

    # Plot 4: Example Emerging Trend Progression
    # Find an actual emerging trend observation
    emerging_samples = df_debug[df_debug['target_label'] == 'Emerging Trend']
    ex_row = emerging_samples.iloc[10]
    ex_trend = ex_row['trend_name']
    ex_date = pd.to_datetime(ex_row['date'])

    sub_gt = df_clean[['date', ex_trend]].copy()
    sub_gt['date'] = pd.to_datetime(sub_gt['date'])
    sub_gt = sub_gt.sort_values('date').reset_index(drop=True)
    idx = sub_gt[sub_gt['date'] == ex_date].index[0]

    window_gt = sub_gt.iloc[max(0, idx - 8) : idx + 9].copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(window_gt['date'], window_gt[ex_trend], marker='o', color='#1f77b4', label='Search Index')
    ax.axvline(ex_date, color='red', linestyle='--', label=f'Current Point t ({ex_date.strftime("%Y-%m-%d")})')
    
    # Highlight future window t+1..t+4
    fut_dates = sub_gt['date'].iloc[idx+1 : idx+5]
    fut_vals = sub_gt[ex_trend].iloc[idx+1 : idx+5]
    ax.plot(fut_dates, fut_vals, marker='s', color='#ff7f0e', linewidth=2.5, label='Future Window (t+1..t+4)')

    ax.set_title(f"Emerging Trend Example: '{ex_trend}' (t = {ex_date.strftime('%Y-%m-%d')})", fontsize=13, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Search Index')
    ax.legend()
    plt.tight_layout()
    p4_path = os.path.join(plots_dir, "example_emerging_trend.png")
    plt.savefig(p4_path, dpi=200)
    plt.close()
    print(f" -> Saved plot: {p4_path}")

    # Plot 5: Comparison of False Spike vs Sustained Trend
    # Find a false spike example (max spike but end drops) and a sustained trend
    false_spikes = df_debug[(df_debug['future_max_growth_pct'] >= 30.0) & (df_debug['target'] == 0)]
    sustained_trends = df_debug[df_debug['target'] == 2]

    if len(false_spikes) > 0 and len(sustained_trends) > 0:
        fs_row = false_spikes.iloc[0]
        st_row = sustained_trends.iloc[0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: False Spike
        fs_trend = fs_row['trend_name']
        fs_dt = pd.to_datetime(fs_row['date'])
        fs_sub = df_clean[['date', fs_trend]].copy()
        fs_sub['date'] = pd.to_datetime(fs_sub['date'])
        fs_idx = fs_sub[fs_sub['date'] == fs_dt].index[0]
        fs_win = fs_sub.iloc[max(0, fs_idx - 4) : fs_idx + 6].copy()

        ax1.plot(fs_win['date'], fs_win[fs_trend], marker='o', color='#d62728', label='Search Index')
        ax1.axvline(fs_dt, color='black', linestyle='--', label=f'Current t ({fs_dt.strftime("%Y-%m-%d")})')
        ax1.plot(fs_sub['date'].iloc[fs_idx+1:fs_idx+5], fs_sub[fs_trend].iloc[fs_idx+1:fs_idx+5], marker='s', color='orange', linewidth=2, label='Future 4 Weeks')
        ax1.set_title(f"False Spike (Class 0: Stable)\n'{fs_trend}': Temporary peak collapses back", fontsize=11, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Search Index')
        ax1.legend()

        # Panel 2: Sustained Strong Trend
        st_trend = st_row['trend_name']
        st_dt = pd.to_datetime(st_row['date'])
        st_sub = df_clean[['date', st_trend]].copy()
        st_sub['date'] = pd.to_datetime(st_sub['date'])
        st_idx = st_sub[st_sub['date'] == st_dt].index[0]
        st_win = st_sub.iloc[max(0, st_idx - 4) : st_idx + 6].copy()

        ax2.plot(st_win['date'], st_win[st_trend], marker='o', color='#2ca02c', label='Search Index')
        ax2.axvline(st_dt, color='black', linestyle='--', label=f'Current t ({st_dt.strftime("%Y-%m-%d")})')
        ax2.plot(st_sub['date'].iloc[st_idx+1:st_idx+5], st_sub[st_trend].iloc[st_idx+1:st_idx+5], marker='s', color='green', linewidth=2, label='Future 4 Weeks')
        ax2.set_title(f"Sustained Outbreak (Class 2: Strong Trend)\n'{st_trend}': Continuous growth progression", fontsize=11, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Search Index')
        ax2.legend()

        plt.tight_layout()
        p5_path = os.path.join(plots_dir, "example_false_spike_vs_sustained.png")
        plt.savefig(p5_path, dpi=200)
        plt.close()
        print(f" -> Saved plot: {p5_path}")

    # ---------------------------------------------------------
    # STEP 6: MANUAL VERIFICATION OF 5 EXAMPLES
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print("MANUAL VERIFICATION OF 5 ACTUAL EXAMPLES ACROSS DIFFERENT TRENDS")
    print("=" * 75)

    sample_indices = [100, 2500, 7500, 12000, 18000]
    verify_rows = df_debug.iloc[sample_indices].copy()

    for idx, row in verify_rows.iterrows():
        print(f"\nExample Trend     : '{row['trend_name']}'")
        print(f"Current Date      : {pd.to_datetime(row['date']).strftime('%Y-%m-%d')}")
        print(f"Current Value (y_t): {row['current_val_t']}")
        print(f"Next 4 Values     : {row['next_4_values']}")
        print(f"Future Val (t+4)  : {row['future_val_t4']}")
        print(f"Future Max        : {row['future_max']}")
        print(f"Future Growth (%) : {row['future_growth_pct']:.2f}%")
        print(f"Future Max Growth%: {row['future_max_growth_pct']:.2f}%")
        print(f"Sustained Trend?  : {row['is_sustained']}")
        print(f"Assigned Target   : {row['target']} ({row['target_label']})")
        print("-" * 55)

    print("\n" + "=" * 75)
    print("TARGET CREATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    main()
