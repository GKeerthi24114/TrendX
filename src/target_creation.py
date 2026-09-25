import pandas as pd
import numpy as np
import os

class TargetCreator:
    """
    Target Creation Module for Early Trend Detection System.
    
    Generates a 3-class future-based target over a 4-week forecast horizon (t+1 to t+4):
      0 = Stable
      1 = Emerging Trend
      2 = Strong Trend
      
    STRICT ANTI-LEAKAGE DESIGN:
    Target labels are computed using future observations (t+1 ... t+4). 
    All future variables (future_value, future_max, future_growth, etc.) 
    are strictly excluded from the final ML input dataset.
    """
    def __init__(self, forecast_horizon=4):
        self.forecast_horizon = forecast_horizon

    def compute_targets(self, df_features, df_clean_gtrends):
        """
        Computes target labels using historical features dataframe and 
        the cleaned wide gtrends time-series dataframe.
        
        Args:
            df_features (pd.DataFrame): Output from FeatureEngineer (outputs/trend_features.csv).
            df_clean_gtrends (pd.DataFrame): Cleaned wide gtrends dataframe (outputs/clean_gtrends.csv).
            
        Returns:
            tuple: (df_ml_dataset, df_debug_analysis)
                - df_ml_dataset: Clean ML input dataset containing features + target ONLY.
                - df_debug_analysis: Debugging dataset retaining future metrics for audit.
        """
        date_col = 'date'
        trend_cols = [c for c in df_clean_gtrends.columns if c != date_col]

        # Convert date to datetime if string
        df_clean_gtrends = df_clean_gtrends.copy()
        df_clean_gtrends[date_col] = pd.to_datetime(df_clean_gtrends[date_col])
        df_features = df_features.copy()
        df_features[date_col] = pd.to_datetime(df_features[date_col])

        debug_records = []

        for trend in trend_cols:
            series_vals = df_clean_gtrends[trend].values
            series_dates = df_clean_gtrends[date_col]
            n = len(series_vals)

            # Process each time step t up to n - forecast_horizon
            for t in range(n - self.forecast_horizon):
                dt = series_dates.iloc[t]
                curr = float(series_vals[t])
                next_4 = series_vals[t + 1 : t + self.forecast_horizon + 1].astype(float)
                
                fut_val = float(next_4[-1])          # Value at t+4
                fut_max = float(np.max(next_4))      # Max value in t+1..t+4
                fut_min = float(np.min(next_4))      # Min value in t+1..t+4
                fut_mean = float(np.mean(next_4))    # Mean value in t+1..t+4

                # Safe baseline to prevent division by zero or near-zero exploding percentages
                base = max(curr, 5.0)

                # Future growth metrics (%)
                fut_growth = ((fut_val - curr) / base) * 100.0
                fut_max_growth = ((fut_max - curr) / base) * 100.0
                fut_mean_growth = ((fut_mean - curr) / base) * 100.0

                # Persistence & Sustained Growth Checks
                # Sustained: Average future level must be strictly higher than current
                is_sustained = (fut_mean > curr) and (fut_val >= curr)

                # Target Classification Logic (Data-Driven & Persistence-Aware):
                # -------------------------------------------------------------
                # 1. Strong Trend (Class 2):
                #    Requires high average future growth (>= 25%) OR high max growth (>= 35%) 
                #    WITH sustained trend progression (fut_val > curr and fut_mean > curr).
                if (fut_mean_growth >= 25.0 or (fut_max_growth >= 35.0 and fut_growth >= 15.0)) and is_sustained:
                    target = 2
                    target_label = 'Strong Trend'

                # 2. Emerging Trend (Class 1):
                #    Requires moderate positive growth (fut_mean_growth >= 5% or fut_max_growth >= 15%)
                #    WITH sustained trend progression (fut_val >= curr and fut_mean > curr).
                elif (fut_mean_growth >= 5.0 or fut_max_growth >= 15.0) and is_sustained:
                    target = 1
                    target_label = 'Emerging Trend'

                # 3. Stable (Class 0):
                #    Includes flat, negative growth, and temporary false spikes where volume drops back.
                else:
                    target = 0
                    target_label = 'Stable'

                debug_records.append({
                    'date': dt,
                    'trend_name': trend,
                    'current_val_t': curr,
                    'next_4_values': next_4.tolist(),
                    'future_val_t4': fut_val,
                    'future_max': fut_max,
                    'future_mean': fut_mean,
                    'future_growth_pct': fut_growth,
                    'future_max_growth_pct': fut_max_growth,
                    'future_mean_growth_pct': fut_mean_growth,
                    'is_sustained': is_sustained,
                    'target': target,
                    'target_label': target_label
                })

        df_debug = pd.DataFrame(debug_records)

        # Merge targets back into historical features dataframe
        # Inner join automatically drops invalid final 4 rows per trend
        df_ml = df_features.merge(
            df_debug[['date', 'trend_name', 'target', 'target_label']],
            on=['date', 'trend_name'],
            how='inner'
        )

        # Ensure correct column ordering
        feature_cols = [c for c in df_features.columns if c not in ['date', 'trend_name', 'trend_type']]
        ml_cols = ['date', 'trend_name', 'trend_type'] + feature_cols + ['target', 'target_label']
        df_ml = df_ml[ml_cols].sort_values(by=['trend_name', 'date']).reset_index(drop=True)

        return df_ml, df_debug

if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.preprocessing import DataPreprocessor
    from src.feature_engineering import FeatureEngineer

    loader = DataLoader()
    raw_gtrends = loader.load_gtrends()
    prep = DataPreprocessor(data_loader=loader)
    clean_gt = prep.clean_gtrends(raw_gtrends)
    trend_cols = [c for c in clean_gt.columns if c != 'date']
    meta_df = prep.generate_trend_metadata(trend_cols)

    fe = FeatureEngineer()
    feat_df = fe.generate_all_features(clean_gt, meta_df)

    tc = TargetCreator(forecast_horizon=4)
    df_ml, df_debug = tc.compute_targets(feat_df, clean_gt)

    print("ML Dataset Shape:", df_ml.shape)
    print("Class Value Counts:")
    print(df_ml['target_label'].value_counts())
    print("Class Percentages:")
    print(df_ml['target_label'].value_counts(normalize=True) * 100)
