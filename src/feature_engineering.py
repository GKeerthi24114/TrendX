import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Time-Series Feature Engineering for Early Trend Detection.
    
    STRICT ANTI-LEAKAGE DESIGN:
    At time step 't', every single feature is computed strictly using historical 
    observations available at or before time 't' (t' <= t). 
    Future information (t+1, t+2, ...) is NEVER used during feature generation.
    """
    def __init__(self, window_size_short=4, window_size_long=8):
        self.window_size_short = window_size_short
        self.window_size_long = window_size_long
        self.feature_names = [
            'current_value',
            'previous_value',
            'absolute_change',
            'percentage_change',
            'moving_average_4',
            'rolling_std_4',
            'moving_average_8',
            'trend_slope',
            'momentum',
            'volatility',
            'persistence'
        ]

    def _calc_slope(self, y_window):
        """
        Calculates ordinary least squares (OLS) linear regression slope 
        over a historical time window y_window.
        slope > 0 indicates an increasing trend.
        slope < 0 indicates a decreasing trend.
        slope ~ 0 indicates a stable trend.
        """
        n = len(y_window)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=float)
        x_mean = np.mean(x)
        y_mean = np.mean(y_window)
        denom = np.sum((x - x_mean) ** 2)
        if denom == 0:
            return 0.0
        numer = np.sum((x - x_mean) * (y_window - y_mean))
        return float(numer / denom)

    def _calc_persistence(self, vals, t):
        """
        Calculates persistence as the number of consecutive weekly periods 
        prior to and including time 't' where search volume strictly increased (y_k > y_k-1).
        """
        count = 0
        idx = t
        while idx > 0 and vals[idx] > vals[idx - 1]:
            count += 1
            idx -= 1
        return count

    def extract_features_for_series(self, series):
        """
        Extracts 11 historical time-series features for a single trend variable.
        Uses expanding window for initial periods (t < 4 or t < 8) to avoid 
        missing values without introducing future data or zero-filling artifacts.
        
        Returns:
            pd.DataFrame: DataFrame containing feature columns for the series.
        """
        vals = series.values.astype(float)
        n = len(vals)
        features_list = []

        for t in range(n):
            curr = vals[t]
            prev = vals[t - 1] if t > 0 else curr
            prev2 = vals[t - 2] if t > 1 else prev

            # 1. Current value (y_t)
            current_value = curr

            # 2. Previous value (y_t-1)
            previous_value = prev

            # 3. Absolute change (y_t - y_t-1)
            abs_change = curr - prev

            # 4. Percentage change (((y_t - y_t-1) / y_t-1) * 100)
            if prev == 0.0 and curr == 0.0:
                pct_change = 0.0
            else:
                pct_change = float(((curr - prev) / (prev + 1e-5)) * 100.0)

            # Historical windows up to time t (No future observations used)
            w4 = vals[max(0, t - self.window_size_short + 1) : t + 1]
            w8 = vals[max(0, t - self.window_size_long + 1) : t + 1]

            # 5. 4-period moving average
            moving_average_4 = float(np.mean(w4))

            # 6. 4-period rolling standard deviation
            rolling_std_4 = float(np.std(w4, ddof=1)) if len(w4) > 1 else 0.0

            # 7. 8-period moving average
            moving_average_8 = float(np.mean(w8))

            # 8. Trend slope over 4-period window
            trend_slope = self._calc_slope(w4)

            # 9. Momentum (discrete acceleration: (y_t - y_t-1) - (y_t-1 - y_t-2))
            momentum = float((curr - prev) - (prev - prev2)) if t >= 2 else 0.0

            # 10. Volatility (coefficient of variation: std / mean)
            volatility = float(rolling_std_4 / (moving_average_4 + 1e-5))

            # 11. Persistence (consecutive positive growth periods)
            persistence = int(self._calc_persistence(vals, t))

            features_list.append({
                'current_value': current_value,
                'previous_value': previous_value,
                'absolute_change': abs_change,
                'percentage_change': pct_change,
                'moving_average_4': moving_average_4,
                'rolling_std_4': rolling_std_4,
                'moving_average_8': moving_average_8,
                'trend_slope': trend_slope,
                'momentum': momentum,
                'volatility': volatility,
                'persistence': persistence
            })

        return pd.DataFrame(features_list)

    def generate_all_features(self, df_gtrends, metadata_df):
        """
        Generates feature dataset for all trend variables in wide gtrends DataFrame.
        Each row in output represents (date, trend_name, trend_type, features...).
        
        Returns:
            pd.DataFrame: Final feature dataset.
        """
        date_col = 'date'
        dates = df_gtrends[date_col]
        trend_cols = [c for c in df_gtrends.columns if c != date_col]
        
        # Create lookup dictionary for trend_type
        type_map = dict(zip(metadata_df['trend_name'], metadata_df['trend_type']))

        records = []
        for var_name in trend_cols:
            series = df_gtrends[var_name]
            feat_df = self.extract_features_for_series(series)
            feat_df['date'] = dates
            feat_df['trend_name'] = var_name
            feat_df['trend_type'] = type_map.get(var_name, 'other')
            records.append(feat_df)

        full_df = pd.concat(records, ignore_index=True)

        # Reorder columns as requested:
        # date | trend_name | trend_type | current_value | previous_value | ...
        cols = ['date', 'trend_name', 'trend_type'] + self.feature_names
        full_df = full_df[cols]

        # Sort chronologically and by trend name
        full_df = full_df.sort_values(by=['trend_name', 'date']).reset_index(drop=True)

        return full_df

if __name__ == "__main__":
    from src.preprocessing import DataPreprocessor
    prep = DataPreprocessor()
    df_clean = prep.clean_gtrends()
    trend_cols = [c for c in df_clean.columns if c != 'date']
    meta_df = prep.generate_trend_metadata(trend_cols)
    
    fe = FeatureEngineer()
    features_df = fe.generate_all_features(df_clean, meta_df)
    print("Features DataFrame shape:", features_df.shape)
    print("Columns:", features_df.columns.tolist())
    print("Missing values count:", features_df.isnull().sum().sum())
    print("Sample head:")
    print(features_df.head(5))
