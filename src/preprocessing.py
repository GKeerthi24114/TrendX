import pandas as pd
import numpy as np
import os
from src.data_loader import DataLoader

class DataPreprocessor:
    """
    Preprocessing module for Google Trends time-series data.
    - Cleans invalid search metrics (-1.0 values -> 0.0)
    - Formats and verifies date continuity
    - Maps trend variables into Category, Color, Fabric, or Other groups
    - Converts wide time-series into clean long-format structure
    """
    def __init__(self, data_loader=None):
        if data_loader is None:
            data_loader = DataLoader()
        self.data_loader = data_loader
        self.labels = data_loader.load_labels()

    def clean_gtrends(self, df=None):
        """
        Cleans raw gtrends DataFrame.
        - Converts date column to proper pandas datetime.
        - Sorts data chronologically.
        - Handles duplicate dates if any exist.
        - Replaces negative search index metrics (-1.0) with 0.0.
        - Converts numerical trend columns to float64.
        
        Returns:
            pd.DataFrame: Cleaned wide-format DataFrame.
        """
        if df is None:
            df = self.data_loader.load_gtrends()

        df_clean = df.copy()

        # 1. Convert date to proper datetime format
        if 'date' in df_clean.columns:
            df_clean['date'] = pd.to_datetime(df_clean['date'])
        
        # 2. Sort chronologically
        df_clean = df_clean.sort_values('date').reset_index(drop=True)

        # 3. Handle duplicate dates
        dup_count = df_clean['date'].duplicated().sum()
        if dup_count > 0:
            df_clean = df_clean.drop_duplicates(subset=['date'], keep='first').reset_index(drop=True)

        # 4. Process numerical trend columns
        trend_cols = [c for c in df_clean.columns if c != 'date']
        for col in trend_cols:
            # Convert to float numeric type
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0)
            # Replace -1.0 (unindexed search index indicator) with 0.0
            df_clean[col] = df_clean[col].apply(lambda x: 0.0 if x < 0 else float(x))

        return df_clean

    def generate_trend_metadata(self, trend_columns):
        """
        Maps trend column names to their attribute type: 'category', 'color', 'fabric', or 'other'.
        
        Returns:
            pd.DataFrame: Metadata with columns ['trend_name', 'trend_type'].
        """
        cat_set = set(self.labels.get('category', {}).keys())
        col_set = set(self.labels.get('color', {}).keys())
        fab_set = set(self.labels.get('fabric', {}).keys())

        records = []
        for col in trend_columns:
            if col in cat_set:
                t_type = 'category'
            elif col in col_set:
                t_type = 'color'
            elif col in fab_set:
                t_type = 'fabric'
            else:
                t_type = 'other'
            records.append({'trend_name': col, 'trend_type': t_type})

        metadata_df = pd.DataFrame(records)
        return metadata_df

    def transform_to_long_format(self, df_clean, metadata_df):
        """
        Converts wide-format gtrends DataFrame into long-format time-series DataFrame.
        Wide format: date | col1 | col2 | ...
        Long format: date | trend_name | trend_type | value
        
        Returns:
            pd.DataFrame: Long-format time-series DataFrame.
        """
        date_col = 'date'
        trend_cols = [c for c in df_clean.columns if c != date_col]

        df_long = df_clean.melt(
            id_vars=[date_col],
            value_vars=trend_cols,
            var_name='trend_name',
            value_name='value'
        )

        # Merge with metadata to get trend_type
        df_long = df_long.merge(metadata_df, on='trend_name', how='left')

        # Reorder columns to date | trend_name | trend_type | value
        df_long = df_long[['date', 'trend_name', 'trend_type', 'value']]
        df_long = df_long.sort_values(by=['trend_name', 'date']).reset_index(drop=True)

        return df_long

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_gtrends()
    trend_cols = [c for c in df_clean.columns if c != 'date']
    meta_df = preprocessor.generate_trend_metadata(trend_cols)
    df_long = preprocessor.transform_to_long_format(df_clean, meta_df)
    print("Cleaned gtrends shape:", df_clean.shape)
    print("Metadata shape:", meta_df.shape)
    print("Long format shape:", df_long.shape)
    print("Sample long format head:")
    print(df_long.head(5))
