import pandas as pd
import numpy as np
from src.data_loader import DataLoader

class ProductAnalyzer:
    """
    Supporting Product Analysis Layer.
    Connects detected trends in gtrends with product catalog records from train.csv.
    Uses normalization scale to un-scale numerical embeddings/features.
    """
    def __init__(self, data_loader=None):
        if data_loader is None:
            data_loader = DataLoader()
        self.data_loader = data_loader
        self.train_df, self.test_df = data_loader.load_fashion_products()
        self.labels = data_loader.load_labels()
        self.scale = data_loader.load_scale()

        self.cat_map = {v: k for k, v in self.labels.get('category', {}).items()}
        self.col_map = {v: k for k, v in self.labels.get('color', {}).items()}
        self.fab_map = {v: k for k, v in self.labels.get('fabric', {}).items()}

    def get_unscaled_features(self, df):
        """
        Unscales numerical continuous feature columns '0' to '11' using normalization scale.
        """
        num_cols = [str(i) for i in range(12) if str(i) in df.columns]
        df_unscaled = df.copy()
        for col in num_cols:
            df_unscaled[col] = df_unscaled[col] * self.scale
        return df_unscaled

    def find_matching_products(self, trend_variable, dataset='train'):
        """
        Finds matching catalog items in train.csv for a given trend variable name
        (whether category, color, or fabric).
        """
        df = self.train_df if dataset == 'train' else self.test_df

        # Match against category, color, or fabric
        matched = df[
            (df['category'].astype(str).str.lower() == trend_variable.lower()) |
            (df['color'].astype(str).str.lower() == trend_variable.lower()) |
            (df['fabric'].astype(str).str.lower() == trend_variable.lower())
        ]
        return matched

    def get_product_attribute_distribution(self, trend_variable):
        """
        Given a trend variable, calculates distributions of related categories, colors, and fabrics.
        """
        matched = self.find_matching_products(trend_variable, dataset='train')
        if len(matched) == 0:
            return {
                'match_count': 0,
                'category_dist': {},
                'color_dist': {},
                'fabric_dist': {}
            }

        return {
            'match_count': len(matched),
            'category_dist': matched['category'].value_counts().head(5).to_dict(),
            'color_dist': matched['color'].value_counts().head(5).to_dict(),
            'fabric_dist': matched['fabric'].value_counts().head(5).to_dict()
        }

if __name__ == "__main__":
    analyzer = ProductAnalyzer()
    res = analyzer.get_product_attribute_distribution("miniskirt")
    print("Matching products for 'miniskirt':", res['match_count'])
    print("Top Colors:", res['color_dist'])
    print("Top Fabrics:", res['fabric_dist'])
