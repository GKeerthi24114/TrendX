import os
import pandas as pd
import numpy as np

try:
    import torch
except ImportError:
    torch = None

class DataLoader:
    """
    Data loader utility to read and structure Google Trends data,
    fashion product datasets, label maps, and normalization constants.
    """
    def __init__(self, data_dir=r"d:\DS PROJECT\dataset"):
        self.data_dir = data_dir
        self.gtrends_path = os.path.join(data_dir, "gtrends.csv")
        self.train_path = os.path.join(data_dir, "train.csv")
        self.test_path = os.path.join(data_dir, "test.csv")
        self.category_labels_path = os.path.join(data_dir, "category_labels.pt")
        self.color_labels_path = os.path.join(data_dir, "color_labels.pt")
        self.fabric_labels_path = os.path.join(data_dir, "fabric_labels.pt")
        self.scale_path = os.path.join(data_dir, "normalization_scale.npy")

    def load_gtrends(self):
        """
        Loads gtrends.csv, formats date column, sorts by date.
        Returns:
            pd.DataFrame: Time-series Google Trends dataframe.
        """
        df = pd.read_csv(self.gtrends_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        return df

    def load_fashion_products(self):
        """
        Loads train.csv and test.csv fashion datasets.
        Returns:
            tuple: (train_df, test_df)
        """
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        return train_df, test_df

    def load_labels(self):
        """
        Loads label mapping dictionaries from PyTorch .pt files.
        Returns:
            dict: Dictionary with keys 'category', 'color', 'fabric'
        """
        labels = {}
        for key, path in [
            ('category', self.category_labels_path),
            ('color', self.color_labels_path),
            ('fabric', self.fabric_labels_path)
        ]:
            if os.path.exists(path):
                loaded = False
                if torch is not None:
                    try:
                        labels[key] = torch.load(path, weights_only=False)
                        loaded = True
                    except Exception:
                        pass
                if not loaded:
                    try:
                        import zipfile, pickle
                        with zipfile.ZipFile(path, 'r') as z:
                            for member in z.namelist():
                                if member.endswith('data.pkl'):
                                    labels[key] = pickle.loads(z.read(member))
                                    loaded = True
                    except Exception:
                        labels[key] = {}
            else:
                labels[key] = {}
        return labels

    def load_scale(self):
        """
        Loads normalization scale value.
        Returns:
            float: Normalization multiplier.
        """
        if os.path.exists(self.scale_path):
            scale = np.load(self.scale_path, allow_pickle=True)
            return float(scale)
        return 1.0

if __name__ == "__main__":
    loader = DataLoader()
    gt = loader.load_gtrends()
    train, test = loader.load_fashion_products()
    labels = loader.load_labels()
    scale = loader.load_scale()
    print(f"Loaded gtrends shape: {gt.shape}")
    print(f"Loaded train shape: {train.shape}, test shape: {test.shape}")
    print(f"Categories count: {len(labels.get('category', {}))}")
    print(f"Colors count: {len(labels.get('color', {}))}")
    print(f"Fabrics count: {len(labels.get('fabric', {}))}")
    print(f"Normalization scale: {scale}")
