import os
import numpy as np
import pandas as pd

def generate_sample_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 220 weeks from 2015-10-05 to 2019-12-16
    dates = pd.date_range(start='2015-10-05', end='2019-12-16', freq='W-MON')
    if len(dates) > 220:
        dates = dates[:220]
    elif len(dates) < 220:
        dates = pd.date_range(start='2015-10-05', periods=220, freq='W-MON')

    categories = [
        'miniskirt', 'culottes', 'long sleeve', 'tracksuit', 'maxi', 'sheath dress', 
        'trapeze dress', 'blazer', 'cardigan', 'hoodie', 'jumpsuit', 'kimono', 
        'leggings', 'parka', 'peplum', 'poncho', 'romper', 'sarong', 'skater dress', 
        'slip dress', 'sweater', 't-shirt', 'trench coat', 'tunic', 'vest', 
        'wrap dress', 'cargo pants', 'bomber jacket', 'crop top', 'flared pants', 
        'pencil skirt', 'corset'
    ]
    
    colors = [
        'yellow', 'brown', 'blue', 'grey', 'green', 'black', 'red', 'white', 'orange', 'violet'
    ]
    
    fabrics = [
        'acrylic', 'scuba crepe', 'tulle', 'angora', 'faux leather', 'georgette', 
        'velvet', 'lace', 'cotton', 'denim', 'linen', 'silk', 'wool', 'cashmere', 
        'chiffon', 'satin', 'spandex', 'polyester', 'viscose', 'corduroy', 'tweed', 
        'fleece', 'flannel', 'organza', 'crepe', 'jacquard', 'jersey', 'modal', 
        'nylon', 'rayon', 'suede', 'taffeta', 'canvas', 'chambray', 'damask', 
        'eyelet', 'gabardine', 'gingham', 'houndstooth', 'khaki', 'lamé', 'madras', 
        'mesh', 'mohair', 'muslin', 'oxford', 'plaid', 'poplin', 'ripstop', 
        'seersucker', 'terry', 'twill', 'velour', 'voile'
    ]
    
    all_trends = categories + colors + fabrics
    
    np.random.seed(42)
    n_steps = len(dates)
    data = {'date': dates}
    
    for trend in all_trends:
        base_level = np.random.uniform(20, 60)
        trend_slope = np.random.uniform(-0.05, 0.15)
        seasonal_freq = np.random.choice([26, 52])
        seasonal = 15 * np.sin(2 * np.pi * np.arange(n_steps) / seasonal_freq)
        noise = np.random.normal(0, 5, n_steps)
        
        # Surge effect for some trends towards the end (to simulate emerging/strong trends)
        if trend in ['miniskirt', 'velvet', 'yellow', 'crop top', 'denim']:
            surge = np.linspace(0, 45, n_steps) ** 1.3 / 10
        else:
            surge = 0
            
        series = np.clip(base_level + trend_slope * np.arange(n_steps) + seasonal + noise + surge, 0, 100)
        data[trend] = np.round(series, 1)

    df_gtrends = pd.DataFrame(data)
    gtrends_path = os.path.join(dataset_dir, "gtrends.csv")
    df_gtrends.to_csv(gtrends_path, index=False)
    print(f"Generated sample gtrends.csv: {gtrends_path} with shape {df_gtrends.shape}")

    # Generate sample train.csv and test.csv catalog data
    catalog_items = []
    for i in range(500):
        catalog_items.append({
            'item_id': i,
            'category': np.random.choice(categories),
            'color': np.random.choice(colors),
            'fabric': np.random.choice(fabrics),
            'season': np.random.choice(['Spring', 'Summer', 'Autumn', 'Winter']),
            'release_date': str(pd.to_datetime(np.random.choice(dates)).date())
        })
    df_train = pd.DataFrame(catalog_items)
    train_path = os.path.join(dataset_dir, "train.csv")
    df_train.to_csv(train_path, index=False)
    
    # Generate PyTorch label dicts
    cat_dict = {cat: idx for idx, cat in enumerate(categories)}
    col_dict = {col: idx for idx, col in enumerate(colors)}
    fab_dict = {fab: idx for idx, fab in enumerate(fabrics)}
    
    import torch
    torch.save(cat_dict, os.path.join(dataset_dir, "category_labels.pt"))
    torch.save(col_dict, os.path.join(dataset_dir, "color_labels.pt"))
    torch.save(fab_dict, os.path.join(dataset_dir, "fabric_labels.pt"))
    np.save(os.path.join(dataset_dir, "normalization_scale.npy"), np.array(1065.0))
    print("Generated label mappings and normalization scale")

if __name__ == "__main__":
    generate_sample_data()
