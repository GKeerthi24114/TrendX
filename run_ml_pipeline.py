import os
import pandas as pd
import numpy as np
from src.train_models import ModelPipeline
from src.warning_engine import WarningEngine

def main():
    print("=" * 75)
    print("RUNNING MACHINE LEARNING TRAINING & EVALUATION PIPELINE")
    print("=" * 75)

    # 1. Train models and generate evaluation metrics
    pipeline = ModelPipeline()
    pipeline.run()

    # 2. Test Warning Engine & Reusable Inference
    print("\n[STEP 2] Testing Warning Engine & Reusable Inference Function...")
    engine = WarningEngine()
    engine.load_models(model_name="random_forest")

    # 3. Manually inspect at least 5 predictions from holdout test set
    df_test = pipeline.df_test.copy().reset_index(drop=True)
    sample_indices = [50, 500, 1200, 2000, 2800]

    print("\n" + "=" * 75)
    print("MANUAL VALIDATION OF 5 HOLDOUT TEST SET PREDICTIONS")
    print("=" * 75)

    for idx in sample_indices:
        row = df_test.iloc[idx]
        trend_name = row['trend_name']
        dt_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
        actual_label = row['target_label']

        feat_dict = row[engine.feature_cols].to_dict()
        pred_res = engine.predict_trend(feat_dict)

        print(f"Date               : {dt_str}")
        print(f"Trend Name         : '{trend_name}' ({row['trend_type']})")
        print(f"Actual Label       : {actual_label}")
        print(f"Predicted Label    : {pred_res['label']} (Class {pred_res['prediction']})")
        print(f"Confidence         : {pred_res['confidence'] * 100:.2f}%")
        print(f"Class Probabilities: {pred_res['probabilities']}")
        print(f"Warning Status     : {pred_res['warning_status']}")
        print(f"Warning Message    : {pred_res['warning_message']}")
        print("-" * 60)

    print("\n" + "=" * 75)
    print("ML PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    main()
