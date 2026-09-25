import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

try:
    import xgboost as xgb
except ImportError:
    xgb = None

class ModelPipeline:
    """
    Time-Aware Machine Learning Training & Evaluation Pipeline.
    
    Models:
      1. Logistic Regression (Scaled, Balanced Class Weights)
      2. Random Forest Classifier (Balanced Class Weights)
      3. XGBoost Classifier (Sample Weighted)
    """
    def __init__(self, data_path=r"d:\DS PROJECT\outputs\ml_dataset.csv", output_dir=r"d:\DS PROJECT\outputs"):
        self.data_path = data_path
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "plots", "model_evaluation")
        self.models_dir = r"d:\DS PROJECT\models"
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        self.feature_cols = [
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
        self.target_col = 'target'
        self.class_names = ['Stable', 'Emerging Trend', 'Strong Trend']

    def load_and_split_data(self):
        """
        Loads ml_dataset.csv and performs chronological time-aware split:
          - 70% earliest dates -> Train Set
          - 15% following dates -> Validation Set
          - 15% latest dates   -> Holdout Test Set
        """
        df = pd.read_csv(self.data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort chronologically
        df = df.sort_values(by=['date', 'trend_name']).reset_index(drop=True)

        unique_dates = np.sort(df['date'].unique())
        n_dates = len(unique_dates)

        train_end_idx = int(0.70 * n_dates)
        val_end_idx = int(0.85 * n_dates)

        train_dates = unique_dates[:train_end_idx]
        val_dates = unique_dates[train_end_idx:val_end_idx]
        test_dates = unique_dates[val_end_idx:]

        train_mask = df['date'].isin(train_dates)
        val_mask = df['date'].isin(val_dates)
        test_mask = df['date'].isin(test_dates)

        self.df_train = df[train_mask].copy()
        self.df_val = df[val_mask].copy()
        self.df_test = df[test_mask].copy()

        self.X_train = self.df_train[self.feature_cols]
        self.y_train = self.df_train[self.target_col].values

        self.X_val = self.df_val[self.feature_cols]
        self.y_val = self.df_val[self.target_col].values

        self.X_test = self.df_test[self.feature_cols]
        self.y_test = self.df_test[self.target_col].values

        # Scale features strictly on train set for Logistic Regression
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # Save scaler object
        joblib.dump(self.scaler, os.path.join(self.models_dir, "scaler.joblib"))

        print("Data Split Summary:")
        print(f" -> Train Dates: {pd.to_datetime(train_dates[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(train_dates[-1]).strftime('%Y-%m-%d')} ({len(self.df_train)} rows)")
        print(f" -> Val Dates  : {pd.to_datetime(val_dates[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(val_dates[-1]).strftime('%Y-%m-%d')} ({len(self.df_val)} rows)")
        print(f" -> Test Dates : {pd.to_datetime(test_dates[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(test_dates[-1]).strftime('%Y-%m-%d')} ({len(self.df_test)} rows)")

    def train_models(self):
        """
        Trains Logistic Regression, Random Forest, and XGBoost models.
        """
        self.models = {}

        # 1. Logistic Regression
        print("\n[TRAINING 1/3] Logistic Regression...")
        lr = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        lr.fit(self.X_train_scaled, self.y_train)
        self.models['Logistic Regression'] = lr
        joblib.dump(lr, os.path.join(self.models_dir, "logistic_regression.joblib"))

        # 2. Random Forest
        print("[TRAINING 2/3] Random Forest Classifier...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        joblib.dump(rf, os.path.join(self.models_dir, "random_forest.joblib"))

        # 3. XGBoost Classifier
        if xgb is not None:
            print("[TRAINING 3/3] XGBoost Classifier...")
            sample_weights = compute_sample_weight('balanced', self.y_train)
            xgb_clf = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                eval_metric='mlogloss',
                random_state=42,
                n_jobs=-1
            )
            xgb_clf.fit(self.X_train, self.y_train, sample_weight=sample_weights)
            self.models['XGBoost'] = xgb_clf
            joblib.dump(xgb_clf, os.path.join(self.models_dir, "xgboost.joblib"))

    def evaluate_all(self):
        """
        Evaluates all trained models on validation and holdout test sets.
        Computes Accuracy, Precision, Recall, F1-Score, and Multiclass ROC-AUC.
        """
        results = []
        feature_importance_dict = {'feature': self.feature_cols}

        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

        for name, model in self.models.items():
            is_lr = (name == 'Logistic Regression')
            X_test_proc = self.X_test_scaled if is_lr else self.X_test

            y_pred = model.predict(X_test_proc)
            y_prob = model.predict_proba(X_test_proc)

            acc = accuracy_score(self.y_test, y_pred)
            prec_macro = precision_score(self.y_test, y_pred, average='macro', zero_division=0)
            rec_macro = recall_score(self.y_test, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(self.y_test, y_pred, average='macro', zero_division=0)

            prec_weighted = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            rec_weighted = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1_weighted = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)

            try:
                roc_auc = roc_auc_score(self.y_test, y_prob, multi_class='ovr', average='macro')
            except Exception:
                roc_auc = np.nan

            results.append({
                'model': name,
                'accuracy': round(acc, 4),
                'precision_macro': round(prec_macro, 4),
                'recall_macro': round(rec_macro, 4),
                'f1_macro': round(f1_macro, 4),
                'precision_weighted': round(prec_weighted, 4),
                'recall_weighted': round(rec_weighted, 4),
                'f1_weighted': round(f1_weighted, 4),
                'roc_auc': round(roc_auc, 4) if not np.isnan(roc_auc) else "N/A"
            })

            # Feature Importance / Coefficient Extraction
            if is_lr:
                # Mean absolute coefficient magnitude across classes
                importances = np.mean(np.abs(model.coef_), axis=0)
            else:
                importances = model.feature_importances_

            # Normalize importances so sum = 1
            importances_norm = importances / np.sum(importances)
            feature_importance_dict[name] = importances_norm

            # Plot Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
            ax.set_title(f"Confusion Matrix: {name}", fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            plt.tight_layout()
            cm_filename = name.lower().replace(' ', '_')
            plt.savefig(os.path.join(self.plots_dir, f"cm_{cm_filename}.png"), dpi=200)
            plt.close()

            # Plot Feature Importance Bar Chart
            fi_df = pd.DataFrame({'feature': self.feature_cols, 'importance': importances_norm}).sort_values('importance', ascending=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(fi_df['feature'], fi_df['importance'], color='#1f77b4')
            ax.set_title(f"Feature Importance: {name}", fontsize=12, fontweight='bold')
            ax.set_xlabel('Normalized Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, f"feature_importance_{cm_filename}.png"), dpi=200)
            plt.close()

        # Save Model Comparison CSV
        self.metrics_df = pd.DataFrame(results)
        comp_path = os.path.join(self.output_dir, "model_comparison.csv")
        self.metrics_df.to_csv(comp_path, index=False)
        print(f"\n -> Saved model comparison table: {comp_path}")

        # Save Feature Importance CSV
        self.fi_df = pd.DataFrame(feature_importance_dict)
        numeric_cols = [c for c in self.fi_df.columns if c != 'feature']
        self.fi_df['mean_importance'] = self.fi_df[numeric_cols].mean(axis=1)
        self.fi_df = self.fi_df.sort_values(by='mean_importance', ascending=False).reset_index(drop=True)
        fi_path = os.path.join(self.output_dir, "feature_importance.csv")
        self.fi_df.to_csv(fi_path, index=False)
        print(f" -> Saved feature importance table: {fi_path}")

        # Model Comparison Visual Chart
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_df = self.metrics_df.set_index('model')[['accuracy', 'f1_macro', 'f1_weighted', 'roc_auc']]
        plot_df.plot(kind='bar', ax=ax, colormap='viridis')
        ax.set_title('Model Performance Comparison (Holdout Test Set)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.05)
        plt.xticks(rotation=0)
        ax.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "model_comparison_chart.png"), dpi=200)
        plt.close()

    def run(self):
        self.load_and_split_data()
        self.train_models()
        self.evaluate_all()

if __name__ == "__main__":
    pipeline = ModelPipeline()
    pipeline.run()
