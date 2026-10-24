import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from src.train_models import ModelTrainer, ModelPipeline

class ModelEvaluator:
    """
    Evaluates machine learning models on time-series test dataset.
    Generates performance comparison metrics, confusion matrices,
    and feature importance analysis.
    """
    def __init__(self, models_dir=None, outputs_dir=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = models_dir if models_dir is not None else os.path.join(base_dir, "models")
        self.outputs_dir = outputs_dir if outputs_dir is not None else os.path.join(base_dir, "outputs")
        os.makedirs(self.outputs_dir, exist_ok=True)
        self.target_names = ['Stable', 'Emerging Trend', 'Strong Trend']

    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Evaluates a single model on test set.
        """
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Emerging trend (Class 1) specific metrics
        prec_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
        rec_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

        em_prec = float(prec_per_class[1]) if len(prec_per_class) > 1 else 0.0
        em_rec = float(rec_per_class[1]) if len(rec_per_class) > 1 else 0.0
        em_f1 = float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0

        roc_auc = None
        if y_prob is not None:
            try:
                roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
            except Exception:
                roc_auc = np.nan

        cm = confusion_matrix(y_test, y_pred)

        metrics = {
            'Model': model_name,
            'Accuracy': acc,
            'Precision (Macro)': prec_macro,
            'Precision (Weighted)': prec_weighted,
            'Recall (Macro)': rec_macro,
            'Recall (Weighted)': rec_weighted,
            'F1-Score (Macro)': f1_macro,
            'F1-Score (Weighted)': f1_weighted,
            'Emerging Precision': em_prec,
            'Emerging Recall': em_rec,
            'Emerging F1-Score': em_f1,
            'ROC-AUC (Macro OvR)': roc_auc
        }

        report = classification_report(
            y_test, y_pred, target_names=self.target_names, zero_division=0, output_dict=True
        )

        return metrics, cm, report

    def plot_and_save_confusion_matrix(self, cm, model_name):
        """
        Saves confusion matrix heatmap plot to outputs directory.
        """
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.target_names,
            yticklabels=self.target_names
        )
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted Class")
        plt.ylabel("Actual Class")
        plt.tight_layout()
        filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        filepath = os.path.join(self.outputs_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        return filepath

    def get_feature_importances(self, model, feature_names, model_name):
        """
        Extracts feature importances for tree-based models or coefficients for linear models.
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.mean(np.abs(model.coef_), axis=0)
        else:
            return None

        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).reset_index(drop=True)

        fi_df['Model'] = model_name
        return fi_df

    def plot_and_save_feature_importance(self, fi_df, model_name):
        """
        Plots and saves feature importance bar chart.
        """
        plt.figure(figsize=(8, 5))
        sns.barplot(data=fi_df, x='Importance', y='Feature', palette='viridis')
        plt.title(f"Feature Importance - {model_name}")
        plt.tight_layout()
        filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
        filepath = os.path.join(self.outputs_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        return filepath

    def run_full_evaluation(self, trainer_data, models_dict):
        """
        Runs evaluation across all models, saves comparison table,
        confusion matrices, feature importances.
        """
        all_metrics = []
        all_fi = []

        feature_cols = trainer_data['X_test'].columns.tolist()

        for name, (model, is_scaled) in models_dict.items():
            X_test_input = trainer_data['X_test_scaled'] if is_scaled else trainer_data['X_test']
            y_test = trainer_data['y_test']

            metrics, cm, report = self.evaluate_model(model, X_test_input, y_test, name)
            all_metrics.append(metrics)

            # Save confusion matrix plot
            self.plot_and_save_confusion_matrix(cm, name)

            # Feature importance
            fi_df = self.get_feature_importances(model, feature_cols, name)
            if fi_df is not None:
                all_fi.append(fi_df)
                self.plot_and_save_feature_importance(fi_df, name)

        # Summary Metrics DataFrame
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(os.path.join(self.outputs_dir, "model_comparison_metrics.csv"), index=False)

        # Feature Importance DataFrame
        if all_fi:
            combined_fi = pd.concat(all_fi, ignore_index=True)
            combined_fi.to_csv(os.path.join(self.outputs_dir, "feature_importances.csv"), index=False)

        return metrics_df, combined_fi if all_fi else None

if __name__ == "__main__":
    trainer = ModelTrainer()
    data_dict = trainer.prepare_data()
    models = trainer.train_all_models(data_dict)

    evaluator = ModelEvaluator()
    metrics_df, fi_df = evaluator.run_full_evaluation(data_dict, models)

    print("\n" + "="*60)
    print("MODEL EVALUATION COMPARISON SUMMARY (TEST SET)")
    print("="*60)
    print(metrics_df.to_string(index=False))
