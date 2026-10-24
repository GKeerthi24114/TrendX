import os
import joblib
import pandas as pd
import numpy as np

class WarningEngine:
    """
    Early Trend Detection Warning Engine.
    
    Converts ML model probability distributions and class predictions into
    actionable early warnings for supply chain, inventory, and design teams.
    
    Warning Statuses:
      - NORMAL (Green): Search volume is stable or flat. No immediate action required.
      - WATCH (Yellow): Emerging growth outbreak detected. Monitor inventory & suppliers.
      - ALERT (Red): Rapid surge / strong growth. Prioritize manufacturing & stock allocation.
    """
    STATUS_MAP = {
        0: {
            "status": "NORMAL",
            "color": "green",
            "message": "No significant emerging trend detected. Search volume remains within baseline levels."
        },
        1: {
            "status": "WATCH",
            "color": "yellow",
            "message": "An emerging trend has been detected. Continue monitoring inventory and supplier preparedness."
        },
        2: {
            "status": "ALERT",
            "color": "red",
            "message": "A strong trend surge has been detected. Increase manufacturing priority and stock allocation."
        }
    }

    LABEL_NAMES = ["Stable", "Emerging Trend", "Strong Trend"]

    def __init__(self, model_dir=None):
        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            local_models = os.path.join(base_dir, "models")
            if os.path.exists(local_models):
                model_dir = local_models
            elif os.path.exists(r"d:\DS PROJECT\models"):
                model_dir = r"d:\DS PROJECT\models"
            else:
                model_dir = local_models
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
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
        self.load_models()

    def load_models(self, model_name="random_forest"):
        """
        Loads saved model and scaler object.
        """
        clean_name = model_name.lower().replace(' ', '_')
        model_path = os.path.join(self.model_dir, f"{clean_name}.joblib")
        scaler_path = os.path.join(self.model_dir, "scaler.joblib")

        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

    def predict_trend(self, feature_data):
        """
        Generates prediction, probabilities, and warning alert for given input feature row or DataFrame.
        
        Args:
            feature_data (dict, pd.Series, or pd.DataFrame): Input feature values.
            
        Returns:
            dict: Structured prediction output dictionary.
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Please train or load a saved model first.")

        if isinstance(feature_data, dict):
            df_input = pd.DataFrame([feature_data])
        elif isinstance(feature_data, pd.Series):
            df_input = pd.DataFrame([feature_data.to_dict()])
        elif isinstance(feature_data, pd.DataFrame):
            df_input = feature_data.copy()
        else:
            raise TypeError("feature_data must be dict, pd.Series, or pd.DataFrame")

        # Ensure all feature columns are present
        X_in = df_input[self.feature_cols].copy()

        # Apply scaling if using logistic regression and scaler is loaded
        is_lr = hasattr(self.model, "coef_") and not hasattr(self.model, "estimators_")
        if is_lr and self.scaler is not None:
            X_proc = self.scaler.transform(X_in)
        else:
            X_proc = X_in

        # Get probabilities and predictions
        probs = self.model.predict_proba(X_proc)[0]
        pred_class = int(np.argmax(probs))
        pred_label = self.LABEL_NAMES[pred_class]
        confidence = float(probs[pred_class])

        prob_dict = {
            "Stable": float(probs[0]),
            "Emerging Trend": float(probs[1]),
            "Strong Trend": float(probs[2])
        }

        status_info = self.STATUS_MAP.get(pred_class, self.STATUS_MAP[0])

        return {
            "prediction": pred_class,
            "label": pred_label,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
            "warning_status": status_info["status"],
            "warning_color": status_info["color"],
            "warning_message": status_info["message"]
        }

if __name__ == "__main__":
    engine = WarningEngine()
    sample_feat = {
        'current_value': 75.0,
        'previous_value': 65.0,
        'absolute_change': 10.0,
        'percentage_change': 15.38,
        'moving_average_4': 68.5,
        'rolling_std_4': 5.2,
        'moving_average_8': 62.0,
        'trend_slope': 3.5,
        'momentum': 5.0,
        'volatility': 0.075,
        'persistence': 3
    }
    if engine.model is not None:
        res = engine.predict_trend(sample_feat)
        print("Sample Prediction Result:")
        print(res)
    else:
        print("WarningEngine initialized. (Model file not found yet - run training script first).")
