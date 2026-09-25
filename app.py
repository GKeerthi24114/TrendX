import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.warning_engine import WarningEngine
from src.product_analysis import ProductAnalyzer

# Set page configuration
st.set_page_config(
    page_title="Early Trend Detection and Warning System",
    page_icon="📈",
    layout="wide"
)

# Page Title and Subtitle
st.title("Early Trend Detection and Warning System")
st.markdown("Data-driven detection of emerging fashion trends using Google Trends and Machine Learning.")

# ---------------------------------------------------------
# CACHED DATA & MODEL LOADERS
# ---------------------------------------------------------
@st.cache_data
def load_datasets():
    output_dir = "outputs"
    clean_gt_path = os.path.join(output_dir, "clean_gtrends.csv")
    meta_path = os.path.join(output_dir, "trend_metadata.csv")
    feat_path = os.path.join(output_dir, "trend_features.csv")

    if os.path.exists(clean_gt_path) and os.path.exists(meta_path) and os.path.exists(feat_path):
        df_clean = pd.read_csv(clean_gt_path)
        df_clean['date'] = pd.to_datetime(df_clean['date'])
        meta_df = pd.read_csv(meta_path)
        df_features = pd.read_csv(feat_path)
        df_features['date'] = pd.to_datetime(df_features['date'])
    else:
        loader = DataLoader()
        prep = DataPreprocessor(loader)
        df_clean = prep.clean_gtrends()
        trend_cols = [c for c in df_clean.columns if c != 'date']
        meta_df = prep.generate_trend_metadata(trend_cols)
        fe = FeatureEngineer()
        df_features = fe.generate_all_features(df_clean, meta_df)

    return df_clean, meta_df, df_features

@st.cache_resource
def load_trained_models():
    models_dir = "models"
    models = {}
    
    rf_path = os.path.join(models_dir, "random_forest.joblib")
    xgb_path = os.path.join(models_dir, "xgboost.joblib")
    lr_path = os.path.join(models_dir, "logistic_regression.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")

    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    if os.path.exists(rf_path):
        models['Random Forest'] = (joblib.load(rf_path), False)
    if os.path.exists(xgb_path):
        models['XGBoost'] = (joblib.load(xgb_path), False)
    if os.path.exists(lr_path):
        models['Logistic Regression'] = (joblib.load(lr_path), True)

    return models, scaler

df_clean, meta_df, df_features = load_datasets()
models, scaler = load_trained_models()

# Create metadata lookup dictionaries
meta_map = dict(zip(meta_df['trend_name'], meta_df['trend_type']))

# ---------------------------------------------------------
# SIDEBAR FILTERS & CONTROLS
# ---------------------------------------------------------
st.sidebar.header("Filter & Select")

# Filter by Trend Type
type_options = ["All", "Category", "Color", "Fabric"]
selected_type = st.sidebar.selectbox("Filter Trend Type", options=type_options, index=0)

# Filter available trends based on selected type
all_trends = [c for c in df_clean.columns if c != 'date']
if selected_type != "All":
    filtered_trends = [t for t in all_trends if meta_map.get(t, 'other').lower() == selected_type.lower()]
else:
    filtered_trends = all_trends

if not filtered_trends:
    filtered_trends = all_trends

# Select Trend Dropdown
default_idx = filtered_trends.index('miniskirt') if 'miniskirt' in filtered_trends else 0
selected_trend = st.sidebar.selectbox("Select Trend", options=sorted(filtered_trends), index=default_idx)

# Select ML Model
model_options = list(models.keys()) if models else ["None"]
selected_model_name = st.sidebar.selectbox("Model", options=model_options, index=0)

st.divider()

# ---------------------------------------------------------
# MAIN TREND VISUALIZATION
# ---------------------------------------------------------
st.subheader(f"Historical Trend: {selected_trend}")
chart_data = df_clean[['date', selected_trend]].set_index('date')
st.line_chart(chart_data)

st.divider()

# ---------------------------------------------------------
# RECENT TREND INFORMATION
# ---------------------------------------------------------
st.subheader("Recent Trend Behavior")

# Get latest available feature row for selected trend
trend_feat_df = df_features[df_features['trend_name'] == selected_trend].sort_values('date').reset_index(drop=True)

if len(trend_feat_df) > 0:
    latest_row = trend_feat_df.iloc[-1]
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Current Value", f"{latest_row['current_value']:.1f}")
    with col2:
        st.metric("Recent % Change", f"{latest_row['percentage_change']:+.2f}%")
    with col3:
        st.metric("Trend Slope", f"{latest_row['trend_slope']:+.2f}")
    with col4:
        st.metric("Momentum", f"{latest_row['momentum']:+.2f}")
    with col5:
        st.metric("Volatility", f"{latest_row['volatility']:.3f}")
    with col6:
        st.metric("Persistence", f"{int(latest_row['persistence'])} wks")
else:
    st.warning("Insufficient feature data available for this trend.")

st.divider()

# ---------------------------------------------------------
# ML PREDICTION & PROBABILITIES
# ---------------------------------------------------------
st.subheader("ML Prediction")

feature_cols = [
    'current_value', 'previous_value', 'absolute_change', 'percentage_change',
    'moving_average_4', 'rolling_std_4', 'moving_average_8', 'trend_slope',
    'momentum', 'volatility', 'persistence'
]

if selected_model_name != "None" and selected_model_name in models and len(trend_feat_df) > 0:
    model, is_scaled = models[selected_model_name]
    
    # Input vector (11 features strictly past/current)
    X_in = pd.DataFrame([latest_row[feature_cols].values], columns=feature_cols)
    if is_scaled and scaler is not None:
        X_in = pd.DataFrame(scaler.transform(X_in), columns=feature_cols)

    pred_class = int(model.predict(X_in)[0])
    probs = model.predict_proba(X_in)[0]

    label_names = ["Stable", "Emerging Trend", "Strong Trend"]
    pred_label = label_names[pred_class]
    confidence = float(probs[pred_class]) * 100

    col_pred, col_prob = st.columns([1, 1])

    with col_pred:
        st.markdown(f"### Prediction: **{pred_label}**")
        st.markdown(f"**Confidence:** `{confidence:.2f}%`")

    with col_prob:
        st.markdown("**Class Probabilities:**")
        st.write(f"- **Stable:** `{probs[0]*100:.1f}%`")
        st.write(f"- **Emerging Trend:** `{probs[1]*100:.1f}%`")
        st.write(f"- **Strong Trend:** `{probs[2]*100:.1f}%`")

    # ---------------------------------------------------------
    # WARNING STATUS
    # ---------------------------------------------------------
    st.divider()
    st.subheader("Warning Status")

    if pred_class == 2:
        st.error("### 🚨 ALERT")
        st.write("A strong trend has been detected. Increase manufacturing priority and stock allocation.")
    elif pred_class == 1:
        st.warning("### ⚠️ WATCH")
        st.write("An emerging trend has been detected. Continue monitoring inventory and supplier preparedness.")
    else:
        st.success("### ✅ NORMAL")
        st.write("No significant emerging trend detected. Search volume remains within baseline levels.")

else:
    st.error("Selected model is unavailable. Please run model training first.")

st.divider()

# ---------------------------------------------------------
# RELATED FASHION INFORMATION
# ---------------------------------------------------------
st.subheader("Related Fashion Information")
trend_type_val = meta_map.get(selected_trend, 'other').capitalize()
st.write(f"**Trend Attribute Type:** {trend_type_val}")

analyzer = ProductAnalyzer()
prod_info = analyzer.get_product_attribute_distribution(selected_trend)

if prod_info['match_count'] > 0:
    st.write(f"Matching Catalog Items in `train.csv`: **{prod_info['match_count']}**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Category**")
        for k, v in prod_info['category_dist'].items():
            st.write(f"- {k}: `{v}` items")
    with c2:
        st.write("**Color**")
        for k, v in prod_info['color_dist'].items():
            st.write(f"- {k}: `{v}` items")
    with c3:
        st.write("**Fabric**")
        for k, v in prod_info['fabric_dist'].items():
            st.write(f"- {k}: `{v}` items")
else:
    st.info("No direct product catalog records matched this exact trend attribute name.")

st.divider()

# ---------------------------------------------------------
# EXPANDABLE SECTIONS: IMPORTANT FEATURES & MODEL PERFORMANCE
# ---------------------------------------------------------
with st.expander("Important Features"):
    fi_path = os.path.join("outputs", "feature_importance.csv")
    if os.path.exists(fi_path):
        fi_df = pd.read_csv(fi_path)
        st.dataframe(fi_df, hide_index=True)
        
        # Horizontal bar chart of top features
        fig, ax = plt.subplots(figsize=(8, 4))
        fi_sorted = fi_df.sort_values(by='mean_importance', ascending=True)
        ax.barh(fi_sorted['feature'], fi_sorted['mean_importance'], color='#1f77b4')
        ax.set_title('Feature Importance (Mean across Models)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Importance Score')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Feature importance data file not found.")

with st.expander("Model Performance"):
    comp_path = os.path.join("outputs", "model_comparison.csv")
    if os.path.exists(comp_path):
        comp_df = pd.read_csv(comp_path)
        st.dataframe(comp_df, hide_index=True)
    else:
        st.info("Model comparison data file not found.")
