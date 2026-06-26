"""
Karachi Taxi Demand Forecasting - Streamlit App
WITH EVIDENTLY AI DATA DRIFT MONITORING + SHAP/LIME EXPLAINABILITY
Evidently 0.7+ (Latest Version)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium
import h3
from datetime import datetime, timedelta, date, time as dt_time
from pathlib import Path
from time import sleep
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns

# Evidently AI imports - Latest version (0.7+)
from evidently.ui.workspace import Workspace
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

# Explainability imports
import shap
import lime
import lime.lime_tabular
from tensorflow import keras
import joblib

# H3 RESOLUTION
H3_RESOLUTION = 6

# KARACHI BOUNDARIES
KARACHI_MIN_LAT = 24.81
KARACHI_MAX_LAT = 25.11
KARACHI_MIN_LNG = 66.95
KARACHI_MAX_LNG = 67.31

# Page config
st.set_page_config(
    page_title="Karachi Taxi Demand Forecasting",
    page_icon="🚕",
    layout="wide"
)

# CSS
st.markdown("""
<style>
.main { background-color: #fafbfc; padding: 1rem; }
h1 { font-size: 2rem !important; font-weight: 600 !important; color: #1a202c !important; }
h2 { font-size: 1.5rem !important; font-weight: 500 !important; color: #2d3748 !important; }
h3 { font-size: 1.1rem !important; font-weight: 500 !important; color: #4a5568 !important; }
.stButton>button {
    background-color: #3182ce !important;
    color: white !important;
    border-radius: 6px !important;
    padding: 0.5rem 1.5rem !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 600 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.85rem !important;
    color: #718096 !important;
}
.info-box {
    background-color: #ebf8ff;
    padding: 1rem;
    border-radius: 6px;
    border-left: 3px solid #3182ce;
    margin: 0.75rem 0;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_test_data():
    """Load test dataset"""
    try:
        data_path = Path("data/processed/test.csv")
        df = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"])
        df = df.set_index("tpep_pickup_datetime")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def load_train_data():
    """Load training dataset for drift reference"""
    try:
        data_path = Path("data/processed/train.csv")
        df = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"])
        df = df.set_index("tpep_pickup_datetime")
        return df
    except Exception as e:
        st.error(f"Error loading training data: {e}")
        return None

@st.cache_data
def load_plot_data():
    """Load plot data for map visualization"""
    try:
        test_df = load_test_data()
        if test_df is not None:
            plot_df = test_df.groupby('region').first().reset_index()
            
            if 'pickup_latitude' not in plot_df.columns:
                coords = plot_df['region'].apply(lambda x: h3.cell_to_latlng(x))
                plot_df['pickup_latitude'] = coords.apply(lambda x: x[0])
                plot_df['pickup_longitude'] = coords.apply(lambda x: x[1])
            
            return plot_df[['region', 'pickup_latitude', 'pickup_longitude']]
    except Exception as e:
        st.error(f"Error creating plot data: {e}")
    return None

@st.cache_resource
def load_model_and_encoder():
    """Load model and encoder for explainability"""
    try:
        model_path = Path("models/model.keras")
        encoder_path = Path("models/encoder.joblib")
        
        if not model_path.exists() or not encoder_path.exists():
            return None, None
        
        # ✅ FIXED: Load without compiling
        model = keras.models.load_model(model_path, compile=False)
        encoder = joblib.load(encoder_path)
        return model, encoder
    except Exception as e:
        st.error(f"Error loading model/encoder: {e}")
        return None, None

def h3_to_polygon(h3_index: str):
    """Convert H3 to polygon"""
    try:
        boundary = h3.cell_to_boundary(h3_index)
        return [(lat, lng) for lat, lng in boundary]
    except:
        return []

def get_h3_neighbors(h3_index: str, k: int = 1):
    """Get H3 neighbors at distance k"""
    try:
        return list(h3.grid_disk(h3_index, k))
    except:
        return [h3_index]

def lat_lng_to_h3(lat: float, lng: float, resolution: int = 6) -> str:
    """Convert lat/lng to H3"""
    try:
        return h3.latlng_to_cell(lat, lng, resolution)
    except Exception as e:
        st.error(f"Error converting to H3: {e}")
        return None

def create_map(plot_df, predictions, current_region):
    """Create Folium map with predictions"""
    center_lat, center_lng = h3.cell_to_latlng(current_region)
    m = folium.Map(location=[center_lat, center_lng], zoom_start=12, tiles='CartoDB positron')
    
    if predictions:
        max_pred = max(predictions.values())
        min_pred = min(predictions.values())
        pred_range = max_pred - min_pred if max_pred != min_pred else 1
    
    for region, pred_value in predictions.items():
        boundary = h3_to_polygon(region)
        if not boundary:
            continue
        
        normalized = (pred_value - min_pred) / pred_range if pred_range > 0 else 0.5
        r = int(50 + 205 * normalized)
        g = int(200 - 150 * normalized)
        b = 50
        color = f'#{r:02x}{g:02x}{b:02x}'
        
        if region == current_region:
            weight, fill_opacity, border_color = 4, 0.8, '#0000FF'
        else:
            weight, fill_opacity, border_color = 2, 0.6, color
        
        folium.Polygon(
            locations=boundary, color=border_color, weight=weight, fill=True,
            fill_color=color, fill_opacity=fill_opacity,
            popup=f"Region: {region[:10]}...<br>Demand: {pred_value:.0f}",
            tooltip=f"Demand: {pred_value:.0f}"
        ).add_to(m)
        
        lat, lng = h3.cell_to_latlng(region)
        label_html = f"""<div style="color: {'#0000FF' if region == current_region else '#1a202c'}; 
                         font-weight: bold; font-size: {'20px' if region == current_region else '16px'}; 
                         text-align: center; text-shadow: -1px -1px 0 #fff, 1px -1px 0 #fff, 
                         -1px 1px 0 #fff, 1px 1px 0 #fff; white-space: nowrap;">
                         {'🎯' if region == current_region else ''}{pred_value:.0f}</div>"""
        folium.Marker(location=[lat, lng], icon=folium.DivIcon(html=label_html)).add_to(m)
    
    current_lat, current_lng = h3.cell_to_latlng(current_region)
    folium.Marker(location=[current_lat, current_lng], popup='Your Location',
                  icon=folium.Icon(color='blue', icon='star', prefix='fa')).add_to(m)
    return m

def get_predictions(datetime_str, regions=None, backend_url="http://localhost:8000"):
    """Get predictions from backend"""
    try:
        payload = {"datetime": datetime_str}
        if regions:
            payload["regions"] = regions
        response = requests.post(f"{backend_url}/predict", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["predictions"]
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to backend. Start FastAPI server first.")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Function to generate drift report
def generate_drift_report(reference_data, current_data):
    """Generate Evidently AI drift report - Evidently 0.7+"""
    
    report = Report(metrics=[
        DataDriftPreset(),
        DataSummaryPreset(),
    ])
    
    # ✅ CORRECT: run() returns an evaluation object
    my_eval = report.run(reference_data=reference_data, current_data=current_data)
    
    # ✅ Call save_html() on the RETURN VALUE, not on report
    my_eval.save_html("report.html")
    
    with open("report.html", "r", encoding="utf-8") as f:
        html_content = f.read()
        
    return html_content

def compute_shap_values(model, encoder, X_sample, background_sample):
    """Compute SHAP values for model predictions"""
    try:
        # Encode the samples
        X_encoded = encoder.transform(X_sample)
        background_encoded = encoder.transform(background_sample)
        
        if isinstance(X_encoded, pd.DataFrame):
            X_encoded = X_encoded.values
        if isinstance(background_encoded, pd.DataFrame):
            background_encoded = background_encoded.values
        
        # Store original dimensions
        n_samples = X_encoded.shape[0]
        n_features = X_encoded.shape[1]
        
        # Reshape for LSTM/GRU (add time dimension)
        X_reshaped = X_encoded.reshape(n_samples, 1, n_features)
        background_reshaped = background_encoded.reshape(background_encoded.shape[0], 1, n_features)
        
        # Use GradientExplainer instead of DeepExplainer for TensorFlow 2.x compatibility
        explainer = shap.GradientExplainer(model, background_reshaped)
        shap_values_raw = explainer.shap_values(X_reshaped)
        
        # Handle list output (GradientExplainer returns [array])
        if isinstance(shap_values_raw, list):
            shap_values_raw = shap_values_raw[0]
        
        # Reshape to exactly (n_samples, n_features) - this handles any shape from explainer
        shap_values = shap_values_raw.reshape(n_samples, n_features)
        
        return shap_values, X_encoded, encoder.get_feature_names_out()
    except Exception as e:
        st.error(f"Error computing SHAP values: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None

def compute_lime_explanation(model, encoder, X_sample, feature_names):
    """Compute LIME explanation for a single prediction"""
    try:
        # Encode the sample
        X_encoded = encoder.transform(X_sample)
        if isinstance(X_encoded, pd.DataFrame):
            X_encoded = X_encoded.values
        
        # Create prediction function for LIME
        def predict_fn(X):
            X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
            return model.predict(X_reshaped, verbose=0)
        
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_encoded,
            feature_names=feature_names,
            mode='regression',
            verbose=False
        )
        
        # Explain the first instance
        exp = explainer.explain_instance(
            X_encoded[0],
            predict_fn,
            num_features=10
        )
        
        return exp
    except Exception as e:
        st.error(f"Error computing LIME explanation: {e}")
        return None

# ==================== SIDEBAR NAVIGATION ====================
st.sidebar.title("🚕 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["📊 Demand Predictions", "🔍 Data Drift Monitoring", "🧠 Explainable AI"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")
st.sidebar.markdown(f"**H3 Resolution:** {H3_RESOLUTION}")
backend_url = st.sidebar.text_input("Backend URL:", value="http://localhost:8000")
st.sidebar.markdown("---")
st.sidebar.caption("**Model:** GRU Neural Network")
st.sidebar.caption(f"**Data:** March 2016")

# ==================== MAIN TITLE ====================
st.title("🚕 Karachi Taxi Demand Forecasting & Monitoring")
st.caption("🔬 Powered by Evidently AI 0.7+ | SHAP & LIME Explainability")

# ==================== PAGE 1: PREDICTIONS ====================
if page == "📊 Demand Predictions":
    df = load_test_data()
    plot_df = load_plot_data()
    if df is None:
        st.error("❌ Could not load data")
        st.stop()

    st.subheader("📅 Date & Time Selection")
    col1, col2 = st.columns(2)
    with col1:
        selected_date = st.date_input("Date:", value=date(2016, 3, 15),
                                     min_value=date(2016, 3, 1), max_value=date(2016, 3, 31))
    with col2:
        hour = st.selectbox("Hour:", list(range(24)), index=12)
        minute = st.selectbox("Minute:", [0, 15, 30, 45], index=0)

    current_datetime = datetime(selected_date.year, selected_date.month, selected_date.day, hour, minute)
    next_datetime = current_datetime + timedelta(minutes=15)
    st.markdown(f"**🔮 Predicting for:** {next_datetime.strftime('%Y-%m-%d %H:%M')}")
    st.markdown("---")

    st.subheader("📍 Location Selection")
    location_mode = st.radio("Method:", ["🎲 Random", "✏️ Manual"], horizontal=True)

    lat, lng, region = None, None, None

    if location_mode == "🎲 Random":
        if plot_df is not None and st.button("🎲 Get Random Location", type="primary"):
            sample = plot_df.sample(1).iloc[0]
            st.session_state['lat'] = sample["pickup_latitude"]
            st.session_state['lng'] = sample["pickup_longitude"]
            st.session_state['region'] = sample["region"]
    else:
        col_lat, col_lng = st.columns(2)
        with col_lat:
            manual_lat = st.number_input("Latitude:", min_value=KARACHI_MIN_LAT, 
                                        max_value=KARACHI_MAX_LAT, value=24.8607, step=0.0001)
        with col_lng:
            manual_lng = st.number_input("Longitude:", min_value=KARACHI_MIN_LNG,
                                        max_value=KARACHI_MAX_LNG, value=67.0011, step=0.0001)
        if st.button("✅ Set Location", type="primary"):
            if KARACHI_MIN_LAT <= manual_lat <= KARACHI_MAX_LAT:
                st.session_state['lat'] = manual_lat
                st.session_state['lng'] = manual_lng
                st.session_state['region'] = lat_lng_to_h3(manual_lat, manual_lng, H3_RESOLUTION)
                st.success("✅ Location set!")

    if 'lat' in st.session_state:
        lat, lng, region = st.session_state['lat'], st.session_state['lng'], st.session_state['region']
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Latitude", f"{lat:.4f}")
        col2.metric("Longitude", f"{lng:.4f}")
        col3.metric("H3 Region", f"{region[:10]}...")

        datetime_str = next_datetime.strftime('%Y-%m-%d %H:%M:%S')
        regions_to_predict = get_h3_neighbors(region, k=1)

        with st.spinner("🔄 Generating predictions..."):
            predictions = get_predictions(datetime_str, regions_to_predict, backend_url)

        if predictions:
            map_obj = create_map(plot_df[plot_df['region'].isin(predictions.keys())], predictions, region)
            st_folium(map_obj, width=None, height=600)
            
            st.markdown("### 📊 Statistics")
            pred_values = list(predictions.values())
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("🎯 Your Region", f"{predictions.get(region, 0):.0f}")
            col2.metric("📈 Highest", f"{max(pred_values):.0f}")
            col3.metric("📊 Average", f"{np.mean(pred_values):.1f}")
            col4.metric("📉 Lowest", f"{min(pred_values):.0f}")
            col5.metric("🔢 Regions", len(predictions))
    else:
        st.info("👆 Select a location above")

# ==================== PAGE 2: DRIFT MONITORING ====================
elif page == "🔍 Data Drift Monitoring":
    st.header("🔍 Data Drift Monitoring with Evidently AI")

    train_data = load_train_data()
    test_data = load_test_data()
    
    if train_data is None or test_data is None:
        st.error("❌ Cannot load data")
        st.stop()

    st.subheader("⚙️ Configuration")
    col1, col2 = st.columns(2)
    with col1:
        ref_size = st.slider("Reference Sample:", 1000, min(50000, len(train_data)), 
                            min(10000, len(train_data)), 1000)
    with col2:
        cur_size = st.slider("Current Sample:", 1000, min(50000, len(test_data)),
                            min(10000, len(test_data)), 1000)

    numerical_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numerical_cols if col not in ['total_pickups']]
    selected_features = st.multiselect("Features:", feature_cols, 
                                       default=feature_cols[:10] if len(feature_cols) > 10 else feature_cols)

    if len(selected_features) == 0:
        st.warning("⚠️ Select at least one feature")
        st.stop()

    if st.button("🔬 Generate Drift Report", type="primary", use_container_width=True):
        with st.spinner("🔄 Analyzing drift with Evidently AI..."):
            try:
                # Sample data
                ref_sample = train_data.sample(n=min(ref_size, len(train_data)), random_state=42)
                cur_sample = test_data.sample(n=min(cur_size, len(test_data)), random_state=42)

                # Prepare DataFrames
                ref_df = ref_sample[selected_features + ['total_pickups']].reset_index(drop=True)
                cur_df = cur_sample[selected_features + ['total_pickups']].reset_index(drop=True)

                # Progress bar
                progress = st.progress(0, text="Computing drift metrics...")

                # Generate report
                html_report = generate_drift_report(ref_df, cur_df)

                # Animate progress
                for i in range(1, 101):
                    sleep(0.01)
                    progress.progress(i, text="Computing drift metrics...")
                progress.empty()

                if html_report:
                    st.success("✅ Drift analysis complete!")
                    st.markdown("---")
                    st.subheader("📊 Evidently AI Drift Report")

                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Reference", f"{len(ref_df):,} rows")
                    col2.metric("Current", f"{len(cur_df):,} rows")
                    col3.metric("Features", len(selected_features))

                    # Download button
                    st.download_button(
                        "📥 Download Report",
                        html_report,
                        file_name=f"drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                st.code(traceback.format_exc())

    with st.expander("ℹ️ Understanding Drift Metrics"):
        st.markdown("""
        **Dataset Drift**: Overall assessment if data has drifted
        
        **Feature Drift Detection**:
        - **PSI (Population Stability Index)**: Measures distribution changes (>0.1 = minor, >0.25 = major)
        - **Wasserstein Distance**: Measures "distance" between distributions
        - **KS Test**: Statistical test for distribution similarity (p-value < 0.05 = drift)
        
        **Colors in Report**:
        - 🟢 Green: No drift detected
        - 🟡 Yellow: Warning - potential drift
        - 🔴 Red: Drift detected
        
        **Action Items**:
        - If drift detected → Investigate root cause
        - Multiple features drifting → Consider retraining
        - Monitor trends over time
        """)

# ==================== PAGE 3: EXPLAINABLE AI ====================
elif page == "🧠 Explainable AI":
    st.header("🧠 Explainable AI - SHAP & LIME")
    st.caption("Understanding model predictions through feature importance")

    # Load model and data
    model, encoder = load_model_and_encoder()
    test_data = load_test_data()
    
    if model is None or encoder is None:
        st.error("❌ Model or encoder not loaded. Check if files exist in models/ directory.")
        st.stop()
    
    if test_data is None:
        st.error("❌ Cannot load test data")
        st.stop()

    st.success("✅ Model and encoder loaded successfully!")
    
    # Select explanation method
    st.subheader("⚙️ Configuration")
    explainer_type = st.radio(
        "Select Explanation Method:",
        ["🔵 SHAP (Global & Local)", "🟢 LIME (Local)"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # Date/Time selection
    st.subheader("📅 Select Prediction Instance")
    col1, col2 = st.columns(2)
    with col1:
        selected_date = st.date_input("Date:", value=date(2016, 3, 15),
                                     min_value=date(2016, 3, 1), max_value=date(2016, 3, 31),
                                     key="xai_date")
    with col2:
        hour = st.selectbox("Hour:", list(range(24)), index=12, key="xai_hour")
        minute = st.selectbox("Minute:", [0, 15, 30, 45], index=0, key="xai_minute")
    
    target_datetime = datetime(selected_date.year, selected_date.month, selected_date.day, hour, minute)
    
    # Get data for selected datetime
    if target_datetime not in test_data.index:
        st.warning(f"⚠️ No data available for {target_datetime}")
        st.stop()
    
    data_at_time = test_data.loc[target_datetime]
    if isinstance(data_at_time, pd.Series):
        data_at_time = pd.DataFrame([data_at_time])
    
    # Sample size selection
    sample_size = st.slider("Sample Size:", 10, min(500, len(data_at_time)), 
                           min(100, len(data_at_time)), 10)
    
    background_size = st.slider("Background Sample (for SHAP):", 50, 
                               min(1000, len(test_data)), 100, 50)
    
    # Get samples
    X_sample = data_at_time.sample(n=min(sample_size, len(data_at_time)), random_state=42)
    X_sample_features = X_sample.drop(columns=['total_pickups'])
    y_sample = X_sample['total_pickups']
    
    # Background sample for SHAP
    background_data = test_data.sample(n=background_size, random_state=42)
    background_features = background_data.drop(columns=['total_pickups'])
    
    st.markdown(f"**Selected:** {len(X_sample_features)} instances from {target_datetime.strftime('%Y-%m-%d %H:%M')}")
    
    if st.button("🧠 Generate Explanations", type="primary", use_container_width=True):
        
        # ==================== SHAP EXPLANATIONS ====================
        if explainer_type == "🔵 SHAP (Global & Local)":
            with st.spinner("🔄 Computing SHAP values... This may take a minute..."):
                shap_values, X_encoded, feature_names = compute_shap_values(
                    model, encoder, X_sample_features, background_features
                )
                
                if shap_values is not None:
                    st.success("✅ SHAP values computed!")
                    
                    # Get predictions
                    X_reshaped = X_encoded.reshape(X_encoded.shape[0], 1, X_encoded.shape[1])
                    predictions = model.predict(X_reshaped, verbose=0).flatten()
                    
                    st.markdown("---")
                    st.subheader("📊 SHAP Analysis Results")
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Instances Analyzed", len(X_sample_features))
                    col2.metric("Avg Prediction", f"{np.mean(predictions):.2f}")
                    col3.metric("Features", len(feature_names))
                    
                    st.markdown("---")
                    
                    # Tab layout for different SHAP plots
                    tab1, tab2, tab3 = st.tabs([
                        "🎯 Summary Plot", 
                        "📊 Feature Importance", 
                        "🔍 Single Instance"
                    ])
                    
                    with tab1:
                        st.markdown("### 🎯 SHAP Summary Plot")
                        st.caption("Shows feature importance and impact direction across all instances")
                        
                        try:
                            fig, ax = plt.subplots(figsize=(10, 8))
                            shap.summary_plot(
                                shap_values, 
                                X_encoded, 
                                feature_names=feature_names,
                                show=False,
                                max_display=15
                            )
                            st.pyplot(fig)
                            plt.close()
                        except Exception as e:
                            st.error(f"Error creating summary plot: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                    
                    with tab2:
                        st.markdown("### 📊 Global Feature Importance")
                        st.caption("Average absolute SHAP values per feature")
                        
                        try:
                            # Calculate mean absolute SHAP values
                            mean_abs_shap = np.abs(shap_values).mean(axis=0)
                            
                            feature_importance = pd.DataFrame({
                                'Feature': list(feature_names),
                                'Importance': mean_abs_shap
                            }).sort_values('Importance', ascending=False)
                            
                            # Plot
                            fig, ax = plt.subplots(figsize=(10, 8))
                            top_features = feature_importance.head(15)
                            ax.barh(range(len(top_features)), top_features['Importance'].values)
                            ax.set_yticks(range(len(top_features)))
                            ax.set_yticklabels(top_features['Feature'].values)
                            ax.set_xlabel('Mean |SHAP Value|')
                            ax.set_title('Top 15 Most Important Features')
                            ax.invert_yaxis()
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                            
                            # Show table
                            st.markdown("#### Feature Importance Table")
                            st.dataframe(
                                feature_importance.head(20).style.format({'Importance': '{:.4f}'}),
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Error creating importance plot: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                    
                    with tab3:
                        st.markdown("### 🔍 Single Instance Explanation")
                        st.caption("SHAP values for a specific prediction")
                        
                        instance_idx = st.selectbox(
                            "Select Instance:",
                            range(len(X_sample_features)),
                            format_func=lambda x: f"Instance {x} (Actual: {y_sample.iloc[x]:.1f}, Predicted: {predictions[x]:.1f})"
                        )
                        
                        try:
                            st.markdown(f"**Actual Demand:** {y_sample.iloc[instance_idx]:.2f}")
                            st.markdown(f"**Predicted Demand:** {predictions[instance_idx]:.2f}")
                            st.markdown(f"**Prediction Error:** {abs(predictions[instance_idx] - y_sample.iloc[instance_idx]):.2f}")
                            
                            # Waterfall plot
                            st.markdown("#### SHAP Waterfall Plot")
                            fig, ax = plt.subplots(figsize=(10, 8))
                            shap.waterfall_plot(
                                shap.Explanation(
                                    values=shap_values[instance_idx],
                                    base_values=np.mean(predictions),
                                    data=X_encoded[instance_idx],
                                    feature_names=feature_names
                                ),
                                show=False,
                                max_display=15
                            )
                            st.pyplot(fig)
                            plt.close()
                            
                            # Force plot (as HTML)
                            st.markdown("#### SHAP Force Plot")
                            # Create force plot as HTML
                            force_plot = shap.force_plot(
                                base_value=np.mean(predictions),
                                shap_values=shap_values[instance_idx],
                                features=X_encoded[instance_idx],
                                feature_names=list(feature_names),
                                matplotlib=False
                            )
                            # Display using components.html
                            import shap
                            shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
                            components.html(shap_html, height=300)
                            
                        except Exception as e:
                            st.error(f"Error creating instance explanation: {e}")
                            import traceback
                            st.code(traceback.format_exc())
        
        # ==================== LIME EXPLANATIONS ====================
        else:  # LIME
            with st.spinner("🔄 Computing LIME explanations..."):
                try:
                    # Get feature names
                    feature_names = encoder.get_feature_names_out()
                    
                    # Compute LIME explanation
                    exp = compute_lime_explanation(
                        model, encoder, X_sample_features, feature_names
                    )
                    
                    if exp is not None:
                        st.success("✅ LIME explanation generated!")
                        
                        # Get prediction
                        X_encoded = encoder.transform(X_sample_features)
                        if isinstance(X_encoded, pd.DataFrame):
                            X_encoded = X_encoded.values
                        X_reshaped = X_encoded.reshape(X_encoded.shape[0], 1, X_encoded.shape[1])
                        prediction = model.predict(X_reshaped, verbose=0)[0][0]
                        
                        st.markdown("---")
                        st.subheader("🟢 LIME Local Explanation")
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Actual Demand", f"{y_sample.iloc[0]:.2f}")
                        col2.metric("Predicted Demand", f"{prediction:.2f}")
                        
                        st.markdown("---")
                        
                        # Get explanation as list
                        explanation_list = exp.as_list()
                        
                        # Create DataFrame for better visualization
                        lime_df = pd.DataFrame(explanation_list, columns=['Feature', 'Weight'])
                        lime_df = lime_df.sort_values('Weight', key=abs, ascending=False)
                        
                        # Plot
                        st.markdown("### 📊 Feature Contributions")
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
                        colors = ['green' if x > 0 else 'red' for x in lime_df['Weight']]
                        ax.barh(range(len(lime_df)), lime_df['Weight'], color=colors, alpha=0.6)
                        ax.set_yticks(range(len(lime_df)))
                        ax.set_yticklabels(lime_df['Feature'])
                        ax.set_xlabel('Feature Contribution')
                        ax.set_title('LIME Feature Importance (Green: Positive, Red: Negative)')
                        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
                        ax.invert_yaxis()
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # Show table
                        st.markdown("### 📋 Feature Weights Table")
                        st.dataframe(
                            lime_df.style.format({'Weight': '{:.4f}'})
                            .background_gradient(subset=['Weight'], cmap='RdYlGn', vmin=-1, vmax=1),
                            use_container_width=True
                        )
                        
                        # Explanation text
                        st.markdown("### 📝 Interpretation")
                        st.markdown("""
                        - **Green bars**: Features that increase the predicted demand
                        - **Red bars**: Features that decrease the predicted demand
                        - **Bar length**: Magnitude of the feature's impact
                        
                        LIME explains individual predictions by approximating the model locally 
                        with an interpretable linear model.
                        """)
                        
                except Exception as e:
                    st.error(f"❌ Error generating LIME explanation: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Explanation info
    with st.expander("ℹ️ Understanding SHAP vs LIME"):
        st.markdown("""
        ### 🔵 SHAP (SHapley Additive exPlanations)
        
        **Advantages:**
        - Theoretically grounded in game theory
        - Consistent and accurate feature attributions
        - Provides both global and local explanations
        - Additive feature attribution (contributions sum to prediction)
        
        **Best for:**
        - Understanding overall model behavior
        - Comparing feature importance across predictions
        - Deep learning models (with DeepExplainer)
        
        ---
        
        ### 🟢 LIME (Local Interpretable Model-agnostic Explanations)
        
        **Advantages:**
        - Model-agnostic (works with any model)
        - Fast computation
        - Intuitive local explanations
        - Easy to understand
        
        **Best for:**
        - Quick local explanations
        - Understanding individual predictions
        - Explaining complex models to stakeholders
        
        ---
        
        ### 💡 When to Use Which?
        
        - Use **SHAP** for comprehensive analysis and when you need theoretically sound attributions
        - Use **LIME** for quick local insights and when computational speed matters
        - Use **both** for cross-validation of explanations
        """)