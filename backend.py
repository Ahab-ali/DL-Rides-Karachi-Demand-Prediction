"""
Karachi Taxi Demand Forecasting - Enhanced Backend
WITH DATA DRIFT DETECTION + SHAP/LIME EXPLAINABILITY
Evidently 0.7+ (Latest)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import logging
from tensorflow import keras

# Evidently AI imports - Latest version (0.7+)
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

# Explainability imports
import shap
import lime
import lime.lime_tabular

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Karachi Taxi Demand API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class PredictionRequest(BaseModel):
    datetime: str = Field(..., description="Format: YYYY-MM-DD HH:MM:SS")
    regions: List[str] = Field(default=None)

class PredictionResponse(BaseModel):
    datetime: str
    predictions: Dict[str, float]

class DriftRequest(BaseModel):
    reference_sample_size: int = Field(default=5000)
    current_sample_size: int = Field(default=5000)
    features: Optional[List[str]] = None

class DriftResponse(BaseModel):
    dataset_drift: bool
    drift_score: float
    drifted_features: List[str]
    feature_drift_scores: Dict[str, float]
    reference_size: int
    current_size: int
    timestamp: str

class ExplainRequest(BaseModel):
    datetime: str = Field(..., description="Format: YYYY-MM-DD HH:MM:SS")
    region: str = Field(..., description="H3 region to explain")
    method: str = Field(default="shap", description="'shap' or 'lime'")
    background_size: int = Field(default=100, description="Background sample size for SHAP")

class ExplainResponse(BaseModel):
    datetime: str
    region: str
    method: str
    prediction: float
    actual: float
    feature_names: List[str]
    feature_values: List[float]
    shap_values: Optional[List[float]] = None
    lime_weights: Optional[Dict[str, float]] = None

# Global variables
MODEL = None
ENCODER = None
TRAIN_DATA = None
TEST_DATA = None

def load_components():
    """Load model, encoder, and datasets"""
    global MODEL, ENCODER, TRAIN_DATA, TEST_DATA
    
    try:
        current_path = Path(__file__).resolve()
        possible_roots = [current_path.parent, current_path.parent.parent, current_path.parent.parent.parent]
        
        model_path = encoder_path = train_path = test_path = None
        
        for root in possible_roots:
            test_model = root / "models" / "model.keras"
            test_encoder = root / "models" / "encoder.joblib"
            test_train = root / "data" / "processed" / "train.csv"
            test_test = root / "data" / "processed" / "test.csv"
            
            if all([test_model.exists(), test_encoder.exists(), test_train.exists(), test_test.exists()]):
                model_path, encoder_path, train_path, test_path = test_model, test_encoder, test_train, test_test
                break
        
        if not all([model_path, encoder_path, train_path, test_path]):
            raise FileNotFoundError("Required files not found")
        
        logger.info(f"Loading model from {model_path}")
        # ✅ FIXED: Load without compiling
        MODEL = keras.models.load_model(model_path, compile=False)
        logger.info("✅ Model loaded")
        
        logger.info(f"Loading encoder from {encoder_path}")
        ENCODER = joblib.load(encoder_path)
        logger.info("✅ Encoder loaded")
        
        logger.info(f"Loading training data from {train_path}")
        TRAIN_DATA = pd.read_csv(train_path, parse_dates=["tpep_pickup_datetime"])
        TRAIN_DATA = TRAIN_DATA.set_index("tpep_pickup_datetime")
        logger.info(f"✅ Loaded {len(TRAIN_DATA)} training records")
        
        logger.info(f"Loading test data from {test_path}")
        TEST_DATA = pd.read_csv(test_path, parse_dates=["tpep_pickup_datetime"])
        TEST_DATA = TEST_DATA.set_index("tpep_pickup_datetime")
        logger.info(f"✅ Loaded {len(TEST_DATA)} test records")
        
    except Exception as e:
        logger.error(f"❌ Loading error: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting API with Evidently 0.7+ & SHAP/LIME...")
    load_components()
    logger.info("✅ API ready!")

@app.get("/")
async def root():
    return {
        "message": "Karachi Taxi Demand API with Drift Detection & Explainability",
        "version": "2.1.0",
        "evidently_version": "0.7+",
        "features": {
            "prediction": True, 
            "drift_detection": True,
            "shap_explainability": True,
            "lime_explainability": True
        },
        "data_loaded": {"train": TRAIN_DATA is not None, "test": TEST_DATA is not None},
        "train_info": {
            "records": len(TRAIN_DATA),
            "date_range": {"start": str(TRAIN_DATA.index.min()), "end": str(TRAIN_DATA.index.max())},
            "regions": int(TRAIN_DATA['region'].nunique())
        } if TRAIN_DATA is not None else None,
        "test_info": {
            "records": len(TEST_DATA),
            "date_range": {"start": str(TEST_DATA.index.min()), "end": str(TEST_DATA.index.max())},
            "regions": int(TEST_DATA['region'].nunique())
        } if TEST_DATA is not None else None
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": MODEL is not None,
        "encoder": ENCODER is not None,
        "train_data": TRAIN_DATA is not None,
        "test_data": TEST_DATA is not None,
        "evidently": "0.7+",
        "explainability": ["shap", "lime"]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions using trained model"""
    try:
        if MODEL is None or ENCODER is None or TEST_DATA is None:
            raise HTTPException(status_code=503, detail="Components not loaded")
        
        dt = pd.to_datetime(request.datetime)
        
        if dt not in TEST_DATA.index:
            raise HTTPException(status_code=400, detail=f"Datetime {dt} not in dataset")
        
        data_at_time = TEST_DATA.loc[dt]
        if isinstance(data_at_time, pd.Series):
            data_at_time = pd.DataFrame([data_at_time])
        
        data_at_time = data_at_time.sort_values('region')
        
        if request.regions:
            data_at_time = data_at_time[data_at_time['region'].isin(request.regions)]
        
        if len(data_at_time) == 0:
            raise HTTPException(status_code=400, detail="No data found")
        
        X = data_at_time.drop(columns=['total_pickups'])
        regions = X['region'].tolist()
        
        X_encoded = ENCODER.transform(X)
        if isinstance(X_encoded, pd.DataFrame):
            X_encoded = X_encoded.values
        
        X_reshaped = X_encoded.reshape(X_encoded.shape[0], 1, X_encoded.shape[1])
        predictions = MODEL.predict(X_reshaped, verbose=0).flatten()
        
        result = {regions[i]: max(0, float(predictions[i])) for i in range(len(regions))}
        
        logger.info(f"Made {len(result)} predictions for {dt}")
        return PredictionResponse(datetime=request.datetime, predictions=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/drift/detect", response_model=DriftResponse)
async def detect_drift(request: DriftRequest):
    """Detect data drift using Evidently AI 0.7+"""
    try:
        if TRAIN_DATA is None or TEST_DATA is None:
            raise HTTPException(status_code=503, detail="Data not loaded")
        
        logger.info("Starting drift detection with Evidently 0.7+...")
        
        # Sample data
        ref_size = min(request.reference_sample_size, len(TRAIN_DATA))
        cur_size = min(request.current_sample_size, len(TEST_DATA))
        
        ref_sample = TRAIN_DATA.sample(n=ref_size, random_state=42)
        cur_sample = TEST_DATA.sample(n=cur_size, random_state=42)
        
        # Get numerical features
        numerical_cols = ref_sample.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col != 'total_pickups']
        features = request.features if request.features else numerical_cols[:20]
        
        # Prepare data
        ref_df = ref_sample[features + ['total_pickups']].reset_index(drop=True)
        cur_df = cur_sample[features + ['total_pickups']].reset_index(drop=True)
        
        logger.info(f"Analyzing {len(features)} features")
        
        # Create Evidently report
        report = Report(metrics=[DataDriftPreset()])
        
        # Run report
        report.run(reference_data=ref_df, current_data=cur_df)
        
        # Extract results
        results_dict = report.as_dict()
        
        # Parse dataset drift
        dataset_drift = results_dict['metrics'][0]['result']['dataset_drift']
        drift_score = results_dict['metrics'][0]['result']['drift_share']
        
        # Parse feature drift
        drifted_features = []
        feature_drift_scores = {}
        
        for metric in results_dict['metrics'][1:]:
            col_name = metric['result']['column_name']
            is_drifted = metric['result']['drift_detected']
            drift_value = metric['result'].get('drift_score', 0.0)
            
            feature_drift_scores[col_name] = drift_value
            if is_drifted:
                drifted_features.append(col_name)
        
        logger.info(f"Drift detected: {dataset_drift}, Features drifted: {len(drifted_features)}/{len(features)}")
        
        return DriftResponse(
            dataset_drift=dataset_drift,
            drift_score=float(drift_score),
            drifted_features=drifted_features,
            feature_drift_scores=feature_drift_scores,
            reference_size=ref_size,
            current_size=cur_size,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Drift detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/drift/summary")
async def drift_summary():
    """Quick drift summary"""
    try:
        if TRAIN_DATA is None or TEST_DATA is None:
            raise HTTPException(status_code=503, detail="Data not loaded")
        
        train_stats = TRAIN_DATA['total_pickups'].describe().to_dict()
        test_stats = TEST_DATA['total_pickups'].describe().to_dict()
        
        return {
            "reference": {
                "records": len(TRAIN_DATA),
                "target_stats": train_stats,
                "date_range": {"start": str(TRAIN_DATA.index.min()), "end": str(TRAIN_DATA.index.max())}
            },
            "current": {
                "records": len(TEST_DATA),
                "target_stats": test_stats,
                "date_range": {"start": str(TEST_DATA.index.min()), "end": str(TEST_DATA.index.max())}
            },
            "indicators": {
                "mean_change_pct": ((test_stats['mean'] - train_stats['mean']) / train_stats['mean'] * 100),
                "std_change_pct": ((test_stats['std'] - train_stats['std']) / train_stats['std'] * 100)
            }
        }
    except Exception as e:
        logger.error(f"Summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain", response_model=ExplainResponse)
async def explain_prediction(request: ExplainRequest):
    """Explain prediction using SHAP or LIME"""
    try:
        if MODEL is None or ENCODER is None or TEST_DATA is None:
            raise HTTPException(status_code=503, detail="Components not loaded")
        
        dt = pd.to_datetime(request.datetime)
        
        if dt not in TEST_DATA.index:
            raise HTTPException(status_code=400, detail=f"Datetime {dt} not in dataset")
        
        data_at_time = TEST_DATA.loc[dt]
        if isinstance(data_at_time, pd.Series):
            data_at_time = pd.DataFrame([data_at_time])
        
        # Get specific region
        region_data = data_at_time[data_at_time['region'] == request.region]
        if len(region_data) == 0:
            raise HTTPException(status_code=400, detail=f"Region {request.region} not found at {dt}")
        
        X = region_data.drop(columns=['total_pickups'])
        y_actual = region_data['total_pickups'].iloc[0]
        
        # Encode features
        X_encoded = ENCODER.transform(X)
        if isinstance(X_encoded, pd.DataFrame):
            X_encoded = X_encoded.values
        
        # Get feature names
        feature_names = ENCODER.get_feature_names_out().tolist()
        feature_values = X_encoded[0].tolist()
        
        # Make prediction
        X_reshaped = X_encoded.reshape(X_encoded.shape[0], 1, X_encoded.shape[1])
        prediction = MODEL.predict(X_reshaped, verbose=0)[0][0]
        
        logger.info(f"Explaining prediction for {request.region} at {dt} using {request.method}")
        
        # SHAP explanation
        if request.method.lower() == "shap":
            # Get background data
            background_sample = TEST_DATA.sample(n=min(request.background_size, len(TEST_DATA)), random_state=42)
            background_features = background_sample.drop(columns=['total_pickups'])
            background_encoded = ENCODER.transform(background_features)
            
            if isinstance(background_encoded, pd.DataFrame):
                background_encoded = background_encoded.values
            
            background_reshaped = background_encoded.reshape(background_encoded.shape[0], 1, background_encoded.shape[1])
            
            # Use GradientExplainer instead of DeepExplainer for TensorFlow 2.x compatibility
            explainer = shap.GradientExplainer(MODEL, background_reshaped)
            shap_values_raw = explainer.shap_values(X_reshaped)
            
            # Handle list output (GradientExplainer returns [array])
            if isinstance(shap_values_raw, list):
                shap_values_raw = shap_values_raw[0]
            
            # Reshape to exactly (n_samples, n_features)
            n_samples = X_encoded.shape[0]
            n_features = X_encoded.shape[1]
            shap_values = shap_values_raw.reshape(n_samples, n_features)
            
            # Extract SHAP values for the single instance
            shap_vals = shap_values[0].tolist()
            
            logger.info(f"SHAP values computed for {request.region}")
            
            return ExplainResponse(
                datetime=request.datetime,
                region=request.region,
                method="shap",
                prediction=float(prediction),
                actual=float(y_actual),
                feature_names=feature_names,
                feature_values=feature_values,
                shap_values=shap_vals,
                lime_weights=None
            )
        
        # LIME explanation
        elif request.method.lower() == "lime":
            # Create prediction function for LIME
            def predict_fn(X):
                X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
                return MODEL.predict(X_reshaped, verbose=0)
            
            # Get sample data for LIME explainer
            sample_data = TEST_DATA.sample(n=min(1000, len(TEST_DATA)), random_state=42)
            sample_features = sample_data.drop(columns=['total_pickups'])
            sample_encoded = ENCODER.transform(sample_features)
            
            if isinstance(sample_encoded, pd.DataFrame):
                sample_encoded = sample_encoded.values
            
            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                sample_encoded,
                feature_names=feature_names,
                mode='regression',
                verbose=False
            )
            
            # Explain the instance
            exp = explainer.explain_instance(
                X_encoded[0],
                predict_fn,
                num_features=len(feature_names)
            )
            
            # Extract LIME weights
            lime_weights = dict(exp.as_list())
            
            logger.info(f"LIME explanation computed for {request.region}")
            
            return ExplainResponse(
                datetime=request.datetime,
                region=request.region,
                method="lime",
                prediction=float(prediction),
                actual=float(y_actual),
                feature_names=feature_names,
                feature_values=feature_values,
                shap_values=None,
                lime_weights=lime_weights
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Invalid method: {request.method}. Use 'shap' or 'lime'")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║   Karachi Taxi Demand API v2.1                           ║
    ║   WITH EVIDENTLY 0.7+ & SHAP/LIME                        ║
    ╚══════════════════════════════════════════════════════════╝
    
    📍 Endpoints:
       • POST /predict              - Demand predictions
       • POST /drift/detect         - Data drift detection
       • GET  /drift/summary        - Quick drift summary
       • POST /explain              - SHAP/LIME explanations
       • GET  /health               - Health check
    
    🚀 http://localhost:8000
    📚 http://localhost:8000/docs
    🔬 Evidently AI 0.7+ Real Drift Detection
    🧠 SHAP & LIME Explainability
    """)
    
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)