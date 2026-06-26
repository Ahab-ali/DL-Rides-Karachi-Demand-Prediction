import mlflow
import dagshub
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
from sklearn import set_config
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras
import tempfile
import shutil

dagshub.init(repo_owner='InshaKhan6593', repo_name='Deep-learning-project', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/InshaKhan6593/Deep-learning-project.mlflow")
mlflow.set_experiment("DVC Pipeline")

set_config(transform_output="pandas")

logger = logging.getLogger("evaluate_model")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

def load_keras_model(model_path):
    model = keras.models.load_model(model_path, compile=False)
    return model

def save_run_information(run_id, artifact_path, model_uri, path):
    run_information = {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "model_uri": model_uri
    }
    with open(path, "w") as f:
        json.dump(run_information, f, indent=4)

if __name__ == "__main__":
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    
    train_data_path = root_path / "data/processed/train.csv"
    test_data_path = root_path / "data/processed/test.csv"
    
    df = pd.read_csv(test_data_path, parse_dates=["tpep_pickup_datetime"])
    logger.info("Data read successfully")
    
    df.set_index("tpep_pickup_datetime", inplace=True)
    
    X_test = df.drop(columns=["total_pickups"])
    y_test = df["total_pickups"].values
    
    encoder_path = root_path / "models/encoder.joblib"
    encoder = joblib.load(encoder_path)
    logger.info("Encoder loaded successfully")
    
    X_test_encoded = encoder.transform(X_test)
    logger.info("Data transformed successfully")
    
    if isinstance(X_test_encoded, pd.DataFrame):
        X_test_encoded = X_test_encoded.values
    
    total_features = X_test_encoded.shape[1]
    X_test_reshaped = X_test_encoded.reshape(X_test_encoded.shape[0], 1, total_features)
    logger.info(f"Data reshaped for GRU: {X_test_reshaped.shape}")
    
    model_path = root_path / "models/model.keras"
    model = load_keras_model(model_path)
    logger.info("Model loaded successfully")
    
    y_pred = model.predict(X_test_reshaped, verbose=0).flatten()
    logger.info(f"Predictions made: {y_pred.shape}")
    
    test_mape = mean_absolute_percentage_error(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_pred)
    
    within_10pct = np.mean(np.abs(y_test - y_pred) / y_test <= 0.1) * 100
    within_20pct = np.mean(np.abs(y_test - y_pred) / y_test <= 0.2) * 100
    
    logger.info(f"Test MAPE: {test_mape:.4f}")
    logger.info(f"Test MAE: {test_mae:.4f}")
    logger.info(f"Test R²: {test_r2:.4f}")
    
    metadata_path = root_path / "models/model_metadata.json"
    with open(metadata_path, 'r') as f:
        model_params = json.load(f)
    
    with mlflow.start_run(run_name="model") as run:
        mlflow.log_params({
            "model_name": model_params['model_name'],
            "gru_units": str(model_params['gru_units']),
            "dense_units": str(model_params['dense_units']),
            "dropout": model_params['dropout'],
            "learning_rate": model_params['learning_rate'],
            "batch_size": model_params['batch_size'],
            "epochs_trained": model_params['epochs_trained'],
            "total_parameters": model_params['total_parameters']
        })
        
        mlflow.log_metrics({
            "test_mape": test_mape,
            "test_mae": test_mae,
            "test_mse": test_mse,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "within_10pct": within_10pct,
            "within_20pct": within_20pct,
            "train_loss_mape": model_params['final_train_loss'],
            "val_loss_mape": model_params['final_val_loss'],
            "train_mae": model_params['final_train_mae'],
            "val_mae": model_params['final_val_mae']
        })
        
        training_data = mlflow.data.from_pandas(
            pd.read_csv(train_data_path, parse_dates=["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime"), 
            targets="total_pickups"
        )
        
        validation_data = mlflow.data.from_pandas(
            pd.read_csv(test_data_path, parse_dates=["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime"), 
            targets="total_pickups"
        )
        
        mlflow.log_input(training_data, "training")
        mlflow.log_input(validation_data, "validation")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_model_path = Path(tmp_dir) / "model.keras"
            tmp_encoder_path = Path(tmp_dir) / "encoder.joblib"
            tmp_metadata_path = Path(tmp_dir) / "model_metadata.json"
            
            shutil.copy(model_path, tmp_model_path)
            shutil.copy(encoder_path, tmp_encoder_path)
            shutil.copy(metadata_path, tmp_metadata_path)
            
            mlflow.log_artifact(str(tmp_model_path), artifact_path="model")
            mlflow.log_artifact(str(tmp_encoder_path), artifact_path="model")
            mlflow.log_artifact(str(tmp_metadata_path), artifact_path="model")
            
            logger.info("Model artifacts logged successfully")
        
        run_id = run.info.run_id
        artifact_path = "model"
        model_uri = f"runs:/{run_id}/{artifact_path}"
        
        logger.info("MLflow logging complete")
    
    json_file_save_path = root_path / "run_information.json"
    save_run_information(
        run_id=run_id,
        artifact_path=artifact_path,
        model_uri=model_uri,
        path=json_file_save_path
    )
    logger.info("Run information saved successfully")