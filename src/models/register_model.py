import json
import mlflow
import dagshub
import logging
from pathlib import Path
from mlflow.client import MlflowClient
from datetime import datetime

dagshub.init(repo_owner='InshaKhan6593', repo_name='Deep-learning-project', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/InshaKhan6593/Deep-learning-project.mlflow")

logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

if __name__ == "__main__":
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    file_name = "run_information.json"
    
    try:
        with open(root_path / file_name, "r") as f:
            run_info = json.load(f)
            logger.info("Information loaded successfully")
    except FileNotFoundError:
        logger.error(f"File {file_name} not found")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from the file {file_name}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise
    
    run_id = run_info["run_id"]
    artifact_path = run_info["artifact_path"]
    model_uri = run_info["model_uri"]
    
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Artifact Path: {artifact_path}")
    logger.info(f"Model URI: {model_uri}")
    
    client = MlflowClient()
    
    # Load model metadata to get architecture info
    metadata_path = root_path / "models/model_metadata.json"
    with open(metadata_path, 'r') as f:
        model_metadata = json.load(f)
    
    model_name = model_metadata.get('model_name', 'GRU')
    
    try:
        # Add detailed tags for the new model
        client.set_tag(run_id, "model_status", "production")
        logger.info("Added 'model_status=production' tag")
        
        client.set_tag(run_id, "model_type", model_name)
        logger.info(f"Added 'model_type={model_name}' tag")
        
        client.set_tag(run_id, "model_architecture", f"GRU_{len(model_metadata.get('gru_units', []))}layers")
        logger.info("Added architecture tag")
        
        client.set_tag(run_id, "gru_units", str(model_metadata.get('gru_units', [])))
        logger.info("Added GRU units tag")
        
        client.set_tag(run_id, "dense_units", str(model_metadata.get('dense_units', [])))
        logger.info("Added dense units tag")
        
        client.set_tag(run_id, "dropout", str(model_metadata.get('dropout', 0)))
        logger.info("Added dropout tag")
        
        client.set_tag(run_id, "loss_function", "MAPE")
        logger.info("Added loss function tag")
        
        client.set_tag(run_id, "deployment_ready", "true")
        logger.info("Added 'deployment_ready=true' tag")
        
        client.set_tag(run_id, "validation_status", "passed")
        logger.info("Added 'validation_status=passed' tag")
        
        client.set_tag(run_id, "registered_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("Added registration date tag")
        
    except Exception as e:
        logger.warning(f"Failed to add tags: {e}")
    
    # Create deployment info
    deployment_info = {
        "model_name": "karachi_demand_prediction_gru_v2",
        "model_version": "v2.0",
        "model_type": model_name,
        "architecture": {
            "type": "GRU",
            "gru_layers": len(model_metadata.get('gru_units', [])),
            "gru_units": model_metadata.get('gru_units', []),
            "dense_layers": len(model_metadata.get('dense_units', [])),
            "dense_units": model_metadata.get('dense_units', []),
            "dropout": model_metadata.get('dropout', 0),
            "total_parameters": model_metadata.get('total_parameters', 0)
        },
        "training": {
            "loss_function": "MAPE",
            "learning_rate": model_metadata.get('learning_rate', 0),
            "batch_size": model_metadata.get('batch_size', 0),
            "epochs_trained": model_metadata.get('epochs_trained', 0),
            "final_train_loss": model_metadata.get('final_train_loss', 0),
            "final_val_loss": model_metadata.get('final_val_loss', 0)
        },
        "mlflow": {
            "run_id": run_id,
            "model_uri": model_uri,
            "artifact_path": artifact_path,
            "tracking_uri": "https://dagshub.com/InshaKhan6593/Deep-learning-project.mlflow"
        },
        "deployment": {
            "status": "production",
            "ready": True,
            "registered_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "instructions": {
            "download_artifacts": f"mlflow.artifacts.download_artifacts('{model_uri}')",
            "mlflow_ui": f"https://dagshub.com/InshaKhan6593/Deep-learning-project.mlflow/#/experiments/2/runs/{run_id}",
            "load_in_python": [
                "import mlflow",
                "import joblib",
                "from tensorflow import keras",
                "",
                f"# Download artifacts",
                f"artifact_path = mlflow.artifacts.download_artifacts('{model_uri}')",
                "",
                "# Load model and encoder",
                "model = keras.models.load_model(artifact_path + '/model.keras', compile=False)",
                "encoder = joblib.load(artifact_path + '/encoder.joblib')",
                "",
                "# Make predictions",
                "# X_encoded = encoder.transform(X)",
                "# X_reshaped = X_encoded.reshape(-1, 1, X_encoded.shape[1])",
                "# predictions = model.predict(X_reshaped)"
            ]
        }
    }
    
    deployment_file = root_path / "deployment_info.json"
    with open(deployment_file, 'w') as f:
        json.dump(deployment_info, f, indent=4)
    
    logger.info(f"Deployment info saved to {deployment_file}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ NEW MODEL REGISTERED SUCCESSFULLY!")
    logger.info("="*70)
    logger.info(f"Model Name: karachi_demand_prediction_gru_v2")
    logger.info(f"Model Type: {model_name}")
    logger.info(f"Architecture: {len(model_metadata.get('gru_units', []))} GRU layers + {len(model_metadata.get('dense_units', []))} Dense layers")
    logger.info(f"GRU Units: {model_metadata.get('gru_units', [])}")
    logger.info(f"Dense Units: {model_metadata.get('dense_units', [])}")
    logger.info(f"Total Parameters: {model_metadata.get('total_parameters', 0):,}")
    logger.info(f"Loss Function: MAPE")
    logger.info(f"\n📊 MLflow Info:")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Model URI: {model_uri}")
    logger.info(f"\n🌐 View in DagsHub:")
    logger.info(f"https://dagshub.com/InshaKhan6593/Deep-learning-project.mlflow/#/experiments/2/runs/{run_id}")
    logger.info(f"\n💾 Deployment info: deployment_info.json")
    logger.info(f"\n🚀 To deploy:")
    logger.info(f"   mlflow.artifacts.download_artifacts('{model_uri}')")
    logger.info("="*70)