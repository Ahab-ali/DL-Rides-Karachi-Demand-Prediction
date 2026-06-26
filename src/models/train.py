import pandas as pd
import numpy as np
import joblib
import logging
import yaml
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn import set_config
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json

set_config(transform_output="pandas")

logger = logging.getLogger("train_model")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

def load_params(params_path):
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    return params

def save_model(model, save_path):
    if not str(save_path).endswith('.keras'):
        save_path = str(save_path).replace('.joblib', '.keras').replace('.h5', '.keras')
    model.save(save_path)
    logger.info(f"Model saved to {save_path}")

def mape_loss(y_true, y_pred):
    epsilon = 1e-10
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    diff = tf.abs((y_true - y_pred) / tf.maximum(tf.abs(y_true), epsilon))
    return 100.0 * tf.reduce_mean(diff)

def build_gru_model(input_shape, gru_units, dense_units, dropout, learning_rate):
    model = keras.Sequential()
    
    for i, units in enumerate(gru_units):
        return_sequences = (i < len(gru_units) - 1)
        model.add(layers.GRU(
            units=units,
            return_sequences=return_sequences,
            input_shape=input_shape if i == 0 else None
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout))
    
    for units in dense_units:
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.Dropout(dropout * 0.5))
    
    model.add(layers.Dense(1))
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=mape_loss, metrics=['mae'])
    
    return model

if __name__ == "__main__":
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    
    params_path = root_path / "params.yaml"
    params = load_params(params_path)
    
    model_name = params['train']['model_name']
    gru_units = params['train']['gru_units']
    dense_units = params['train']['dense_units']
    dropout = params['train']['dropout']
    learning_rate = params['train']['learning_rate']
    batch_size = params['train']['batch_size']
    epochs = params['train']['epochs']
    
    logger.info(f"Loaded parameters from {params_path}")
    logger.info(f"Model: {model_name}")
    logger.info(f"GRU units: {gru_units}")
    logger.info(f"Dense units: {dense_units}")
    logger.info(f"Dropout: {dropout}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Batch size: {batch_size}")
    
    data_path = root_path / "data/processed/train.csv"
    df = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"])
    logger.info("Data read successfully")
    
    df.set_index("tpep_pickup_datetime", inplace=True)
    
    X_train = df.drop(columns=["total_pickups"])
    y_train = df["total_pickups"].values
    logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    
    encoder = ColumnTransformer([
        ("ohe", OneHotEncoder(drop="first", sparse_output=False), ["region", "day_of_week"])
    ], remainder="passthrough", n_jobs=-1, force_int_remainder_cols=False)
    
    encoder.fit(X_train)
    
    encoder_save_path = root_path / "models/encoder.joblib"
    joblib.dump(encoder, encoder_save_path)
    logger.info(f"Encoder saved to {encoder_save_path}")
    
    X_train_encoded = encoder.fit_transform(X_train)
    logger.info("Data encoded successfully")
    
    if isinstance(X_train_encoded, pd.DataFrame):
        X_train_encoded = X_train_encoded.values
    
    total_features = X_train_encoded.shape[1]
    X_train_reshaped = X_train_encoded.reshape(X_train_encoded.shape[0], 1, total_features)
    logger.info(f"Data reshaped for GRU: {X_train_reshaped.shape}")
    
    input_shape = (1, total_features)
    model = build_gru_model(
        input_shape=input_shape,
        gru_units=gru_units,
        dense_units=dense_units,
        dropout=dropout,
        learning_rate=learning_rate
    )
    
    logger.info("Model architecture:")
    model.summary(print_fn=lambda x: logger.info(x))
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1)
    ]
    
    logger.info("Starting model training...")
    history = model.fit(
        X_train_reshaped, 
        y_train,
        validation_split=0.2,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info(f"Model trained successfully in {len(history.history['loss'])} epochs")
    
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_train_mae = history.history['mae'][-1]
    final_val_mae = history.history['val_mae'][-1]
    
    logger.info(f"Final Training Loss (MAPE): {final_train_loss:.4f}")
    logger.info(f"Final Validation Loss (MAPE): {final_val_loss:.4f}")
    logger.info(f"Final Training MAE: {final_train_mae:.4f}")
    logger.info(f"Final Validation MAE: {final_val_mae:.4f}")
    
    model_save_path = root_path / "models/model.keras"
    save_model(model, model_save_path)
    
    history_save_path = root_path / "models/training_history.json"
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_save_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    logger.info(f"Training history saved to {history_save_path}")
    
    metadata = {
        'model_name': model_name,
        'gru_units': gru_units,
        'dense_units': dense_units,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs_trained': len(history.history['loss']),
        'final_train_loss': float(final_train_loss),
        'final_val_loss': float(final_val_loss),
        'final_train_mae': float(final_train_mae),
        'final_val_mae': float(final_val_mae),
        'total_parameters': model.count_params()
    }
    
    metadata_save_path = root_path / "models/model_metadata.json"
    with open(metadata_save_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Model metadata saved to {metadata_save_path}")
    
    logger.info("✅ Training pipeline completed successfully!")