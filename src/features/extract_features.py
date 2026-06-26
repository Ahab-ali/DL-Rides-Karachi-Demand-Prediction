import joblib
import pandas as pd
import logging
from pathlib import Path
from yaml import safe_load
import h3

joblib.parallel_backend('threading')

# create a logger
logger = logging.getLogger("extract_features")
logger.setLevel(logging.INFO)

# attach a console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# make a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


def lat_lng_to_h3(lat, lng, resolution):
    """Convert latitude/longitude to H3 hexagon index"""
    try:
        # h3 library expects (lat, lng) as separate arguments
        return h3.latlng_to_cell(lat, lng, resolution)
    except Exception as e:
        logger.warning(f"Failed to convert lat={lat}, lng={lng}: {e}")
        return None


def read_cluster_input(data_path, chunksize=100000, usecols=["pickup_latitude", "pickup_longitude"]):
    df_reader = pd.read_csv(data_path, chunksize=chunksize, usecols=usecols)
    return df_reader


def read_params(params_path="params.yaml"):
    with open(params_path, "r") as file:
        params = safe_load(file)
    return params


if __name__ == "__main__":
    # current path
    current_path = Path(__file__)
    # set the root path
    root_path = current_path.parent.parent.parent
    # data_path
    data_path = root_path / "data/interim/df_without_outliers.csv"
    
    # read the parameters for H3 resolution
    h3_params = read_params()["extract_features"]["h3"]
    h3_resolution = h3_params["resolution"]
    logger.info(f"Using H3 resolution: {h3_resolution}")
    
    # read the full data
    df_final = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"])
    logger.info("Data read successfully")
    
    # Check for null values in coordinates
    null_coords = df_final[['pickup_latitude', 'pickup_longitude']].isnull().sum()
    logger.info(f"Null coordinates - Latitude: {null_coords['pickup_latitude']}, Longitude: {null_coords['pickup_longitude']}")
    
    # Check coordinate ranges
    logger.info(f"Latitude range: {df_final['pickup_latitude'].min()} to {df_final['pickup_latitude'].max()}")
    logger.info(f"Longitude range: {df_final['pickup_longitude'].min()} to {df_final['pickup_longitude'].max()}")
    
    # Drop rows with null coordinates
    df_final = df_final.dropna(subset=['pickup_latitude', 'pickup_longitude'])
    logger.info(f"Rows after dropping nulls: {len(df_final)}")
    
    # convert coordinates to H3 hexagons
    logger.info("Converting coordinates to H3 hexagons...")
    df_final['region'] = [
        lat_lng_to_h3(lat, lng, h3_resolution) 
        for lat, lng in zip(df_final['pickup_latitude'], df_final['pickup_longitude'])
    ]
    
    # Remove rows where H3 conversion failed
    df_final = df_final[df_final['region'].notna()]
    logger.info(f"Rows after H3 conversion: {len(df_final)}")
    logger.info(f"Created {df_final['region'].nunique()} unique H3 regions")
    
    # drop the latitude and longitude columns from data
    df_final = df_final.drop(columns=["pickup_latitude", "pickup_longitude"])
    logger.info("Latitude and Longitude columns are dropped")
    
    # Ensure datetime index is set properly BEFORE groupby
    df_final['tpep_pickup_datetime'] = pd.to_datetime(df_final['tpep_pickup_datetime'])
    df_final = df_final.set_index('tpep_pickup_datetime')
    logger.info("Datetime index set successfully")
    
    # group the data by region
    region_grp = df_final.groupby("region")
    
    # resample the data in 15 minute intervals
    resampled_data = (
        region_grp['region']
        .resample("15min")
        .count()
    )
    logger.info("Data converted to 15 min intervals successfully")
    resampled_data.name = "total_pickups"
    
    # convert back to df
    resampled_data = resampled_data.reset_index(level=0)
    
    # replace the zeros with an arbitrary value
    epsilon_val = 10
    resampled_data.replace({'total_pickups': {0: epsilon_val}}, inplace=True)
    
    # read the alpha parameters
    ewma_params = read_params()["extract_features"]["ewma"]
    print("Parameters for EWMA are ", ewma_params)
    
    # calculate avg pickups using EWMA
    # dataset with pickup smoothing applied
    resampled_data["avg_pickups"] = (
        resampled_data
        .groupby("region")['total_pickups']
        .ewm(**ewma_params)
        .mean()
        .round()
        .values
    )
    logger.info("Average pickups calculated successfully using EWMA")
    
    # save the data
    save_path = root_path / "data/processed/resampled_data.csv"
    resampled_data.to_csv(save_path, index=True)
    logger.info("Data saved successfully")