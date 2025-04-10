import geopandas as gpd
import zipfile
import pyogrio
import numpy as np
import pandas as pd
import glob, os

"""
This stage loads the raw data from the Bavarian building registry.
"""

def configure(context):
    context.config("data_path")
    context.config("bavaria.buildings_path", "bavaria/buildings")
    
    context.stage("bavaria.data.spatial.iris")

def execute(context):
    df_zones = context.stage("bavaria.data.spatial.iris")
    df_combined = []
    
    # Debug paths
    data_path = context.config("data_path")
    buildings_path = context.config("bavaria.buildings_path")
    zip_pattern = "{}/{}/*_Hausumringe.zip".format(data_path, buildings_path)
    
    print("\nPath Analysis:")
    print(f"Data path: {data_path}")
    print(f"Buildings path: {buildings_path}")
    print(f"Looking for zip files in: {zip_pattern}")
    
    # Check if any zip files exist
    zip_files = glob.glob(zip_pattern)
    print(f"\nFound {len(zip_files)} zip files:")
    
    start_index = 0
    for path in zip_files:
        region_name = path.split("/")[-1].split("_Haus")[0]
        print(f"\nProcessing {region_name}")
        
        extract_dir = os.path.join(context.path(), region_name)
        os.makedirs(extract_dir, exist_ok=True)
        
        # Extract files
        with zipfile.ZipFile(path) as archive:
            archive.extractall(extract_dir)
        
        # Correct path including the additional directory level
        shp_path = os.path.join(extract_dir, f"{region_name}_Hausumringe", "hausumringe.shp")
        print(f"Looking for shapefile at: {shp_path}")
        
        if not os.path.exists(shp_path):
            print("\nExtracted directory structure:")
            for root, dirs, files in os.walk(extract_dir):
                print(f"Directory: {root}")
                for f in files:
                    print(f"  - {f}")
            raise FileNotFoundError(f"Cannot find hausumringe.shp at: {shp_path}")
        
        # Read the shapefile
        df_buildings = pyogrio.read_dataframe(shp_path, columns=[])[["geometry"]]
        
        # Weighting by area
        df_buildings["weight"] = df_buildings.area

        # Attributes
        df_buildings["building_id"] = np.arange(len(df_buildings)) + start_index
        start_index += len(df_buildings) + 1

        df_buildings["geometry"] = df_buildings.centroid

        # Filter
        df_buildings = df_buildings[
            (df_buildings["weight"] >= 40) & (df_buildings["weight"] < 400)
        ].copy()

        # Impute spatial identifiers
        df_buildings = gpd.sjoin(df_buildings, df_zones[["geometry", "commune_id", "iris_id"]], 
            how = "left", predicate = "within").reset_index(drop = True).drop(columns = ["index_right"])

        df_combined.append(df_buildings[[
            "building_id", "weight", "commune_id", "iris_id", "geometry"
        ]])

    df_combined = gpd.GeoDataFrame(pd.concat(df_combined), crs = df_combined[0].crs)

    required_zones = set(df_zones["commune_id"].unique())
    available_zones = set(df_combined["commune_id"].unique())
    missing_zones = required_zones - available_zones

    if len(missing_zones) > 0:
        print("Adding {} centroids as buildings for missing municipalities".format(len(missing_zones)))

        df_missing = df_zones[df_zones["commune_id"].isin(missing_zones)][["commune_id", "iris_id", "geometry"]].copy()
        df_missing["geometry"] = df_missing["geometry"].centroid
        df_missing["building_id"] = np.arange(len(df_missing)) + start_index
        df_missing["weight"] = 1.0

        df_combined = pd.concat([df_combined, df_missing])
    
    return df_combined

def validate(context):
    total_size = 0

    for path in glob.glob("{}/{}/*_Hausumringe.zip".format(context.config("data_path"), context.config("bavaria.buildings_path"))):
        total_size += os.path.getsize(path)

    if total_size == 0:
        raise RuntimeError("Did not find any building data for Bavaria")

    return total_size
