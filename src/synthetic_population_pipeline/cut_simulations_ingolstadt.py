import os
import subprocess
import os
import subprocess
import multiprocessing
from pathlib import Path
import geopandas as gpd
import time

'''
This script processes city-specific simulation data by cutting the network for each city using the RunScenarioCutter.
It identifies cities to process by reading city boundary files and ensures that the output for each city is generated correctly.
The script handles cities with multiple and just one polygons differently and verifies the existence and validity of required output files.
It processes cities sequentially, prioritizing smaller files first, and provides detailed logging for each step.
'''

# These cities need extra treatment
cities_with_multiple_polygons = ["bayreuth", "nuernberg"]
cities_with_just_one_polygon = ["kempten", "neuulm", "ingolstadt"]
cities_with_combined_stadt_and_landkreis = ["augsburg",'wuerzburg']

def check_city_output(output_path: Path, city_prefix: str) -> bool:
    """
    Check if the city has already been processed successfully
    """
    if not output_path.exists():
        return False
        
    # Key files that should exist after a successful cut
    required_files = [
        output_path / f"{city_prefix}network.xml.gz",
        output_path / f"{city_prefix}transit_schedule.xml.gz",
        output_path / f"{city_prefix}transit_vehicles.xml.gz",
        output_path / f"{city_prefix}facilities.xml.gz",
        output_path / f"{city_prefix}population.xml.gz",
        output_path / f"{city_prefix}config.xml"
    ]
    
    # Check if all required files exist and are not empty
    all_files_exist = True
    for f in required_files:
        if not f.exists() or f.stat().st_size == 0:
            all_files_exist = False
            break
    
    if all_files_exist:
        print(f"Output for city {city_prefix[:-1]} exists")
        # Print file sizes for verification
        for f in required_files:
            size_mb = f.stat().st_size / (1024 * 1024)  # Convert to MB
            print(f"  {f.name}: {size_mb:.2f} MB")
    
    return all_files_exist



def get_largest_polygon_for_multiple_polygons(gpkg_path: Path) -> Path:
    """
    Key differences and functionalities compared to `get_largest_polygon.py`:
    1. **Targeted Processing**: While `cut_simulations.py` might handle a broader range of cities or scenarios,
    this script is designed to focus on specific cases, potentially indicated by the name "severe cases".
    
    2. **City Selection**: The script identifies cities to process by reading city boundary files. It may include
    additional logic to filter or prioritize cities based on specific criteria relevant to severe cases.

     We used it for the cities of Nuernberg and Bayreuth, whose city boundaries are not just one polygon but multiple ones.
    """
    # Read the geopackage
    gdf = gpd.read_file(gpkg_path)
    
    if len(gdf) < 2:
        raise ValueError(f"Expected at least 2 polygons (city and Landkreis) in {gpkg_path}, but found {len(gdf)}")
    
    # Calculate areas and find the polygons
    gdf['area'] = gdf.geometry.area
    landkreis = gdf.loc[gdf['area'].idxmax()]
    
    # Get all other polygons (cities)
    cities = gdf[gdf['area'] != landkreis['area']]
    
    print(f"\nProcessing {gpkg_path.name}:")
    print(f"Landkreis area: {landkreis['area']:,.2f} square meters")
    
    # Verify that the Landkreis contains all cities and check area ratios
    for _, city in cities.iterrows():
        city_ratio = (city['area'] / landkreis['area']) * 100
        print(f"\nCity information:")
        print(f"  Area: {city['area']:,.2f} square meters")
        print(f"  Ratio to Landkreis: {city_ratio:.2f}%")
        print(f"  Contained within Landkreis: {landkreis.geometry.contains(city.geometry)}")
        
        # Verify the ratio is reasonable (city should be between 5% and 30% of Landkreis)
        if not (5 <= city_ratio <= 30):
            print(f"WARNING: City area ratio ({city_ratio:.2f}%) is outside expected range (5-30%)")
    
    # Create a new GeoDataFrame with just the city polygon
    city_gdf = gpd.GeoDataFrame([cities.iloc[0]], geometry='geometry')
    
    # Clean and validate the polygon
    # 1. First ensure it's valid
    city_gdf.geometry = city_gdf.geometry.buffer(0)
    
    # 2. Get the largest polygon if there are multiple parts
    if len(city_gdf.geometry.iloc[0].geoms) > 1:
        print("Warning: City polygon has multiple parts, using the largest part")
        largest_polygon = max(city_gdf.geometry.iloc[0].geoms, key=lambda p: p.area)
        city_gdf.geometry = gpd.GeoSeries([largest_polygon])
    
    # 3. Ensure it's a single polygon
    if not city_gdf.geometry.iloc[0].is_valid:
        print("Warning: Polygon is not valid, attempting to fix")
        city_gdf.geometry = city_gdf.geometry.buffer(0)
    
    # 4. Ensure it's connected
    if not city_gdf.geometry.iloc[0].is_ring:
        print("Warning: Polygon is not a ring, attempting to fix")
        city_gdf.geometry = city_gdf.geometry.buffer(0)
    
    # Create temporary file with unique name
    temp_gpkg = gpkg_path.parent / f"{gpkg_path.stem}_city.gpkg"
    city_gdf.to_file(temp_gpkg, driver='GPKG')
    
    print(f"\nCreated temporary file with city polygon: {temp_gpkg}")
    print(f"Using city polygon with area: {city_gdf.geometry.area.iloc[0]:,.2f} square meters")
    print(f"Polygon is valid: {city_gdf.geometry.iloc[0].is_valid}")
    print(f"Polygon is connected: {city_gdf.geometry.iloc[0].is_ring}")
    
    return temp_gpkg

def get_full_extent(gpkg_path: Path) -> Path:
    gdf = gpd.read_file(gpkg_path)
    full_union = gdf.unary_union
    full_gdf = gpd.GeoDataFrame(geometry=[full_union], crs=gdf.crs)

    # Properly create the output file name using Path
    temp_file = gpkg_path.parent / f"{gpkg_path.stem}_combined_temp.gpkg"
    full_gdf.to_file(temp_file, driver="GPKG")
    return temp_file
    return temp_file

def get_polygon(gpkg_path: Path, is_for_landkreis: bool = False) -> Path:
    """
    Accepts a .gpkg that may contain a single MultiPolygon representing both Stadt + Landkreis.
    Returns a temporary GPKG for use in network cutting.
    """
    gdf = gpd.read_file(gpkg_path)

    if len(gdf) != 1:
        raise ValueError(f"Expected exactly 1 feature in {gpkg_path}, but found {len(gdf)}")

    geom = gdf.iloc[0].geometry
    print(f"\nProcessing {gpkg_path.name}:")
    print(f"Geometry type: {geom.geom_type}, Area: {geom.area:,.2f} square meters")

    # Create a new GeoDataFrame with the full MultiPolygon
    polygon_gdf = gpd.GeoDataFrame(geometry=[geom], crs=gdf.crs)

    # Save to temporary file
    temp_gpkg = gpkg_path.parent / f"{gpkg_path.stem}_full.gpkg"
    polygon_gdf.to_file(temp_gpkg, driver="GPKG")

    return temp_gpkg


def cut_network_for_city(city: str, base_dir: Path, is_for_landkreis: bool = False) -> None:
    """
    Cut the network for a single city using RunScenarioCutter
    """
    # Convert all paths to absolute paths
    jar_path = base_dir / "src/synthetic_population_pipeline/bavaria/output/bavaria_run.jar"
    config_path = base_dir / "src/synthetic_population_pipeline/bavaria/output/bavaria_config.xml"
    output_path = base_dir / "data" / "simulation_data_per_city_new" / city
    original_extent_path = base_dir / "data" / "city_boundaries" / city / f"{city}.gpkg"
    city_prefix = f"{city}_"

    # Store the temporary file path for cleanup
    temp_extent_path = None

    try:
        if city in cities_with_combined_stadt_and_landkreis:
            print(f"Using full extent (Stadt + Landkreis) for {city}")
            temp_extent_path = get_full_extent(original_extent_path)

        elif city in cities_with_multiple_polygons:
            temp_extent_path = get_largest_polygon_for_multiple_polygons(original_extent_path)

        elif city in cities_with_just_one_polygon:
            temp_extent_path = get_full_extent(original_extent_path)

        else:
            pass
            
        # Ensure all paths are absolute
        jar_path = jar_path.resolve()
        config_path = config_path.resolve()
        output_path = output_path.resolve()
        extent_path = temp_extent_path.resolve()
        
        # Verify the extent file exists and is valid
        if not extent_path.exists():
            raise FileNotFoundError(f"Extent file not found: {extent_path}")
        
        # Verify the extent file contains a valid polygon
        extent_gdf = gpd.read_file(extent_path)
        if len(extent_gdf) != 1:
            raise ValueError(f"Extent file should contain exactly one polygon (city), but contains {len(extent_gdf)}")
        
        print(f"\nUsing city polygon for cutting:")
        print(f"Area: {extent_gdf.geometry.area.iloc[0]:.2f} square meters")
        
        command = [
            "java", "-Xmx240g",
            "-cp", str(jar_path),
            "org.eqasim.core.scenario.cutter.RunScenarioCutter",
            "--config-path", str(config_path),
            "--output-path", str(output_path),
            "--extent-path", str(extent_path),
            "--prefix", f"{city}_",
            "--config:global.numberOfThreads", "24",
            "--config:qsim.numberOfThreads", "24",
            "--config:eqasim.crossingPenalty", "0.0"
        ]
        print(f"Executing command for {city}:")
        print("Paths being used:")
        print(f"JAR: {jar_path}")
        print(f"Config: {config_path}")
        print(f"Output: {output_path}")
        print(f"Extent: {extent_path}")
        print("\nFull command:")
        print(" ".join(command))
        
        print(f"Starting processing for {city}")
        print(f"Output will be written to: {output_path}")
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        subprocess.check_call(command)
                
        # Verify output was created successfully
        if check_city_output(output_path, city_prefix):
            print(f"Successfully processed {city}")
        else:
            print(f"Warning: Some output files are missing for {city}")
    finally:
        # Clean up temporary file if it exists
        if temp_extent_path and temp_extent_path.exists():
            try:
                temp_extent_path.unlink()
                print(f"Cleaned up temporary file: {temp_extent_path}")
            except Exception as e:
                print(f"Warning: Failed to clean up temporary file {temp_extent_path}: {e}")

def main():
    base_dir = Path(__file__).parent.parent.parent
    
    #Process Augsburg city
    #print("\nProcessing Augsburg city:")
    #cut_network_for_city("augsburg", base_dir, is_for_landkreis=False)
    
    #Process Augsburg Landkreis
    #print("\nProcessing Augsburg Landkreis:")
    #cut_network_for_city("augsburg_landkreis", base_dir, is_for_landkreis=True)
    
   # print("\nProcessing entire Augsburg city:")
    #cut_network_for_city("augsburg", base_dir, is_for_landkreis=True)
    
    #process for Ingolstadt
    print("\nProcessing Ingolstadt city:")
    cut_network_for_city("ingolstadt", base_dir, is_for_landkreis=False)

if __name__ == "__main__":
    main()
