import os
import subprocess
import os
import subprocess
import multiprocessing
from pathlib import Path
import geopandas as gpd
import time

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
        print(f"\nSkipping {city_prefix[:-1]} - output already exists:")
        # Print file sizes for verification
        for f in required_files:
            size_mb = f.stat().st_size / (1024 * 1024)  # Convert to MB
            print(f"  {f.name}: {size_mb:.2f} MB")
    
    return all_files_exist


def get_largest_polygon(gpkg_path: Path) -> Path:
    """
    Read the geopackage file and select the city polygon (the smaller polygon).
    Returns the path to the temporary file containing the city polygon.
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
    
    # Create a new GeoDataFrame with just the city polygon (smallest polygon)
    city_gdf = gpd.GeoDataFrame([cities.iloc[0]], geometry='geometry')
    
    # Create temporary file with unique name
    temp_gpkg = gpkg_path.parent / f"{gpkg_path.stem}_city.gpkg"
    city_gdf.to_file(temp_gpkg, driver='GPKG')
    
    print(f"\nCreated temporary file with city polygon: {temp_gpkg}")
    print(f"Using city polygon with area: {cities.iloc[0]['area']:,.2f} square meters")
    
    return temp_gpkg

def cut_network_for_city(city: str, base_dir: Path) -> None:
    """
    Cut the network for a single city using RunScenarioCutter
    """
    # Convert all paths to absolute paths
    jar_path = base_dir / "src/synthetic_population_pipeline/bavaria/output/bavaria_run.jar"
    config_path = base_dir / "src/synthetic_population_pipeline/bavaria/output/bavaria_config.xml"
    output_path = base_dir / "data" / "simulation_data_per_city_new" / city
    original_extent_path = base_dir / "data" / "city_boundaries" / city / f"{city}.gpkg"
    city_prefix = f"{city}_"

    # Check if output already exists before doing any expensive operations
    if check_city_output(output_path, city_prefix):
        print(f"Skipping {city} - output already exists")
        return

    # Process the extent file to get the city polygon
    extent_path = get_largest_polygon(original_extent_path)

    # Ensure all paths are absolute
    jar_path = jar_path.resolve()
    config_path = config_path.resolve()
    output_path = output_path.resolve()
    extent_path = extent_path.resolve()
    
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

def main():
    base_dir = Path("/hppfs/work/pn39mu/ge49wav3/mount_point_work_dir/bavaria-pipeline/")
    cities_dir = base_dir / "data" / "city_boundaries"
    print(f"Looking for cities in: {cities_dir}")
    
    # Get list of cities and sort by input file size
    cities = []
    for city_dir in cities_dir.iterdir():
        if not city_dir.is_dir() or city_dir.name.startswith('._'):
            continue
            
        gpkg_file = city_dir / f"{city_dir.name}.gpkg"
        if gpkg_file.exists():
            print(f"Found city: {city_dir.name} with file: {gpkg_file}")
            try:
                # Verify the GPKG file contains the correct polygons
                gdf = gpd.read_file(gpkg_file)
                if len(gdf) < 2:
                    print(f"WARNING: {city_dir.name} GPKG file has less than 2 polygons. Skipping.")
                    continue
                    
                # Check if output already exists before adding to processing list
                output_path = base_dir / "data" / "simulation_data_per_city_new" / city_dir.name
                if check_city_output(output_path, f"{city_dir.name}_"):
                    print(f"Skipping {city_dir.name} - output already exists")
                    continue
                    
                file_size = gpkg_file.stat().st_size
                cities.append((city_dir.name, file_size))
                print(f"Added {city_dir.name} with size {file_size}")
            except Exception as e:
                print(f"Error processing {city_dir.name}: {e}")
                continue    
    
    if not cities:
        print("ERROR: No valid cities found to process!")
        return
        
    print(f"\nCities found: {cities}")
    
    # Sort cities by file size (process smaller files first)
    cities.sort(key=lambda x: x[1])
    cities = [city for city, _ in cities]
    
    print(f"\nFound {len(cities)} cities to process: {cities}")
    print("Processing order (smallest to largest):")
    for i, city in enumerate(cities, 1):
        print(f"{i}. {city}")
    
    # Process cities sequentially
    for city in cities:
        try:
            print(f"\n{'='*50}")
            print(f"Processing city: {city}")
            print(f"{'='*50}")
            cut_network_for_city(city, base_dir)
            # Give system time to clean up between cities
            time.sleep(30)
        except Exception as e:
            print(f"Failed to process {city}: {e}")
            continue

if __name__ == "__main__":
    main()