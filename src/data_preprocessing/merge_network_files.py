'''
The goal of this script is to merge the network files with capacity reductions of the city with the network files for only the landkreis.

We want to find out whether we can restore the whole network of the Landkreis, with capacity reductions in the cities. 

This will then be the input for the agent-based simulation.

We do this exemplary for Augsburg first. The Landkreis network file is found in "simulation_data_per_city/augsburg/augsburg_network.xml.gz" and the City Network file is found in "simulation_data_per_city_new/augsburg/augsburg_network.xml.gz"

Note: The network files for the cities are not included in the repository. Further: For the cities where the city equals the landkreis (Ingolstadt, Neu-Ulm and Landshut), we don't need to do this.

Important will also be, how hard is the merging. Does it make sense to perform the merging just before starting the simulation, or should we add it as a further step?

Usage:
1. Merge network files:
   a. Process a single city:
      python merge_network_files.py --city augsburg
   b. Process specific cities:
      python merge_network_files.py --cities augsburg munich regensburg
   c. Process all cities:
      python merge_network_files.py

2. Analyze merged networks:
   a. Analyze a single merged network:
      python analyze_network_files.py --city augsburg --merged
   b. Analyze specific merged networks:
      python analyze_network_files.py --cities augsburg munich regensburg --merged
   c. Analyze all merged networks:
      python analyze_network_files.py --merged

The merged networks will be saved in: /hppfs/work/pn39mu/ge49wav3/mount_point_work_dir/bavaria-pipeline/data/merged_networks/{city}/{city}_merged_network.xml.gz
'''

import os
import subprocess
from pathlib import Path
import shutil
import gzip
import xml.etree.ElementTree as ET
import logging
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cities where city equals Landkreis (no merging needed)
CITIES_WITH_ONE_POLYGON = ["kempten", "neuulm", "ingolstadt"]

def decompress_gz(gz_path: Path) -> Path:
    """
    Decompress a .xml.gz file and return the path to the decompressed file
    """
    if not gz_path.suffixes == ['.xml', '.gz']:
        raise ValueError(f"Expected .xml.gz file, got {gz_path}")
    output_path = gz_path.with_suffix('').with_suffix('')  # Remove both .gz and .xml
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return output_path

def compress_to_gz(file_path: Path) -> Path:
    """
    Compress a file to .xml.gz format and return the path to the compressed file
    """
    gz_path = file_path.with_suffix('.xml.gz')
    with open(file_path, 'rb') as f_in:
        with gzip.open(gz_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return gz_path

def merge_networks(city: str, base_dir: Path) -> None:
    """
    Merge the city network (with capacity reductions) with the Landkreis network
    """
    # Skip cities where city equals Landkreis
    if city in CITIES_WITH_ONE_POLYGON:
        logger.info(f"Skipping {city} - city equals Landkreis, no merging needed")
        return

    # Define paths
    city_network = base_dir / "data" / "simulation_data_per_city_new" / city / f"{city}_network.xml.gz"
    landkreis_network = base_dir / "data" / "simulation_data_per_city" / city / f"{city}_network.xml.gz"
    output_dir = base_dir / "data" / "merged_networks" / city
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify input files exist and have correct extension
    if not city_network.exists() or city_network.suffixes != ['.xml', '.gz']:
        raise FileNotFoundError(f"City network file not found or invalid format: {city_network}")
    if not landkreis_network.exists() or landkreis_network.suffixes != ['.xml', '.gz']:
        raise FileNotFoundError(f"Landkreis network file not found or invalid format: {landkreis_network}")
    
    logger.info(f"Processing {city}")
    logger.info(f"City network: {city_network}")
    logger.info(f"Landkreis network: {landkreis_network}")
    
    try:
        # Decompress both network files
        city_network_decompressed = decompress_gz(city_network)
        landkreis_network_decompressed = decompress_gz(landkreis_network)
        
        # Parse both networks
        city_tree = ET.parse(city_network_decompressed)
        landkreis_tree = ET.parse(landkreis_network_decompressed)
        
        city_root = city_tree.getroot()
        landkreis_root = landkreis_tree.getroot()
        
        # Validate XML structure
        if city_root.tag != 'network' or landkreis_root.tag != 'network':
            raise ValueError("Invalid network XML structure")
        
        # Create a mapping of node IDs to their coordinates in the city network
        city_nodes = {node.get('id'): (float(node.get('x')), float(node.get('y'))) 
                     for node in city_root.findall('.//node')}
        
        # Create a mapping of link IDs to their attributes in the city network
        city_links = {link.get('id'): link.attrib 
                     for link in city_root.findall('.//link')}
        
        # Log network statistics
        logger.info(f"City network: {len(city_nodes)} nodes, {len(city_links)} links")
        logger.info(f"Landkreis network: {len(landkreis_root.findall('.//node'))} nodes, {len(landkreis_root.findall('.//link'))} links")
        
        # Update nodes in Landkreis network with city network data
        nodes_updated = 0
        for node in landkreis_root.findall('.//node'):
            node_id = node.get('id')
            if node_id in city_nodes:
                # Update coordinates with city network data
                node.set('x', str(city_nodes[node_id][0]))
                node.set('y', str(city_nodes[node_id][1]))
                nodes_updated += 1
        
        # Update links in Landkreis network with city network data
        links_updated = 0
        for link in landkreis_root.findall('.//link'):
            link_id = link.get('id')
            if link_id in city_links:
                # Update link attributes with city network data
                for key, value in city_links[link_id].items():
                    link.set(key, value)
                links_updated += 1
        
        logger.info(f"Updated {nodes_updated} nodes and {links_updated} links")
        
        # Validate merged network
        if not landkreis_root.findall('.//node') or not landkreis_root.findall('.//link'):
            raise ValueError("Merged network is empty")
        
        # Save merged network
        output_network = output_dir / f"{city}_merged_network.xml"
        landkreis_tree.write(output_network, encoding='utf-8', xml_declaration=True)
        
        # Compress the merged network
        output_network_gz = compress_to_gz(output_network)
        
        # Clean up temporary files
        city_network_decompressed.unlink()
        landkreis_network_decompressed.unlink()
        output_network.unlink()
        
        logger.info(f"Successfully merged networks for {city}")
        logger.info(f"Output saved to: {output_network_gz}")
        
    except Exception as e:
        logger.error(f"Error processing {city}: {e}")
        raise

def main():
    base_dir = Path("/hppfs/work/pn39mu/ge49wav3/mount_point_work_dir/bavaria-pipeline/")
    
    # Get list of all available cities
    cities_dir = base_dir / "data" / "city_boundaries"
    all_cities = [d.name for d in cities_dir.iterdir() 
                  if d.is_dir() and not d.name.startswith('._')]
    
    # Get command line arguments
    parser = argparse.ArgumentParser(description='Merge city and Landkreis network files')
    parser.add_argument('--city', type=str, help='Process a single city')
    parser.add_argument('--cities', nargs='+', help='Process specific cities')
    parser.add_argument('--analyze', action='store_true', help='Run network analysis after merging')
    args = parser.parse_args()
    
    # Determine which cities to process
    if args.city:
        cities = [args.city]
    elif args.cities:
        cities = args.cities
    else:
        cities = all_cities
    
    # Validate cities
    invalid_cities = [city for city in cities if city not in all_cities]
    if invalid_cities:
        logger.error(f"Invalid cities specified: {invalid_cities}")
        logger.error(f"Available cities: {all_cities}")
        return
    
    logger.info(f"Processing cities: {cities}")
    logger.info(f"Results will be saved in: {base_dir}/data/merged_networks/")
    
    # Process each city
    for city in cities:
        try:
            merge_networks(city, base_dir)
            
            # Run network analysis if requested
            if args.analyze:
                logger.info(f"Running network analysis for {city}")
                analyze_cmd = [
                    "python", "src/data_preprocessing/analyze_network_files.py",
                    "--city", city,
                    "--merged"
                ]
                subprocess.run(analyze_cmd, check=True)
                
        except Exception as e:
            logger.error(f"Failed to process {city}: {e}")
            continue

if __name__ == "__main__":
    main()


