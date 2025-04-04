import gzip
from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import xml.etree.ElementTree as ET

'''
This script is used to analyze the network files for each city obtained from the cut_simulation_general.py script.
It plots the network with the city boundary (optional) and saves the plot to a file.

input : city_name_network.xml.gz and city_name.json
output : city_name_network_plot.png

Directory structure:
src/
├── data_preprocessing
│   ├── analyze_network_files.py
│   └── ...

data/
├── simulation_data_per_city_new/
│   ├── city1/
│   │   └── city1_network.xml.gz
│   └── ...
└── entire_network_plots/
    ├── city1_network_plot.png
    └── ...
'''

# list of cities to analyze
cities = ['augsburg', 'nuernberg', 'regensburg', 'ingolstadt', 'fuerth', 'wuerzburg', 'erlangen', 'bamberg', 'landshut', 
              'bayreuth', 'aschaffenburg', 'kempten','rosenheim','schweinfurt','muenchen','neuulm'] 

def parse_network(network_path: Path):
    """
    Parses a MATSim network .xml.gz file into GeoDataFrames for nodes and links.
    """
    with gzip.open(network_path, 'rb') as f:
        tree = ET.parse(f)
    root = tree.getroot()

    # Parse nodes
    nodes = [{
        'id': node.attrib['id'],
        'x': float(node.attrib['x']),
        'y': float(node.attrib['y'])
    } for node in root.find('nodes')]

<<<<<<< HEAD
    nodes_df = pd.DataFrame(nodes)
    nodes_gdf = gpd.GeoDataFrame(
        nodes_df, 
        geometry=gpd.points_from_xy(nodes_df.x, nodes_df.y), 
        crs="EPSG:25832"
    ).set_index('id')

    # Parse links
    links = []
    for link in root.find('links'):
        try:
            from_node = nodes_gdf.loc[link.attrib['from']]
            to_node = nodes_gdf.loc[link.attrib['to']]
            links.append({
                'id': link.attrib['id'],
                'geometry': LineString([(from_node.x, from_node.y), (to_node.x, to_node.y)])
            })
        except KeyError:
            continue

    links_gdf = gpd.GeoDataFrame(links, geometry='geometry', crs="EPSG:25832")
    return nodes_gdf.reset_index(), links_gdf
=======
def analyze_networks(base_dir: Path, cities: list = None, is_merged: bool = False, is_for_landkreis: bool = False):
    """
    Analyze and plot networks for specified cities or Landkreise
    
    Args:
        base_dir: Base directory for all paths
        cities: List of cities to process
        is_merged: Whether to analyze merged networks
        is_for_landkreis: Whether to analyze Landkreis networks
    """
    # Validate base directory
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    
    # Set up data directory based on network type and entity type
    if is_merged:
        data_dir = base_dir / "data" / "merged_networks"
        file_pattern = "{}_merged_network.xml.gz"
    else:
        data_dir = base_dir / "data" / "simulation_data_per_city_new"
        file_pattern = "{}_network.xml.gz"
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # If no cities specified, process all cities in the directory
    if not cities:
        cities = [d.name for d in data_dir.iterdir() if d.is_dir()]
        if not cities:
            logger.warning("No cities found in directory")
            return
    
    # Create output directory
    output_dir = base_dir / "data" / "network_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.exists():
        raise FileNotFoundError(f"Failed to create output directory: {output_dir}")
    
    # Process each city or Landkreis
    for entity_name in cities:
        entity_dir = data_dir / entity_name
        if not entity_dir.exists():
            logger.warning(f"Directory not found for {entity_name}")
            continue
            
        network_file = entity_dir / file_pattern.format(entity_name)
        
        if not network_file.exists():
            logger.warning(f"Network file not found for {entity_name}: {network_file}")
            continue
            
        try:
            logger.info(f"\nProcessing {entity_name}...")
            
            # Parse network
            nodes = nio.parse_nodes(network_file)
            edges_df = nio.parse_edges(network_file, nodes)
            
            if len(nodes) == 0 or len(edges_df) == 0:
                logger.warning(f"Empty network for {entity_name}")
                continue
            
            # Add additional analysis for Landkreis networks
            if is_for_landkreis:
                # Analyze city vs non-city edges
                city_edges = edges_df[edges_df['scenario_edge'] == 'true']
                non_city_edges = edges_df[edges_df['scenario_edge'] == 'false']
                
                logger.info(f"Network statistics for {entity_name}:")
                logger.info(f"Total nodes: {len(nodes)}")
                logger.info(f"Total links: {len(edges_df)}")
                logger.info(f"City links: {len(city_edges)}")
                logger.info(f"Non-city links: {len(non_city_edges)}")
            
            # Create and save static plot
            plot_name = f"{entity_name}_{'merged_' if is_merged else ''}network.png"
            plot_file = output_dir / plot_name
            plot_network(edges_df, entity_name, plot_file, is_merged)
            logger.info(f"Network plot saved to: {plot_file}")
            
        except Exception as e:
            logger.error(f"Error processing {entity_name}: {e}")
            continue

def main():
    base_dir = Path(__file__).parent.parent.parent
    
    # Process Augsburg city
    print("\nProcessing Augsburg city:")
    cut_network_for_city("augsburg", base_dir, is_for_landkreis=False)
    
    # Process Augsburg Landkreis
    print("\nProcessing Augsburg Landkreis:")
    cut_network_for_city("augsburg_landkreis", base_dir, is_for_landkreis=True)
>>>>>>> bbc18b54c425c0532359264592e72adc317c42dd

if __name__ == "__main__":
    '''
    This is the main function that is used to analyze the network files for each city.
    It plots the network with the city boundary (optional) and saves the plot to a file.
    '''
    for city in cities:
        ####################### Adapt according to the working directory ############################
        base_dir = Path(__file__).parent.parent.parent
        network_path = base_dir / "data" / "simulation_data_per_city_new" / city / f"{city}_network.xml.gz"
        boundary_path = base_dir / "data" / "city_boundaries" / city / f"{city}.json"
        plot_path = base_dir / "data" / "entire_network_plots" / f"{city}_network_plot.png"
        #############################################################################################
        # Parse network
        nodes, links = parse_network(network_path)
        print(f"Parsed {len(nodes)} nodes and {len(links)} links")

        # Load boundary if available
        if boundary_path.exists():
            boundary_gdf = gpd.read_file(boundary_path)
        else:
            boundary_gdf = None
            print("Warning: City boundary file not found. Proceeding without overlay.")

        # Plot and save to file
        fig, ax = plt.subplots(figsize=(10, 10))

        if boundary_gdf is not None:
            boundary_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)

        links.plot(ax=ax, linewidth=0.5, color='gray')
        # nodes.plot(ax=ax, markersize=1, color='red')  # Optional, if you want to plot the nodes

        plt.title(f"{city.capitalize()} Network with City Boundary")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f"Plot saved to: {plot_path}")
