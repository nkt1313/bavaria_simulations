"""
Network Analysis and Visualization Script

This script analyzes and visualizes road networks from MATSim network files (.xml.gz format).
It processes network files for different cities and creates static visualizations of their road networks.

Key functionalities:
1. Reads compressed XML network files containing nodes and links (roads)
2. Parses network structure:
   - Nodes with their coordinates
   - Links (roads) with properties like speed limits and capacity
3. Creates visualizations that distinguish between:
   - Major roads (red, freespeed > 30)
   - Medium roads (blue, capacity > 1000)
   - Minor roads (gray, other roads)
4. Generates statistics about network size (number of nodes and links)

Input:
- Base directory containing city-specific simulation data
- Network files named as '{city_name}_network.xml.gz'
- Optional: Merged network files in 'merged_networks' directory

Output:
- Static PNG maps for each city's road network
- Network statistics in console output

Usage:
1. Analyze regular networks:
   a. Process a single city:
      python analyze_network_files.py --city augsburg
   b. Process specific cities:
      python analyze_network_files.py --cities augsburg munich regensburg
   c. Process all cities:
      python analyze_network_files.py

2. Analyze merged networks:
   a. Process a single merged network:
      python analyze_network_files.py --city augsburg --merged
   b. Process specific merged networks:
      python analyze_network_files.py --cities augsburg munich regensburg --merged
   c. Process all merged networks:
      python analyze_network_files.py --merged

Directory structure expected:
data/
├── simulation_data_per_city/
│   ├── city1/
│   │   └── city1_network.xml.gz
│   └── ...
├── merged_networks/
│   ├── city1/
│   │   └── city1_merged_network.xml.gz
│   └── ...
└── network_analysis/
    ├── city1_network.png
    ├── city1_merged_network.png
    └── ...
"""

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import gzip
import xml.etree.ElementTree as ET
from shapely.geometry import Point, LineString

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_network(network_path: Path) -> tuple:
    """
    Parse MATSim network XML file directly.
    Returns tuple of (nodes_df, links_gdf)
    """
    # Parse the XML
    with gzip.open(network_path, 'rb') as f:
        tree = ET.parse(f)
    root = tree.getroot()

    # Extract nodes
    nodes = []
    for node in root.find('nodes'):
        nodes.append({
            'id': node.attrib['id'],
            'x': float(node.attrib['x']),
            'y': float(node.attrib['y'])
        })

    nodes_df = pd.DataFrame(nodes)
    nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry=gpd.points_from_xy(nodes_df.x, nodes_df.y), crs="EPSG:25832")

    # Extract links
    links = []
    for link in root.find('links'):
        from_id = link.attrib['from']
        to_id = link.attrib['to']
        
        from_node = nodes_df[nodes_df['id'] == from_id].iloc[0]
        to_node = nodes_df[nodes_df['id'] == to_id].iloc[0]
        
        links.append({
            'id': link.attrib['id'],
            'geometry': LineString([(from_node['x'], from_node['y']), (to_node['x'], to_node['y'])])
        })

    links_gdf = gpd.GeoDataFrame(links, geometry='geometry', crs="EPSG:25832")
    
    return nodes_df, links_gdf

def analyze_network_files():
    """
    Analyze network files for each city from the cut_simulations_general.py output.
    Generates and saves plots in the same output directory.
    """
    # Get the base directory (same as in cut_simulations_general.py)
    base_dir = Path(__file__).parent.parent.parent
    
    # Define the cities list (same as in cut_simulations_general.py)
    cities = ['augsburg', 'nuernberg', 'regensburg', 'ingolstadt', 'fuerth', 'wuerzburg', 
              'erlangen', 'bamberg', 'landshut', 'bayreuth', 'aschaffenburg', 'kempten',
              'rosenheim', 'schweinfurt', 'muenchen', 'neuulm']
    
    # Create a DataFrame to store results
    results = []
    
    for city in cities:
        print(f"\nAnalyzing {city}:")
        city_dir = base_dir / "data" / "simulation_data_per_city_new" / city
        network_path = city_dir / f"{city}_network.xml.gz"
        
        if not network_path.exists():
            print(f"Warning: Network file not found for {city}")
            continue
            
        try:
            # Parse the network file
            nodes_df, links_gdf = parse_network(network_path)
            
            # Calculate statistics
            stats = {
                'city': city,
                'num_nodes': len(nodes_df),
                'num_links': len(links_gdf),
                'total_length_km': links_gdf.geometry.length.sum() / 1000,  # Convert to km
                'avg_link_length_m': links_gdf.geometry.length.mean(),
                'max_link_length_m': links_gdf.geometry.length.max(),
                'min_link_length_m': links_gdf.geometry.length.min()
            }
            
            results.append(stats)
            
            # Create and save plot
            fig, ax = plt.subplots(figsize=(15, 15))
            links_gdf.plot(ax=ax, linewidth=0.5, color='gray')
            nodes_df.plot(ax=ax, markersize=1, color='red')
            ax.set_title(f"Network - {city.capitalize()}")
            ax.axis("equal")
            
            # Save plot
            plot_path = city_dir / f"{city}_network_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved plot to: {plot_path}")
            
            # Save individual city DataFrame
            city_df = pd.DataFrame([stats])
            city_df.to_csv(city_dir / f"{city}_network_stats.csv", index=False)
            print(f"Saved city statistics to: {city_dir / f'{city}_network_stats.csv'}")
            
            print(f"Successfully analyzed {city}")
            
        except Exception as e:
            print(f"Error analyzing {city}: {e}")
    
    # Create DataFrame from results
    df_results = pd.DataFrame(results)
    
    # Save results to CSV in the base output directory
    output_path = base_dir / "data" / "network_analysis_results.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total cities analyzed: {len(df_results)}")
    print("\nAverage values across cities:")
    for col in df_results.columns:
        if col != 'city':
            print(f"{col}: {df_results[col].mean():.2f}")
    
    return df_results

if __name__ == "__main__":
    df_results = analyze_network_files()

