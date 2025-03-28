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

import geopandas as gpd
import networkx as nx
import folium
from pathlib import Path
import xml.etree.ElementTree as ET
import gzip
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import network_io as nio
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def plot_network(edges_df: pd.DataFrame, city_name: str, output_path: Path, is_merged: bool = False):
    """
    Create a static matplotlib plot of the network
    """
    plt.figure(figsize=(15, 15))
    
    # Plot different road types in different colors
    for _, row in edges_df.iterrows():
        coords = list(row['geometry'].coords)
        x_coords = [x for x, y in coords]
        y_coords = [y for x, y in coords]
        
        # Color scheme based on road properties
        freespeed = float(row.get('freespeed', 0))
        capacity = float(row.get('capacity', 0))
        
        if freespeed > 30:  # Major roads
            plt.plot(x_coords, y_coords, 'r-', linewidth=1, alpha=0.6)
        elif capacity > 1000:  # Medium roads
            plt.plot(x_coords, y_coords, 'g-', linewidth=1, alpha=0.4)
        else:  # Minor roads
            plt.plot(x_coords, y_coords, 'black', linewidth=1, alpha=0.2)

    title = f"{city_name} {'Merged ' if is_merged else ''}Road Network"
    plt.title(title)
    plt.axis('equal')
    plt.grid(True, alpha=0.2)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='r', linewidth=1.5, label='Major Roads'),
        Line2D([0], [0], color='blue', linewidth=1, label='Medium Roads'),
        Line2D([0], [0], color='black', linewidth=0.5, label='Minor Roads')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

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

if __name__ == "__main__":
    main()

