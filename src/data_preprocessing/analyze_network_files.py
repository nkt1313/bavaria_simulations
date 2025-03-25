
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

Output:
- Static PNG maps for each city's road network
- Network statistics in console output

Usage:
python analyze_network_files.py

Directory structure expected:
data/
├── simulation_data_per_city/
│   ├── city1/
│   │   └── city1_network.xml.gz
│   ├── city2/
│   │   └── city2_network.xml.gz
│   └── ...
└── network_analysis/
    ├── city1_network.png
    ├── city2_network.png
    └── ...
"""

def plot_network(edges_df: pd.DataFrame, city_name: str, output_path: Path):
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

    plt.title(f"{city_name} Road Network")
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

def analyze_networks(base_dir: Path):
    """
    Analyze and plot networks for all cities
    """
    simulations_dir = base_dir / "simulation_data_per_city_new/"
    
    if not Path(simulations_dir).exists():
        raise FileNotFoundError(f"Simulations directory not found: {simulations_dir}")
    
    # Process each city
    for city_dir in Path(simulations_dir).iterdir():
        if not city_dir.is_dir():
            continue
            
        city_name = city_dir.name
        network_file = city_dir / f"{city_name}_network.xml.gz"
        
        if not network_file.exists():
            print(f"Network file not found for {city_name}")
            continue
            
        try:
            print(f"\nProcessing {city_name}...")
            
            # Parse network
            nodes = nio.parse_nodes(network_file)
            edges_df = nio.parse_edges(network_file, nodes)
            
            print(f"Network statistics for {city_name}:")
            print(f"Number of nodes: {len(nodes)}")
            print(f"Number of links: {len(edges_df)}")
            
            # Create output directory
            output_dir =  base_dir / "network_file_per_city"
            output_dir.mkdir(exist_ok=True)
            
            # Create and save static plot
            plot_file = output_dir / f"{city_name}_city_network.png"
            plot_network(edges_df, city_name, plot_file)
            print(f"Network plot saved to: {plot_file}")
            
        except Exception as e:
            print(f"Error processing {city_name}: {e}")
            continue

if __name__ == "__main__":
    base_dir = Path("data/")
    analyze_networks(base_dir)

