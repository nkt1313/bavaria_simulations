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
from network_io import parse_nodes, parse_edges

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def matsim_network_to_gdf(network_path: Path) -> tuple:
    """
    Convert MATSim network XML file to GeoDataFrame.
    Returns tuple of (nodes_dict, df_edges, gdf)
    """
    # Parse nodes and edges using network_io
    nodes_dict = parse_nodes(network_path)
    df_edges = parse_edges(network_path)
    
    # Create GeoDataFrame from edges
    gdf = gpd.GeoDataFrame(df_edges, geometry='geometry', crs='EPSG:25832')
    
    return nodes_dict, df_edges, gdf

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
            nodes_dict, df_edges, gdf = matsim_network_to_gdf(network_path)
            
            # Calculate statistics
            stats = {
                'city': city,
                'num_nodes': len(nodes_dict),
                'num_links': len(df_edges),
                'total_length_km': df_edges['length'].sum() / 1000,  # Convert to km
                'avg_link_length_m': df_edges['length'].mean(),
                'max_link_length_m': df_edges['length'].max(),
                'min_link_length_m': df_edges['length'].min(),
                'total_capacity': df_edges['capacity'].sum(),
                'avg_capacity': df_edges['capacity'].mean(),
                'num_freespeed_links': len(df_edges[df_edges['freespeed'] > 0]),
                'avg_freespeed': df_edges['freespeed'].mean(),
                'num_lanes': df_edges['lanes'].sum(),
                'avg_lanes': df_edges['lanes'].mean()
            }
            
            # Add road type distribution if available
            if 'highway' in df_edges.columns:
                road_types = df_edges['highway'].value_counts()
                for road_type, count in road_types.items():
                    stats[f'num_{road_type}'] = count
                    stats[f'pct_{road_type}'] = (count / len(df_edges)) * 100
            
            results.append(stats)
            
            # Create and save plot for this city
            if 'highway' in df_edges.columns:
                # Create figure
                fig, ax = plt.subplots(figsize=(15, 10))
                
                # Plot network in gray
                gdf.plot(ax=ax, color='gray', linewidth=0.5)
                
                ax.set_title(f'Road Network - {city.capitalize()}')
                ax.axis('off')
                
                # Adjust layout and save
                plt.tight_layout()
                plot_path = city_dir / f"{city}_network_analysis.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved plot to: {plot_path}")
            
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

