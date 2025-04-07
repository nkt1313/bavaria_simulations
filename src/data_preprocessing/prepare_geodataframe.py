import sys
sys.path.append('src/data_preprocessing')
import network_io as nio
import xml.etree.ElementTree as ET
from shapely.geometry import LineString
import network_io as nio
from shapely.geometry import LineString, box  # Add box to the import
import osmnx as ox
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import shapely.geometry as sgeo
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os
import random
import gzip
import pandas as pd
from shapely import wkt
from pathlib import Path
import logging
import networkx as nx
import seaborn as sns
from matplotlib.colors import Normalize
from pathlib import Path
import logging
import networkx as nx
import seaborn as sns
from matplotlib.colors import Normalize
import networkx as nx
import multiprocessing as mp
from itertools import islice
from functools import reduce
from collections import Counter
'''
This script is used to prepare the geodataframe for the network files.
It is used to create the geodataframe for the network files and to prepare the geodataframe for creating the subgraphs.
It also incorporates the openness and betweenness centrality of the nodes into the resulting geodataframe.
'''

#os.chdir(r"C:\Users\nktba\bavaria_simulations") # Set working directory
#base_dir = os.getcwd()
base_dir = Path(__file__).resolve().parent.parent.parent
#administrative_boundary_json_path = os.path.join(base_dir, "data", "output", "boundary_files", "Augsburg.json")
administrative_boundary_json_path = base_dir / "data" / "city_boundaries" / "augsburg" / "augsburg.json"
#matsim_network_file_path = os.path.join(base_dir, "data", "output", "simulation_data_per_city_new","augsburg", "augsburg_network.xml.gz")
#csv_dir = Path(r"C:\Users\nktba\misc\simulation_outputs")
#csv_filepath = csv_dir / f"augsburg"/f"augsburg_seed_1/output_links.csv.gz"
csv_filepath=  base_dir / "data" / "simulation_output" / 'output_links.csv.gz'
#csv_output_path = csv_dir/f"augsburg"/f"augsburg_seed_1/augsburg_output_links.csv"
#output_base_path = os.path.join(base_dir, "data", "output", "network_files")
output_base_path = base_dir / "data"

#Define variables
hexagon_size = 1500  # Size in meters for EPSG:25832 and in degrees for EPSG:4326 **********VERY IMPORTANT********** 
capacity_tuning_factor = 0.5 #This is the factor by which the capacity of the links is reduced

#Define the path to the check output subgraph
#check_output_subgraph_path = os.path.join(base_dir, "data", "output", "network_files", "Augsburg", "networks", "networks_0", "network_residential_n7_s1.xml.gz")

def matsim_network_input_to_gdf(network_file):
    """
    Convert MATSim network XML to GeoDataFrame using network_io
    """
    # Parse nodes and edges using nio
    nodes_dict = nio.parse_nodes(network_file)
    df_edges = nio.parse_edges(network_file, nodes_dict)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df_edges, geometry='geometry', crs='EPSG:25832')
    
    # Ensure correct CRS
    if gdf.crs != "EPSG:25832":
        gdf = gdf.to_crs(epsg=25832)
    
    return gdf,nodes_dict,df_edges

def clean_duplicates_based_on_modes(file_path):
    """
    Cleans the network by:
    1. First removing full duplicates (same from_node, to_node, geometry, modes)
    2. For remaining duplicates with same node pairs but different modes:
       - Keep the entry with modes that contain both "car" and "car_passenger"
       - If multiple or none meet this criteria, keep the one with the longest modes string
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        The cleaned DataFrame
    """
    # Load the CSV file
    print(f"Loading file: {file_path}")
    df = pd.read_csv(file_path, delimiter=";", low_memory=False)
    
    print(f"Total edges in original file: {len(df)}")
    
    # Step 1: First identify duplicates by the key columns
    key_columns = ["from_node", "to_node", "geometry", "modes"]
    full_duplicates_mask = df.duplicated(subset=key_columns, keep=False)
    full_duplicates = df[full_duplicates_mask]
    
    print(f"\nFull duplicates (same from_node, to_node, geometry, modes): {len(full_duplicates)}")
    
    # Now process these duplicates - keep the highest vol_car
    df_cleaned = df[~full_duplicates_mask].copy()
    
    for _, group in full_duplicates.groupby(key_columns):
        # Keep the row with the highest vol_car
        max_idx = group['vol_car'].idxmax()
        row_to_keep = group.loc[max_idx]
        df_cleaned = pd.concat([df_cleaned, pd.DataFrame([row_to_keep])])
    
    print(f"Edges after removing full duplicates (based on higher car volume): {len(df_cleaned)}")
    print(f"Number of full duplicates removed: {len(df) - len(df_cleaned)}")
    
    # Step 2: Now look for edges with the same from_node and to_node (leftover duplicates)
    node_columns = ["from_node", "to_node"]
    leftover_duplicates_mask = df_cleaned.duplicated(subset=node_columns, keep=False)
    leftover_duplicates = df_cleaned[leftover_duplicates_mask]
    
    print(f"\nLeftover duplicates (same from_node, to_node only): {len(leftover_duplicates)}")
    print(f"Number of unique node pairs with leftover duplicates: {len(leftover_duplicates.groupby(node_columns))}")
    
    # Step 3: Process leftover duplicates based on modes criteria
    df_final = df_cleaned[~leftover_duplicates_mask].copy()
    
    # Track selection statistics
    selection_stats = {
        'total_groups': 0,
        'selected_by_car_modes': 0,
        'selected_by_length': 0,
        'selected_by_vol_car': 0
    }
    
    for (from_node, to_node), group in leftover_duplicates.groupby(node_columns):
        selection_stats['total_groups'] += 1
        
        # Convert modes to string and check for both "car" and "car_passenger"
        # (ensuring robust string comparison)
        group['has_both_car_modes'] = group['modes'].apply(
            lambda x: ('"car"' in str(x) or "'car'" in str(x) or ",car," in str(x) or "[car" in str(x)) and 
                      ('"car_passenger"' in str(x) or "'car_passenger'" in str(x) or 
                       ",car_passenger," in str(x) or "[car_passenger" in str(x))
        )
        
        # Check if any entry has both car modes
        if group['has_both_car_modes'].any():
            # Filter to entries with both car modes
            car_mode_entries = group[group['has_both_car_modes']]
            
            # If multiple entries have both car modes, select by longest modes string
            if len(car_mode_entries) > 1:
                mode_lengths = car_mode_entries['modes'].apply(lambda x: len(str(x)))
                
                # If there's a tie in length, use vol_car
                if mode_lengths.nunique() == 1:
                    row_to_keep = car_mode_entries.loc[car_mode_entries['vol_car'].idxmax()]
                    selection_stats['selected_by_vol_car'] += 1
                else:
                    row_to_keep = car_mode_entries.loc[mode_lengths.idxmax()]
                    selection_stats['selected_by_length'] += 1
            else:
                # Only one entry has both car modes - keep it
                row_to_keep = car_mode_entries.iloc[0]
                selection_stats['selected_by_car_modes'] += 1
        else:
            # No entry has both car modes - select by longest modes string
            mode_lengths = group['modes'].apply(lambda x: len(str(x)))
            
            # If there's a tie in length, use vol_car
            if mode_lengths.nunique() == 1:
                row_to_keep = group.loc[group['vol_car'].idxmax()]
                selection_stats['selected_by_vol_car'] += 1
            else:
                row_to_keep = group.loc[mode_lengths.idxmax()]
                selection_stats['selected_by_length'] += 1
        
        # Add the selected row to the final DataFrame
        df_final = pd.concat([df_final, pd.DataFrame([row_to_keep.drop('has_both_car_modes')])])
    
    print(f"\nFinal edges after processing all duplicates: {len(df_final)}")
    print(f"Total edges removed: {len(df) - len(df_final)}")
    
    # Print selection statistics
    print("\nSelection criteria statistics:")
    print(f"Total duplicate groups processed: {selection_stats['total_groups']}")
    print(f"Selected by having both car modes: {selection_stats['selected_by_car_modes']} ({selection_stats['selected_by_car_modes']/selection_stats['total_groups']*100:.1f}%)")
    print(f"Selected by longest modes string: {selection_stats['selected_by_length']} ({selection_stats['selected_by_length']/selection_stats['total_groups']*100:.1f}%)")
    print(f"Selected by highest vol_car (tiebreaker): {selection_stats['selected_by_vol_car']} ({selection_stats['selected_by_vol_car']/selection_stats['total_groups']*100:.1f}%)")
    
    # Step 4: Verify no duplicates remain
    final_duplicates_mask = df_final.duplicated(subset=node_columns, keep=False)
    final_duplicates = df_final[final_duplicates_mask]
    
    if len(final_duplicates) > 0:
        print(f"\n⚠️ WARNING: {len(final_duplicates)} duplicate edges remain!")
        print(f"Number of unique node pairs with remaining duplicates: {len(final_duplicates.groupby(node_columns))}")
        
        # Analyze remaining duplicates
        print("\n=== REMAINING DUPLICATES ===")
        for i, ((from_node, to_node), group) in enumerate(final_duplicates.groupby(node_columns)):
            print(f"\nRemaining Duplicate {i+1}: from_node={from_node}, to_node={to_node}")
            print(f"Number of edges: {len(group)}")
            
            # Display the group
            important_cols = ['modes', 'vol_car', 'geometry']
            if 'link' in group.columns:
                important_cols.insert(0, 'link')
            
            print(group[important_cols].to_string())
            
            # Only show a few examples
            if i >= 4:
                remaining = len(final_duplicates.groupby(node_columns)) - (i+1)
                print(f"\n... and {remaining} more duplicate pairs (not shown) ...")
                break
    else:
        print("\n✅ SUCCESS: No duplicates remain in the final network!")
    
    # Save the cleaned network
    #output_path = file_path.replace('.csv', '_wo_duplicates.csv')
    #df_final.to_csv(output_path, index=False, sep=';')
    #print(f"\nCleaned network saved to: {output_path}")
    df_final=df_final[df_final['from_node']!=df_final['to_node']]
    return df_final

def get_road_types(csv_filepath,csv_output_path,crs='EPSG:25832'):
    """
    Download OSM road network for the given bounds and convert to GeoDataFrame.
    
    Parameters:
        bounds (tuple): Tuple of (minx, miny, maxx, maxy) in the input CRS
        crs (str): Coordinate reference system of the input bounds. Default 'EPSG:25832'
    
    Returns:
        geopandas.GeoDataFrame: OSM road network with columns including:
            - highway: road type classification
            - geometry: LineString geometry of the road segment
            - other OSM attributes (name, maxspeed, lanes, etc.)
    """
    print("Unzipping file...")
    with gzip.open(csv_filepath, 'rb') as f_in:
        with open(csv_output_path, 'wb') as f_out:
            f_out.write(f_in.read())
    print(f"File unzipped to: {csv_output_path}")
        
    # Read CSV
    print("Reading CSV...")
    df = pd.read_csv(csv_output_path, delimiter=';')
    gdf_csv = gpd.GeoDataFrame(
    df,  # Your original DataFrame
    geometry=df['geometry'].apply(wkt.loads),  # Convert WKT strings to geometries
    crs=crs  # Set the coordinate reference system
    )
    return gdf_csv
    
def consolidate_road_types(highway_type):
    """
    Consolidate OSM road types into 5 broad categories
    """
    if highway_type is None:
        return None
    
    # If it's a list, process each type and return the highest hierarchy
    if isinstance(highway_type, list):
        consolidated = [consolidate_road_types(h) for h in highway_type]
        # Remove None values
        consolidated = [h for h in consolidated if h is not None]
        if not consolidated:
            return None
        # Return the highest hierarchy road type
        hierarchy = ['trunk', 'primary', 'secondary', 'tertiary', 'residential']
        for h in hierarchy:
            if h in consolidated:
                return h
        return 'residential'  # default if no match in hierarchy

    # Mapping of road types to broad categories
    road_type_mapping = {
        'motorway': 'trunk',
        'motorway_link': 'trunk',
        'trunk': 'trunk',
        'trunk_link': 'trunk',
        'primary': 'primary',
        'primary_link': 'primary',
        'secondary': 'secondary',
        'secondary_link': 'secondary',
        'tertiary': 'tertiary',
        'tertiary_link': 'tertiary',
        'residential': 'residential',
        'unclassified': 'residential',
    }
    
    return road_type_mapping.get(highway_type, 'residential')


def modify_districts_geodataframe(gdf):
    '''
    This function modifies the districts geodataframe to ensure it is in the correct CRS and has the correct zone_id
    '''
    if (gdf.geometry.apply(lambda x: x.geom_type == "MultiPolygon")).any():
        gdf["geometry"] = gdf.geometry.apply(multipolygon_to_polygon)
    gdf["area"] = gdf.geometry.area
    gdf["perimetre"] = gdf.geometry.length
    gdf["zone_id"] = range(1, len(gdf)+1) #zone id
    districts_gdf = gdf[["zone_id", "area", "perimetre", "geometry"]]
    # Ensure the data is in the correct CRS (EPSG:25832) **********VERY IMPORTANT**********
    if districts_gdf.crs != "EPSG:25832": #should match with the CRS of the Network Geodataframe
        districts_gdf = districts_gdf.to_crs(epsg=25832)
    return districts_gdf


def multipolygon_to_polygon(geom):
    '''
    This function converts a MultiPolygon to a Polygon with the largest area.
    '''
    return max(geom.geoms, key=lambda p: p.area)


def merge_edges_with_districts(gdf_csv, districts_gdf):
    """
    Merge network edges with districts using spatial join.
    
    Parameters:
        gdf_csv: GeoDataFrame containing network edges with your specific columns
        districts_gdf: GeoDataFrame containing district polygons
    
    Returns:
        GeoDataFrame with edges and their intersecting districts
    """
    # Perform spatial join
    gdf_edges_with_districts = gpd.sjoin(gdf_csv, districts_gdf, how='left', predicate='intersects')
    
    # Group by edge ID and aggregate attributes
    gdf_edges_with_districts = gdf_edges_with_districts.groupby('link').agg({
        'from_node': 'first',
        'to_node': 'first',
        'length': 'first',
        'freespeed': 'first',
        'capacity': 'first',
        'lanes': 'first',
        'modes': 'first',
        'vol_car': 'first',
        'osm:way:footway': 'first',
        'osm:way:vehicle': 'first',
        'osm:way:traffic_calming': 'first',
        'osm:way:junction': 'first',
        'osm:way:motorcycle': 'first',
        'osm:way:lanes': 'first',
        'osm:way:psv': 'first',
        'osm:way:service': 'first',
        'osm:way:id': 'first',
        'osm:way:access': 'first',
        'osm:way:oneway': 'first',
        'osm:way:highway': 'first',
        'osm:relation:route': 'first',
        'osm:way:railway': 'first',
        'osm:way:name': 'first',
        'storageCapacityUsedInQsim': 'first',
        'osm:way:tunnel': 'first',
        'geometry': 'first',
        'zone_id': lambda x: [int(i) for i in x.dropna()]
    }).reset_index()

    # Convert numeric columns
    numeric_columns = ['freespeed', 'capacity', 'lanes', 'vol_car', 'storageCapacityUsedInQsim']
    for col in numeric_columns:
        if col in gdf_edges_with_districts.columns:
            gdf_edges_with_districts[col] = pd.to_numeric(gdf_edges_with_districts[col], errors='coerce')

    # Ensure it's a GeoDataFrame
    gdf_edges_with_districts = gpd.GeoDataFrame(
        gdf_edges_with_districts, 
        geometry='geometry', 
        crs='EPSG:25832'
    )
    
    return gdf_edges_with_districts


def generate_hexagon_grid_for_polygon(polygon, hexagon_size, projection='EPSG:25832'):
    """
    Generates a hexagonal grid that fits within a given polygon.

    Parameters:
        polygon (shapely.geometry.Polygon): The polygon to clip the grid to.
        hexagon_size (float): The distance from the hexagon's center to any vertex.
        projection (str): The coordinate reference system for the polygon and grid.
    
    Returns:
        GeoDataFrame: A GeoDataFrame containing hexagons (as polygons) clipped to the input polygon,
                      each with a unique hexagon grid_id.
    """
    # Create a GeoDataFrame from the input polygon using the given projection.
    poly_gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs=projection)
    
    # Obtain the bounding box of the polygon.
    xmin, ymin, xmax, ymax = poly_gdf.total_bounds

    # Compute the vertical scaling factor using sin(60°) (~0.866).
    # This factor is used to correctly space the hexagon vertices vertically.
    a = np.sin(np.pi / 3) 

    # Define the x positions (columns) for hexagon grid centers.
    # Here, the horizontal spacing between potential hexagon positions is 3 * hexagon_size.
    cols = np.arange(np.floor(xmin), np.ceil(xmax), 3 * hexagon_size)
    
    # Define the y positions (rows) for hexagon grid centers.
    # The y positions are scaled by the factor 'a' to account for the vertical distance between rows.
    rows = np.arange(np.floor(ymin) / a, np.ceil(ymax) / a, hexagon_size)

    # Generate hexagon geometries for each grid position.
    hexagons = []
    for x in cols:
        for i, y in enumerate(rows):
            # Offset every other row horizontally to create a staggered hexagon grid.
            if i % 2 == 0:
                x0 = x
            else:
                x0 = x + 1.5 * hexagon_size

            # Create a hexagon by specifying its six vertices.
            # The vertices are calculated relative to (x0, y) and scaled vertically by 'a'.
            hexagon = sgeo.Polygon([
                (x0, y * a),
                (x0 + hexagon_size, y * a),
                (x0 + 1.5 * hexagon_size, (y + hexagon_size) * a),
                (x0 + hexagon_size, (y + 2 * hexagon_size) * a),
                (x0, (y + 2 * hexagon_size) * a),
                (x0 - 0.5 * hexagon_size, (y + hexagon_size) * a),
            ])
            hexagons.append(hexagon)
    
    # Convert the list of hexagons into a GeoDataFrame with the specified projection.
    grid = gpd.GeoDataFrame({'geometry': hexagons}, crs=projection)
    
    # Clip the hexagon grid to the input polygon so that only hexagons (or portions thereof)
    # that fall within the polygon are retained.
    grid_clipped = gpd.clip(grid, poly_gdf)
    
    # Reset the index and assign a unique grid_id to each hexagon.
    grid_clipped = grid_clipped.reset_index(drop=True)
    grid_clipped['grid_id'] = grid_clipped.index
    
    return grid_clipped


def generate_hexagon_grid_for_districts(districts_gdf, hexagon_size, gdf_edges_with_districts, projection='EPSG:25832'):
    '''
    This function generates a hexagon grid for zone_id 1 and assigns each edge to the hexagon(s) it falls into
    '''
    # Get the boundary of zone_id 1
    zone_1_boundary = districts_gdf[districts_gdf['zone_id'] == 1].geometry.values[0]

    # Create a single continuous hexagon grid for zone 1
    hexagon_grid_all = generate_hexagon_grid_for_polygon(zone_1_boundary, hexagon_size, projection='EPSG:25832')
    
    # Add district information to each hexagon
    def get_intersecting_districts(hex_geom):
        """
        This function takes a hexagon geometry and returns a list of district IDs that intersect with the hexagon.
        """
        intersecting_districts = []
        for idx, row in districts_gdf.iterrows():
            if hex_geom.intersects(row['geometry']):
                district_id = row.get('zone_id', idx+1)
                intersecting_districts.append(district_id)
        return intersecting_districts

    hexagon_grid_all['hex_zone_id'] = hexagon_grid_all['geometry'].apply(get_intersecting_districts)
    print(f"Total number of hexagons created: {len(hexagon_grid_all)}")
    print(f"Number of hexagons in multiple districts: {len(hexagon_grid_all[hexagon_grid_all['hex_zone_id'].apply(lambda x: len(x) >= 2)])}")

    # Spatial join to assign each edge the hexagon(s) it falls into
    gdf_edges_with_hex = gpd.sjoin(gdf_edges_with_districts, hexagon_grid_all[['geometry', 'hex_zone_id', 'grid_id']], 
                                how='left', predicate='intersects')

    # Group by edge 'link' and aggregate the hexagon IDs into a list
    gdf_edges_with_hex = gdf_edges_with_hex.groupby('link').agg({
        'from_node': 'first',
        'to_node': 'first',
        'length': 'first',
        'freespeed': 'first',
        'capacity': 'first',
        'lanes': 'first',
        'modes': 'first',
        'vol_car': 'first',
        'osm:way:footway': 'first',
        'osm:way:vehicle': 'first',
        'osm:way:traffic_calming': 'first',
        'osm:way:junction': 'first',
        'osm:way:motorcycle': 'first',
        'osm:way:lanes': 'first',
        'osm:way:psv': 'first',
        'osm:way:service': 'first',
        'osm:way:id': 'first',
        'osm:way:access': 'first',
        'osm:way:oneway': 'first',
        'osm:way:highway': 'first',
        'osm:relation:route': 'first',
        'osm:way:railway': 'first',
        'osm:way:name': 'first',
        'storageCapacityUsedInQsim': 'first',
        'osm:way:tunnel': 'first',
        'geometry': 'first',
        'grid_id': lambda x: list(x.dropna()),  # Aggregate hexagon IDs
        'zone_id': lambda x: list(set([d for dist_list in x.dropna() if isinstance(dist_list, list) for d in dist_list])),  # Aggregate district IDs
        'hex_zone_id': lambda x: list(set([d for dist_list in x.dropna() for d in dist_list if isinstance(dist_list, list)])),  # Aggregate unique districts
    }).reset_index()
    
    # Convert back to GeoDataFrame
    gdf_edges_with_hex = gpd.GeoDataFrame(
        gdf_edges_with_hex,
        geometry='geometry',
        crs=gdf_edges_with_districts.crs  # Preserve the original CRS
    )

    # Rename the aggregated columns
    gdf_edges_with_hex.rename(columns={'grid_id': 'hexagon'}, inplace=True)

    # Convert numeric columns
    numeric_columns = ['freespeed', 'capacity', 'lanes', 'vol_car', 'storageCapacityUsedInQsim']
    for col in numeric_columns:
        if col in gdf_edges_with_hex.columns:
            gdf_edges_with_hex[col] = pd.to_numeric(gdf_edges_with_hex[col], errors='coerce')

    # Create is_in_stadt column
    gdf_edges_with_hex['is_in_stadt'] = gdf_edges_with_hex['hex_zone_id'].apply(lambda x: 1 if 1 in x else 0)

    # Hexagon Statistics
    print("\nEdge Statistics:")
    print(f"Total number of edges: {len(gdf_edges_with_hex)}")
    print('Edges in stadt: ', len(gdf_edges_with_hex[gdf_edges_with_hex['hex_zone_id'].apply(lambda x: 1 in x)]))
    print('Edges not in stadt: ', len(gdf_edges_with_hex[gdf_edges_with_hex['hex_zone_id'].apply(lambda x: 1 not in x)]))
   

    return gdf_edges_with_hex, hexagon_grid_all

def check_hexagon_statistics(gdf_edges_with_hex, hexagon_grid_all):
    unique_values = set(item for sublist in gdf_edges_with_hex['hexagon'] for item in sublist)
    print(unique_values)
    print('Number of Hexagons containing edges: ', len (unique_values))
    print('Total number of Hexagons created: ', len(hexagon_grid_all))

def plot_grid_and_edges(gdf_edges_with_hex, hexagon_grid_all, districts_gdf):
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(15, 15))

    def get_edge_color(zones):
        if not isinstance(zones, list):
            return 'gray'
        if 1 in zones and 2 in zones:
            return 'green'
        elif 1 in zones:
            return 'blue'
        elif 2 in zones:
            return 'gray'
        else:
            return 'gray'
    # Plot roads in gray
    gdf_edges_with_hex['edge_color'] = gdf_edges_with_hex['zone_id'].apply(get_edge_color)

    # Plot all edges at once, grouped by color
    for color in ['blue', 'gray', 'green']:
        edges = gdf_edges_with_hex[gdf_edges_with_hex['edge_color'] == color]
        if not edges.empty:
            edges.plot(
                ax=ax,
                color=color,
                linewidth=0.5,
                label=None
            )

    # Plot districts in yellow (uniform)
    districts_gdf.plot(
    ax=ax, 
    column='zone_id',       # Use a numeric or categorical column for coloring
    cmap='PuBuGn',          # Soothing gradient
    alpha=0.1,
    edgecolor='black',
    linewidth=0.5,
    legend=False,
    label='Districts'
    )

    # Plot hexagons in red (edges only)
    hexagon_grid_all.plot(
        ax=ax, 
        color='none', 
        edgecolor='red',
        alpha=0.7,
        linewidth=0.6,
        label='Hexagons'
    )

    # Create custom legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=0.7, label='Zone 1'),
        Line2D([0], [0], color='gray', linewidth=0.7, label='Zone 2'),
        Line2D([0], [0], color='green', linewidth=0.7, label='Zones 1 & 2'),
        Patch(facecolor='yellow', edgecolor='black', alpha=0.2, label='Districts'),
        Line2D([0], [0], color='red', linewidth=0.8, label='Hexagons')
    ]

    ax.legend(handles=legend_elements)
    plt.title('Network with Hexagon Grid and Districts')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
import os
import logging

def chunks(lst, n):
    """Split list into n roughly equal chunks"""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def compute_edge_betweenness_for_chunk(args):
    G, nodes, weight = args
    return nx.edge_betweenness_centrality_subset(G, sources=nodes, targets=None, weight=weight)

def parallel_edge_betweenness(G, weight='length', processes=None):
    nodes = list(G.nodes())
    n_processes = processes or mp.cpu_count()
    node_chunks = chunks(nodes, n_processes)

    print(f"Running on {n_processes} processes...")

    with mp.Pool(processes=n_processes) as pool:
        results = pool.map(compute_edge_betweenness_for_chunk, [(G, chunk, weight) for chunk in node_chunks])

    # Merge all partial results
    combined = reduce(lambda x, y: Counter(x) + Counter(y), results)
    
    # Normalize like NetworkX does
    scale = 1.0 / ((len(G) - 1) * (len(G) - 2))
    if not G.is_directed():
        scale *= 2.0
    for k in combined:
        combined[k] *= scale

    return dict(combined)



# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def create_network_from_csv(df) -> nx.DiGraph:
    """
    Create a NetworkX directed graph from a CSV file containing link information.
    
    Args:
        csv_path: Path to the CSV file with columns 'from_node' and 'to_node'
        
    Returns:
        nx.DiGraph: A directed graph representing the network
    """
    try:
        # First, let's check if the file exists
        #if not csv_path.exists():
           # raise FileNotFoundError(f"File not found: {csv_path}")
            
        # Try to read the file
        #df = pd.read_csv(csv_path)
        
        # Print initial statistics about the DataFrame
        print(f"\nNumber of edges in CSV: {len(df)}")
        print(f"Number of unique from_nodes: {df['from_node'].nunique()}")
        print(f"Number of unique to_nodes: {df['to_node'].nunique()}")
        print(f"Number of unique links: {df['link'].nunique()}")
        
        # Print the first few rows to see what we're working with
        logger.info("First few rows of the CSV:")
        logger.info(df.head())
        
        # Print the column names
        logger.info("\nColumn names:")
        logger.info(df.columns.tolist())
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add edges to the graph
        for _, row in df.iterrows():
            G.add_edge(
                row['from_node'],
                row['to_node'],
                length=row['length'],  # Use length as weight
                id=row['link']
            )
        
        # Add basic graph properties
        #G.graph['name'] = csv_path.stem
        #G.graph['source'] = str(csv_path)
        
        # Log graph statistics
        #logger.info(f"Created network from {csv_path}")
        logger.info(f"Number of nodes: {G.number_of_nodes()}")
        logger.info(f"Number of edges: {G.number_of_edges()}")
        logger.info(f"Is directed: {G.is_directed()}")
        
        return G
        
    except Exception as e:
        logger.error(f"Error creating network: {e}")
        raise

import networkx as nx
import numpy as np

import networkx as nx
import numpy as np

def edge_closeness_centrality(G, centrality_weight='length'):
    edge_closeness = {}
    
    print("\n=== Edge Closeness Centrality Calculation ===")
    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Total edges: {G.number_of_edges()}")

    # Get strongly connected components
    scc = list(nx.strongly_connected_components(G))
    node_to_component = {}
    for component in scc:
        for node in component:
            node_to_component[node] = component

    # Calculate component sizes
    component_sizes = {node: len(component) for node, component in node_to_component.items()}
    
    print("\nComponent size distribution:")
    size_distribution = Counter(component_sizes.values())
    for size, count in sorted(size_distribution.items()):
        print(f"Components of size {size}: {count}")

    print("\nCalculating edge centrality...")
    total_edges = G.number_of_edges()
    processed_edges = 0
    zero_centrality = 0
    non_zero_centrality = 0

    for u, v in G.edges():
        processed_edges += 1
        if processed_edges % 1000 == 0:
            print(f"Processed {processed_edges}/{total_edges} edges...")

        # Get component size
        component_size = component_sizes[u]
        
        # Only handle isolated nodes differently
        if component_size == 1:  # Isolated node
            edge_closeness[(u, v)] = 0.0000000
            zero_centrality += 1
            continue

        # Calculate shortest paths from source node
        dist_u = nx.single_source_dijkstra_path_length(G, u, weight=centrality_weight)
        
        # Calculate total distance and count reachable nodes
        total_distance = 0
        reachable_count = 0
        
        for node in node_to_component[u]:
            if node != u and node != v:  # Exclude source and target nodes
                distance = dist_u.get(node, 0)
                if distance > 0:  # Only count actually reachable nodes
                    total_distance += distance
                    reachable_count += 1

        # Calculate normalized centrality
        if reachable_count > 0:
            average_distance = total_distance / reachable_count
            centrality = round(1 / average_distance, 7) if average_distance > 0 else 0.0000000
        else:
            centrality = 0.0000000

        edge_closeness[(u, v)] = centrality
        
        if centrality > 0:
            non_zero_centrality += 1
        else:
            zero_centrality += 1

    print("\n=== Results Summary ===")
    print(f"Total edges processed: {processed_edges}")
    print(f"Edges with non-zero centrality: {non_zero_centrality}")
    print(f"Edges with zero centrality: {zero_centrality}")
    print(f"Percentage of edges with non-zero centrality: {(non_zero_centrality/processed_edges)*100:.2f}%")
    
    # Print centrality statistics for non-zero values
    centrality_values = [v for v in edge_closeness.values() if v > 0]
    if centrality_values:
        print("\nCentrality Statistics (non-zero values only):")
        print(f"Average centrality: {sum(centrality_values)/len(centrality_values):.7f}")
        print(f"Max centrality: {max(centrality_values):.7f}")
        print(f"Min centrality: {min(centrality_values):.7f}")
    else:
        print("\nNo edges with non-zero centrality found!")

    return edge_closeness


def plot_centrality_measures(gdf_edges_with_hex, centrality_df, output_dir):
    """
    Create visualizations for betweenness and closeness centrality measures.
    
    Args:
        gdf_edges_with_hex: GeoDataFrame containing the network edges
        centrality_df: DataFrame containing centrality measures
        output_dir: Directory to save the plots
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot betweenness distribution
    sns.histplot(data=centrality_df, x='betweenness', ax=ax1, bins=50)
    ax1.set_title('Betweenness Centrality Distribution')
    ax1.set_xlabel('Betweenness Centrality')
    ax1.set_ylabel('Count')

    # Plot closeness distribution
    sns.histplot(data=centrality_df, x='closeness', ax=ax2, bins=50)
    ax2.set_title('Closeness Centrality Distribution')
    ax2.set_xlabel('Closeness Centrality')
    ax2.set_ylabel('Count')

    # Add some statistics
    print("\nBetweenness Centrality Statistics:")
    print(centrality_df['betweenness'].describe())
    print("\nCloseness Centrality Statistics:")
    print(centrality_df['closeness'].describe())

    # Save the plot
    plt.tight_layout()
    plot_path = output_dir / 'centrality_distributions.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved centrality distributions plot to: {plot_path}")
    plt.close()  # Close the figure to free memory

    # Create separate plots for better resolution
    # Betweenness plot
    plt.figure(figsize=(10, 6))
    sns.histplot(data=centrality_df, x='betweenness', bins=50)
    plt.title('Betweenness Centrality Distribution')
    plt.xlabel('Betweenness Centrality')
    plt.ylabel('Count')
    plt.tight_layout()
    betweenness_path = output_dir / 'betweenness_distribution.png'
    plt.savefig(betweenness_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Closeness plot
    plt.figure(figsize=(10, 6))
    sns.histplot(data=centrality_df, x='closeness', bins=50)
    plt.title('Closeness Centrality Distribution')
    plt.xlabel('Closeness Centrality')
    plt.ylabel('Count')
    plt.tight_layout()
    closeness_path = output_dir / 'closeness_distribution.png'
    plt.savefig(closeness_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved betweenness distribution plot to: {betweenness_path}")
    print(f"Saved closeness distribution plot to: {closeness_path}")

def analyze_centrality_measures(gdf_edges_with_hex, output_dir, city_only=True):
    """
    Analyze and visualize centrality measures for the network.
    
    Args:
        gdf_edges_with_hex: GeoDataFrame containing the network edges
        output_dir: Directory to save output files
        city_only: If True, only analyze edges with is_in_stadt = 1
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter edges if city_only is True
        if city_only:
            print("Filtering edges to include only city edges (is_in_stadt = 1)")
            gdf_filtered = gdf_edges_with_hex[gdf_edges_with_hex['is_in_stadt'] == 1].copy()
            print(f"Number of edges in city: {len(gdf_filtered)} out of {len(gdf_edges_with_hex)} total edges")
        else:
            gdf_filtered = gdf_edges_with_hex.copy()
        
        # Save the filtered GeoDataFrame to CSV temporarily
        temp_csv = output_dir / "temp_network.csv"
        # Save only the necessary columns
        temp_df = gdf_filtered[['link', 'from_node', 'to_node', 'length']].copy()
        temp_df.to_csv(temp_csv, index=False)
        
        # Create the network
        print("Creating network from GeoDataFrame...")
        G = create_network_from_csv(temp_df)
        
        # Remove temporary file
        temp_csv.unlink()
        
        # Compute centrality measures
        print("Computing betweenness centrality...")
        betweenness = parallel_edge_betweenness(G, weight='length', processes=8)
        
        print("Computing edge closeness centrality...")
        closeness = edge_closeness_centrality(G,weight='length')
        
        # Create DataFrame for centrality measures
        
        centrality_df = pd.DataFrame({
            'from_node': [u for u, v in betweenness.keys()],
            'to_node': [v for u, v in betweenness.keys()],
            'link_id': [G[u][v].get('id', f"{u}-{v}") for u, v in betweenness.keys()],
            'betweenness': list(betweenness.values()),
            'closeness': list(closeness.values())
        })
        
        # Save results
        output_file = output_dir / ('city_centrality_measures.csv' if city_only else 'all_centrality_measures.csv')
        centrality_df.to_csv(output_file, index=False)
        print(f"Saved centrality measures to {output_file}")
        
        # Print summary statistics
        print("\nCentrality Measures Summary:")
        print(f"Number of edges analyzed: {len(centrality_df)}")
        print("\nBetweenness Centrality Statistics:")
        print(centrality_df['betweenness'].describe())
        print("\nCloseness Centrality Statistics:")
        print(centrality_df['closeness'].describe())
        
        # Plot results
        plot_centrality_measures(gdf_filtered, centrality_df, output_dir)
        
        return centrality_df
        
    except Exception as e:
        print(f"Error in analyze_centrality_measures: {e}")
        raise

def main():
    # Load MATSim network
    #matsim_network, nodes_dict, df_edges = matsim_network_input_to_gdf(matsim_network_file_path)
    # Get road types
    cleaned_network = clean_duplicates_based_on_modes(csv_filepath)
    cleaned_network['geometry'] = cleaned_network['geometry'].apply(wkt.loads)
    cleaned_network=gpd.GeoDataFrame(cleaned_network,geometry='geometry')
    #gdf_csv = get_road_types(csv_filepath,csv_output_path)
    
    gdf = gpd.read_file(administrative_boundary_json_path)
    # Modify districts geodataframe
    districts_gdf = modify_districts_geodataframe(gdf)
    # Merge edges with districts
    gdf_edges_with_districts = merge_edges_with_districts(cleaned_network, districts_gdf)
    # Generate hexagon grid for districts
    gdf_edges_with_hex, hexagon_grid_all = generate_hexagon_grid_for_districts(districts_gdf, hexagon_size, gdf_edges_with_districts)
    # Plot grid and edges
    plot_grid_and_edges(gdf_edges_with_hex, hexagon_grid_all, districts_gdf)
    # Check hexagon statistics
    check_hexagon_statistics(gdf_edges_with_hex, hexagon_grid_all)
    # Analyze centrality measures
    centrality_df = analyze_centrality_measures(gdf_edges_with_hex, output_dir=base_dir / "data" / "simulation_output"/ "centrality_measures", city_only=True)

if __name__ == "__main__":
    main()
