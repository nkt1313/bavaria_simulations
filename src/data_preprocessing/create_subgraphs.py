'''
The goal of this script is to create 10,000 network.xml.gz files for each city, with the capacity reduction policies applied in different spatial and graph-based ways.

The steps are as follows, for each city:

1. Read the network file and create hexagons for the city
2. For each hexagon, identify roads within the hexagon, fullfilling the following criteria: 
   For each road type (trunk, primary, secondary, tertiary, residential, etc.): 
    - Low betweenness centrality
    - High centrality (networkx has a function for this, and a start is provided in notebook ``investigate_closeness_criteria.ipynb``)
3. Combine the hexagon scenarios: Given n hexagons and r road types, there are (2^n * 2^r) possible combinations. Of the possible combinations, choose 10,000 combinations with their set length following a normal distribution of size n/5.
4. Find a way to label the scenario combinations (in the following: ``scenarios''), for their file names.
5. Create network.xml.gz files for each scenario, with the file names found in step 4. For easier processing, save them in folders of 1,000 files each. The network structure should be: {city}/networks/networks_{1000}/network_scenario_x.xml.gz, {city}/networks/networks_{2000}/network_scenario_y.xml.gz, etc.
'''

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

os.chdir(r"C:\Users\nktba\bavaria_simulations") # Set working directory
base_dir = os.getcwd()
administrative_boundary_json_path = os.path.join(base_dir, "data", "output", "boundary_files", "Augsburg.json")
matsim_network_file_path = os.path.join(base_dir, "data", "output", "simulation_data_per_city","augsburg", "augsburg_network.xml.gz")
output_base_path = os.path.join(base_dir, "data", "output", "network_files")
distribution_mean_factor =5 
distribution_std_factor = 10 # for n denoting the number of hexagons, we create subgraphs whose length follows a normal distribution with mean (n/distribution_mean_factor and std dev (n/distribution_std_factor)
#Define variables
hexagon_size = 1500  # Size in meters for EPSG:25832 and in degrees for EPSG:4326 **********VERY IMPORTANT********** 
capacity_tuning_factor = 0.5 #This is the factor by which the capacity of the links is reduced
betweenness_centrality_cutoff = 0.8 # Take the lowest 80% of the links based on betweenness centrality
closeness_centrality_cutoff = 0.8 # Take the highest 80% of the links based on closeness centrality
distribution_mean_factor = 5
distribution_std_factor = 10 # for n denoting the number of hexagons, we create subgraphs whose length follows a normal distribution with mean (n/distribution_mean_factor and std dev (n/distribution_std_factor)

#Define the path to the check output subgraph
check_output_subgraph_path = os.path.join(base_dir, "data", "output", "network_files", "Augsburg", "networks", "networks_0", "network_residential_n7_s1.xml.gz")

def matsim_network_to_gdf(network_file):
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

def get_osm_roads(bounds, crs='EPSG:25832'):
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
    # Convert bounds to lat/lon (EPSG:4326) if necessary
    if crs != 'EPSG:4326':
        bounds_gdf = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=crs)
        bounds_wgs84 = bounds_gdf.to_crs('EPSG:4326')
        bounds = bounds_wgs84.total_bounds
    
    # Create bbox tuple (left/west, bottom/south, right/east, top/north)
    bbox = (
        float(bounds[0]),  # left/west
        float(bounds[1]),  # bottom/south
        float(bounds[2]),  # right/east
        float(bounds[3])   # top/north
    )
    
    try:
        # Download OSM data
        G = ox.graph_from_bbox(
            bbox,
            network_type='all',
            simplify=True,
            retain_all=False,
            truncate_by_edge=True
        )
        
        # Convert to GeoDataFrame (edges only, no nodes)
        gdf_osm = ox.graph_to_gdfs(G, nodes=False, edges=True)
        
        # Convert back to original CRS if necessary
        if crs != 'EPSG:4326':
            gdf_osm = gdf_osm.to_crs(crs)
        
        return gdf_osm
        
    except Exception as e:
        print(f"Error downloading OSM data: {str(e)}")
        return None
    
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
        'unclassified': 'unclassified',
        'living_street': 'living_street',
        'pedestrian': 'other',
        'service': 'other',
        'track': 'other',
        'path': 'other',
        'cycleway': 'other',
        
    }
    
    return road_type_mapping.get(highway_type, 'residential')

def spatial_join_networks(matsim_gdf, osm_gdf):
    """
    Perform spatial join between MATSim and OSM networks
    
    Parameters:
        matsim_gdf (GeoDataFrame): MATSim network with 'id', 'geometry' etc.
        osm_gdf (GeoDataFrame): OSM network with 'highway', 'geometry' etc.
    
    Returns:
        GeoDataFrame: MATSim network enriched with OSM road types
    """
    # Buffer the MATSim links slightly to ensure matching
    buffer_distance = 5  # meters
    matsim_gdf['geometry_buffer'] = matsim_gdf.geometry.buffer(buffer_distance)
    
    # Ensure we have unique indices for the spatial join
    matsim_gdf = matsim_gdf.reset_index(drop=True)
    osm_gdf = osm_gdf.reset_index(drop=True)
    
    # Perform spatial join
    joined = gpd.sjoin(
        matsim_gdf,
        osm_gdf[['highway', 'geometry']],  # only keep relevant OSM columns
        how='left',
        predicate='intersects'
    )
    
    def get_most_common_highway(x):
        """Helper function to get most common highway type"""
        if x.empty:
            return None
        # Remove any NA values
        x = x.dropna()
        if x.empty:
            return None
        # Get value counts and return the most common
        counts = x.value_counts()
        if len(counts) > 0:
            return counts.index[0]
        return None

    # Group by the original MATSim link ID and aggregate OSM attributes
    result = joined.groupby('id').agg({
        'from': 'first',
        'to': 'first',
        'length': 'first',
        'freespeed': 'first',
        'capacity': 'first',
        'permlanes': 'first',
        'oneway': 'first',
        'modes': 'first',
        'geometry': 'first',
        'highway': get_most_common_highway  # Using our custom aggregation function
    }).reset_index()
    
    # Drop the buffer geometry if it exists in the result
    if 'geometry_buffer' in result.columns:
        result = result.drop(columns=['geometry_buffer'])
    
    # Convert back to GeoDataFrame
    result = gpd.GeoDataFrame(result, geometry='geometry', crs=matsim_gdf.crs)
    
    # Print some statistics
    print("\nRoad type distribution:")
    print(result['highway'].value_counts(dropna=False))
    print(f"\nUnmatched links: {result['highway'].isna().sum()}")
    
    return result,joined


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
    districts_gdf = districts_gdf[districts_gdf['zone_id']==1]#only keep the stadt region
    return districts_gdf


def multipolygon_to_polygon(geom):
    '''
    This function converts a MultiPolygon to a Polygon with the largest area.
    '''
    return max(geom.geoms, key=lambda p: p.area)
    

def merge_edges_with_districts(enriched_network, districts_gdf):
    gdf_edges_with_districts = gpd.sjoin(enriched_network, districts_gdf, how='left', predicate='intersects') #left join to ensure all edges are included

    # some edges may be overlapping multiple districts, so we need to aggregate the district IDs for each edge
    gdf_edges_with_districts = gdf_edges_with_districts.groupby('id').agg({
        'from': 'first',
        'to': 'first',
        'length': 'first',
        'freespeed': 'first',
        'capacity': 'first',
        'permlanes': 'first',
        'oneway': 'first',
        'modes': 'first',
        'geometry': 'first',
        'zone_id': lambda x: list(x.dropna()), #aggregate the district IDs for each edge and drop the NA values,
        'highway': 'first'
    }).reset_index()

    # Convert freespeed and capacity to numeric values
    gdf_edges_with_districts['freespeed'] = pd.to_numeric(gdf_edges_with_districts['freespeed'], errors='coerce')
    gdf_edges_with_districts['capacity'] = pd.to_numeric(gdf_edges_with_districts['capacity'], errors='coerce')

    gdf_edges_with_districts = gpd.GeoDataFrame(gdf_edges_with_districts, geometry='geometry', crs='EPSG:25832')
    
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


def generate_hexagon_grid_for_districts(districts_gdf, hexagon_size,gdf_edges_with_districts, projection='EPSG:25832'):
    '''
    This function generates a hexagon grid for the districts and assigns each edge to the hexagon(s) it falls into
    '''
    # First, get the outer boundary of the entire region
    region_boundary = districts_gdf.union_all()

    # Create a single continuous hexagon grid for the entire region
    hexagon_grid_all = generate_hexagon_grid_for_polygon(region_boundary, hexagon_size, projection='EPSG:25832')
    # Add district information to each hexagon
    def get_intersecting_districts(hex_geom):
        """
        This function takes a hexagon geometry and returns a list of district IDs that intersect with the hexagon.
        """
        intersecting_districts = []
        for idx, row in districts_gdf.iterrows():
            if hex_geom.intersects(row['geometry']):
                district_id = row.get('district_name', idx+1)
                intersecting_districts.append(district_id)
        return intersecting_districts

    hexagon_grid_all['districts'] = hexagon_grid_all['geometry'].apply(get_intersecting_districts)
    print(f"Total number of hexagons created: {len(hexagon_grid_all)}")
    print(f"Number of hexagons in multiple districts: {len(hexagon_grid_all[hexagon_grid_all['districts'].apply(lambda x: len(x) >= 2)])}")

    # Spatial join to assign each edge the hexagon(s) it falls into
    gdf_edges_with_hex = gpd.sjoin(gdf_edges_with_districts, hexagon_grid_all[['geometry', 'districts', 'grid_id']], 
                                how='left', predicate='intersects')

    # Group by edge 'id' and aggregate the hexagon IDs into a list
    gdf_edges_with_hex = gdf_edges_with_hex.groupby('id').agg({
        'from': 'first',
        'to': 'first',
        'length': 'first',
        'freespeed': 'first',
        'capacity': 'first',
        'permlanes': 'first',
        'oneway': 'first',
        'modes': 'first',
        'geometry': 'first',
        'grid_id': lambda x: list(x.dropna()), #aggregate the hexagon IDs for each edge and drop the NA values
        'districts': lambda x: list(set([d for dist_list in x.dropna() for d in dist_list if isinstance(dist_list, list)])),  # Aggregate unique districts
        'highway': 'first'
    }).reset_index()

    # Rename the aggregated column
    gdf_edges_with_hex.rename(columns={'grid_id': 'hexagon'}, inplace=True)
    gdf_edges_with_hex.rename(columns={'districts': 'zones'}, inplace=True)

    # Hexagon Statistics
    print("\nEdge Statistics:")
    print(f"Total number of edges: {len(gdf_edges_with_hex)}")
    print(f"Number of edges in multiple hexagons: {len(gdf_edges_with_hex[gdf_edges_with_hex['hexagon'].apply(lambda x: len(x) >= 2)])}")
    print(f"Number of edges in multiple districts: {len(gdf_edges_with_hex[gdf_edges_with_hex['zones'].apply(lambda x: len(x) >= 2)])}")
    return gdf_edges_with_hex,hexagon_grid_all

def check_hexagon_statistics(gdf_edges_with_hex, hexagon_grid_all):
    unique_values = set(item for sublist in gdf_edges_with_hex['hexagon'] for item in sublist)
    print(unique_values)
    print('Number of Hexagons containing edges: ', len (unique_values))
    print('Total number of Hexagons created: ', len(hexagon_grid_all))

def plot_grid_and_edges(gdf_edges_with_hex, hexagon_grid_all, enriched_network,districts_gdf):
    # Define road styles with colors and line widths
    road_styles = {
        'trunk':      {'color': '#e31a1c', 'width': 2.5},  # red
        'primary':    {'color': '#ff7f00', 'width': 2.0},  # orange
        'secondary':  {'color': '#33a02c', 'width': 1.5},  # green
        'tertiary':   {'color': '#1f78b4', 'width': 1.0},  # blue
        'residential': {'color': '#984ea3', 'width': 0.5}, # purple
        None:         {'color': '#999999', 'width': 0.5}   # gray for unmatched roads
    }

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(15, 15))

    # Plot each road type separately, starting with smaller roads
    road_types = ['residential', 'tertiary', 'secondary', 'primary', 'trunk', None]
    for road_type in road_types:
        mask = enriched_network['highway'] == road_type
        style = road_styles[road_type]
        enriched_network[mask].plot(
            ax=ax,
            color=style['color'],
            linewidth=style['width'],
            label=f"{road_type if road_type is not None else 'unmatched'}"
        )

    # Plot districts
    districts_gdf.plot(ax=ax, 
                    column='zone_id',  # Color by c_ar values
                    cmap='YlGnBu',  # Color scheme
                    alpha=0.3,       # Transparency
                    legend=False,     # Show legend
                    legend_kwds={'label': 'District Area (m²)'},  # Legend label
                    label='Districts')

    # Plot all hexagons
    hexagon_grid_all.plot(ax=ax, color='none', edgecolor='green', alpha=0.3, label='Hexagons')

    # Highlight hexagons which contain edges crossing multiple districts
    hexagon_grid_all[hexagon_grid_all['districts'].apply(lambda x: len(x) >= 2)].plot(
        ax=ax, color='red', alpha=0.3, label='Hexagons in multiple districts'
    )

    legend_elements = [
        Line2D([0], [0], color='blue', alpha=0.8, label='Network Edges'),
        #Patch(facecolor='lightgray', alpha=0.7, label='Districts'),
        Patch(facecolor='none', edgecolor='green', alpha=0.3, label='Hexagons'),
        Patch(facecolor='red', alpha=0.3, label='Hexagons in multiple districts')
    ]

    ax.legend(handles=legend_elements)
    plt.title('Continuous Hexagon Grid with Network Edges and Districts')
    plt.axis('equal')
    plt.show()
    

def check_road_type_distribution(gdf_edges_with_hex):
    gdf_edges_with_hex['highway_consolidated'] = gdf_edges_with_hex['highway'].apply(consolidate_road_types)

    # Print some statistics to verify the consolidation
    print("\nRoad type distribution after consolidation:")
    print(gdf_edges_with_hex['highway_consolidated'].value_counts())

    # Optional: Check for any None values
    print("\nNumber of None values:", gdf_edges_with_hex['highway_consolidated'].isna().sum())
    return gdf_edges_with_hex

########################################################################

###################################################################

def generate_road_type_specific_subsets(gdf_edges_with_hex, target_size=10000, std_dev=3, seed=13):
    """
    Generate unique subsets of hexagon IDs for each road type, where the total number of subsets
    is target_size, distributed evenly across road types.
    
    Parameters:
    -----------
    gdf_edges_with_hex : GeoDataFrame
        GeoDataFrame containing road network data with hexagon assignments and highway types
    target_size : int
        Total number of subsets to generate (will be divided among road types)
    std_dev : float
        Standard deviation for the normal distribution of subset sizes
    seed : int
        Random Seed for reproducibility
    
    Returns:
    --------
    dict : Dictionary mapping each road type to its list of hexagon ID subsets
    """
    # Set the seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Get unique hexagon IDs containing edges and convert to list
    hexagon_ids = list(gdf_edges_with_hex['hexagon'].explode().dropna().unique())
    
    # Get unique road types (after consolidation)
    road_types = gdf_edges_with_hex['highway_consolidated'].unique()
    road_types = [rt for rt in road_types if rt is not None]
    
    # Calculate target mean for subset size (overall)
    target_mean = len(hexagon_ids) / 5
    
    # Calculate number of subsets per road type
    subsets_per_type = target_size // len(road_types)
    
    # Dictionary to store subsets for each road type
    road_type_subsets = {}
    
    for road_type in road_types:
        # Generate subsets for this road type
        unique_subsets = set()
        
        while len(unique_subsets) < subsets_per_type:
            # Sample subset size from normal distribution
            subset_size = max(1, int(np.random.normal(target_mean, std_dev)))
            subset_size = min(subset_size, len(hexagon_ids))
            
            # Randomly choose subset_size hexagons and sort them
            subset = tuple(sorted(random.sample(hexagon_ids, subset_size)))
            unique_subsets.add(subset)
        
        road_type_subsets[road_type] = list(unique_subsets)
    
    # Calculate overall statistics
    all_subsets = [subset for subsets in road_type_subsets.values() for subset in subsets]
    overall_mean = np.mean([len(s) for s in all_subsets])
    
    print(f"\nOverall Statistics:")
    print(f"Total number of subsets: {len(all_subsets)}")
    print(f"Target mean subset length: {target_mean:.2f}")
    print(f"Actual mean subset length: {overall_mean:.2f}")
    print(f"Number of road types: {len(road_types)}")
    print(f"Subsets per road type: {subsets_per_type}")
    print(f"Seed used: {seed}")
    
    return road_type_subsets


def generate_scenario_labels(road_type_subsets):
    """
    Generate meaningful labels for scenario combinations based on active road types.
    
    Parameters:
    -----------
    road_type_subsets : dict
        Dictionary mapping each road type to its list of hexagon ID subsets
    
    Returns:
    --------
    dict : Dictionary mapping scenario indices to their labels
    """
    # Get all road types
    road_types = list(road_type_subsets.keys())
    
    # Create a mapping for scenario labels
    scenario_labels = {}
    
    # For each road type
    for road_type in road_types:
        # For each subset in this road type
        for i, subset in enumerate(road_type_subsets[road_type]):
            # Create a label that includes:
            # 1. Road type
            # 2. Number of hexagons in the subset
            # 3. Scenario number
            label = f"{road_type}_n{len(subset)}_s{i+1}"
            scenario_labels[(road_type, i)] = label
    
    return scenario_labels


def create_scenario_networks(gdf_edges_with_hex, road_type_subsets, scenario_labels, city_name, output_base_path, nodes_dict):
    """
    Create network.xml.gz files for each scenario, organized in folders of 1,000 files.
    Includes all edges from parent network but adjusts capacities for edges in scenario hexagons.
    
    Parameters:
    -----------
    gdf_edges_with_hex : GeoDataFrame
        Original network data with hexagon assignments
    road_type_subsets : dict
        Dictionary mapping road types to their hexagon subsets
    scenario_labels : dict
        Dictionary mapping scenario indices to their labels
    city_name : str
        Name of the city (e.g., 'Augsburg')
    output_base_path : str
        Base path for saving the network files
    nodes_dict : dict
        Dictionary containing node coordinates
    """
    import os
    import gzip
    import xml.etree.ElementTree as ET
    
    # Create base directory structure
    networks_base = os.path.join(output_base_path, city_name, 'networks')
    os.makedirs(networks_base, exist_ok=True)
    
    # Counter for total scenarios
    total_scenarios = 0
    
    # Process each road type and its scenarios
    for road_type, subsets in road_type_subsets.items():
        for i, subset in enumerate(subsets):
            # Get the scenario label
            label = scenario_labels[(road_type, i)]
            
            # Calculate which folder this scenario belongs to (1000 files per folder)
            folder_number = (total_scenarios // 1000) * 1000
            folder_name = f"networks_{folder_number}"
            folder_path = os.path.join(networks_base, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
            # Create the network file name
            network_filename = f"network_{label}.xml.gz"
            network_path = os.path.join(folder_path, network_filename)
            
            # Get edges in scenario hexagons (and the correct road type)
            scenario_mask = gdf_edges_with_hex['hexagon'].apply(
                lambda x: any(h in subset for h in x) if isinstance(x, list) else False
            ) & (gdf_edges_with_hex['highway_consolidated'] == road_type)
            
            scenario_edges = gdf_edges_with_hex[scenario_mask]
            
            # Create network XML structure
            root = ET.Element('network')
            
            # Add all nodes from the parent network
            nodes = ET.SubElement(root, 'nodes')
            node_ids = set()
            for _, edge in gdf_edges_with_hex.iterrows():
                node_ids.add(edge['from'])
                node_ids.add(edge['to'])
            
            for node_id in node_ids:
                node = ET.SubElement(nodes, 'node')
                node.set('id', str(node_id))
                if node_id in nodes_dict:
                    x, y = nodes_dict[node_id]
                    node.set('x', str(x))
                    node.set('y', str(y))
                else:
                    print(f"Warning: Node {node_id} not found in nodes_dict")
                    continue
            
            # Add all links with adjusted capacities for scenario edges
            links = ET.SubElement(root, 'links')
            for _, edge in gdf_edges_with_hex.iterrows():
                link = ET.SubElement(links, 'link')
                link.set('id', str(edge['id']))
                link.set('from', str(edge['from']))
                link.set('to', str(edge['to']))
                link.set('length', str(edge['length']))
                link.set('freespeed', str(edge['freespeed']))
                
                # Adjust capacity if edge is in scenario hexagons
                if edge['id'] in scenario_edges['id'].values:
                    capacity = float(edge['capacity']) * capacity_tuning_factor
                    link.set('capacity', str(capacity))
                    link.set('scenario_edge', 'true')  # Identifier for scenario edges
                else:
                    link.set('capacity', str(edge['capacity']))
                    link.set('scenario_edge', 'false')
                
                link.set('permlanes', str(edge['permlanes']))
                link.set('oneway', str(edge['oneway']))
                link.set('modes', str(edge['modes']))
            
            # Create the XML tree and save it
            tree = ET.ElementTree(root)
            
            # Save as gzipped XML
            with gzip.open(network_path, 'wb') as f:
                tree.write(f, encoding='utf-8', xml_declaration=True)
            
            total_scenarios += 1
            
            # Print progress
            if total_scenarios % 100 == 0:
                print(f"Created {total_scenarios} network files...")
    
    print(f"\nFinished creating {total_scenarios} network files")
    print(f"Files are organized in folders under: {networks_base}")
    

    

def plot_check_for_created_networks(check_output_subgraph_path, districts_gdf, hexagon_grid_all, gdf_edges_with_hex, scenario_labels, road_type_subsets):
    # Load the network file
    matsim_network, nodes_subgraph, edges_subgraph = matsim_network_to_gdf(check_output_subgraph_path)
    
    # Match the highway_consolidated column from gdf_edges_with_hex to matsim_network
    # Create a mapping dictionary from edge IDs to highway_consolidated values
    highway_mapping = dict(zip(gdf_edges_with_hex['id'], gdf_edges_with_hex['highway_consolidated']))
    
    # Add highway_consolidated column to matsim_network using the mapping
    matsim_network['highway_consolidated'] = matsim_network['id'].map(highway_mapping)
    
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Plot the districts
    districts_gdf.plot(ax=ax, 
                      column='zone_id',
                      cmap='YlGnBu',
                      alpha=0.3,
                      legend=False,
                      legend_kwds={'label': 'District Area (m²)'},
                      label='Districts')
    
    # Plot hexagons
    hexagon_grid_all.plot(ax=ax, color='none', edgecolor='green', alpha=0.3, label='Hexagons')
    
    # Get scenario label from filename
    scenario_label = os.path.basename(check_output_subgraph_path).replace('network_', '').split('.xml.gz')[0]
    key = next((k for k, v in scenario_labels.items() if v == scenario_label), None)
    hex_ids = road_type_subsets[key[0]][key[1]]
    
    # Create mask for scenario edges
    scenario_mask = gdf_edges_with_hex['hexagon'].apply(
        lambda x: any(h in hex_ids for h in x) if isinstance(x, list) else False
    )
    
    # Plot parent network edges (not in scenario)
    parent_edges = matsim_network[~matsim_network['id'].isin(gdf_edges_with_hex[scenario_mask]['id'])]
    parent_edges.plot(ax=ax, 
                     color='gray',
                     linewidth=0.5,
                     alpha=0.5,
                     label='Parent Network')
    
    # Plot all scenario edges
    scenario_edges = matsim_network[matsim_network['id'].isin(gdf_edges_with_hex[scenario_mask]['id'])]
    scenario_edges.plot(ax=ax, 
                       color='red',
                       linewidth=0.5,
                       alpha=0.8,
                       label='Scenario Edges')
    
    # Plot scenario edges that match the road type from scenario label
    road_type = key[0]  # Get the road type from the key
    matching_edges = scenario_edges[scenario_edges['highway_consolidated'] == road_type]
    matching_edges.plot(ax=ax,
                       color='blue',
                       linewidth=0.5,
                       alpha=0.8,
                       label=f'Road Type: {road_type}')
    
    legend_elements = [
        Patch(facecolor='none', edgecolor='green', alpha=0.3, label='Hexagons'),
        Line2D([0], [0], color='gray', linewidth=0.5, alpha=0.5, label='Parent Network'),
        Line2D([0], [0], color='red', linewidth=0.5, alpha=0.8, label='Scenario Edges'),
        Line2D([0], [0], color='blue', linewidth=2, alpha=0.8, label=f'Road Type: {road_type}')
    ]

    ax.legend(handles=legend_elements)
    plt.title(f'MATSim Network with Scenario Edges ({scenario_label})')
    plt.axis('equal')
    plt.show()
    return matsim_network
    
def plot_hexagons_with_ids(gdf_hexagons, gdf_edges=None, title="Hexagon Grid with IDs"):
    """
    Plot hexagons with their IDs and optionally overlay the road network.
    
    Args:
        gdf_hexagons (GeoDataFrame): GeoDataFrame containing hexagon geometries
        gdf_edges (GeoDataFrame, optional): GeoDataFrame containing road network edges
        title (str): Title for the plot
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Plot hexagons
    gdf_hexagons.plot(ax=ax, facecolor='none', edgecolor='black', alpha=0.5)
    
    # Add hexagon IDs
    for idx, row in gdf_hexagons.iterrows():
        # Get the centroid of the hexagon
        centroid = row.geometry.centroid
        # Add the ID text
        ax.text(centroid.x, centroid.y, str(row['grid_id']), 
                ha='center', va='center', fontsize=8)
    
    # Plot road network if provided
    if gdf_edges is not None:
        gdf_edges.plot(ax=ax, color='red', linewidth=0.5, alpha=0.5)
    
    # Set title and remove axes
    ax.set_title(title)
    ax.axis('off')
    
    # Add a legend
    if gdf_edges is not None:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='black', linewidth=1, label='Hexagon Boundaries'),
            Line2D([0], [0], color='red', linewidth=0.5, label='Road Network')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()

def cross_check_for_created_networks(check_output_subgraph_path, gdf_edges_with_hex, road_type_subsets, scenario_labels):
    """
    Cross check the created network files to verify:
    1. Which edges are in the selected hexagons
    2. Which edges match the road type
    3. How capacities have been modified
    """
    # Load the network file
    matsim_network, nodes_subgraph, edges_subgraph = matsim_network_to_gdf(check_output_subgraph_path)
    
    # Get scenario information
    scenario_label = os.path.basename(check_output_subgraph_path).replace('network_', '').split('.xml.gz')[0]
    key = next((k for k, v in scenario_labels.items() if v == scenario_label), None)
    hex_ids = road_type_subsets[key[0]][key[1]]
    
    # Get edges in selected hexagons
    mask = gdf_edges_with_hex['hexagon'].apply(
        lambda x: any(hex_id in x for hex_id in hex_ids) if isinstance(x, list) else False
    )
    edges_in_selected_hexagon = gdf_edges_with_hex[mask]
    
    # Get edges that match both hexagon and road type
    edges_in_selected_hexagon_and_road_type = edges_in_selected_hexagon[
        edges_in_selected_hexagon['highway_consolidated'] == key[0]
    ]
    
    # Create a comparison DataFrame for the capacity changes
    comparison_df = pd.DataFrame({
        'edge_id': edges_in_selected_hexagon['id'],
        'road_type': edges_in_selected_hexagon['highway_consolidated'],
        'original_capacity': edges_in_selected_hexagon['capacity'],
        'modified_capacity': matsim_network[matsim_network['id'].isin(edges_in_selected_hexagon['id'])]['capacity'],
        'capacity_reduced': edges_in_selected_hexagon['highway_consolidated'] == key[0]
    })
    
    return edges_in_selected_hexagon_and_road_type, edges_in_selected_hexagon, comparison_df

def main():
    # Load MATSim network
    matsim_network, nodes_dict, df_edges = matsim_network_to_gdf(matsim_network_file_path)
    # Get network bounds
    bounds = matsim_network.total_bounds
    # Get OSM data
    osm_network = get_osm_roads(bounds, crs=matsim_network.crs)
    # Perform spatial join
    enriched_network ,joined = spatial_join_networks(matsim_network, osm_network)
    
    gdf = gpd.read_file(administrative_boundary_json_path)
    #modify the districts geodataframe
    districts_gdf = modify_districts_geodataframe(gdf)
    #merge the edges with the districts
    gdf_edges_with_districts = merge_edges_with_districts(enriched_network, districts_gdf)
    #generate the hexagon grid for the polygon
    gdf_edges_with_hex,hexagon_grid_all = generate_hexagon_grid_for_districts(districts_gdf, hexagon_size ,gdf_edges_with_districts ,projection='EPSG:25832')
    #check the hexagon statistics
    check_hexagon_statistics(gdf_edges_with_hex, hexagon_grid_all)
    #plot the grid and the edges
    plot_grid_and_edges(gdf_edges_with_hex, hexagon_grid_all, enriched_network,districts_gdf)
    #check the road type distribution
    gdf_with_grouped_road_types = check_road_type_distribution(gdf_edges_with_hex)
    #generate the road type specific subsets
    road_type_subsets = generate_road_type_specific_subsets(gdf_edges_with_hex,seed=13)
    #generate the scenario labels
    scenario_labels = generate_scenario_labels(road_type_subsets)
    #create the scenario networks
    create_scenario_networks(gdf_edges_with_hex, road_type_subsets, scenario_labels, city_name="Augsburg", output_base_path=output_base_path, nodes_dict=nodes_dict)
    #plot the check for the created networks
    matsim_network = plot_check_for_created_networks(check_output_subgraph_path,districts_gdf,hexagon_grid_all,gdf_edges_with_hex,scenario_labels,road_type_subsets)
    #plot the hexagons with ids
    plot_hexagons_with_ids(hexagon_grid_all, title="Hexagon Grid with IDs")
    #cross check the created networks
    edges_with_road_type, edges_in_hexagons, capacity_changes = cross_check_for_created_networks(
    check_output_subgraph_path,
    gdf_edges_with_hex,
    road_type_subsets,
    scenario_labels
    )   
    edges_with_road_type
    edges_in_hexagons
    capacity_changes
