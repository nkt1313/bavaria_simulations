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


# THIS IS CODE FROM THE OLD VERSION, WHICH CAN BE USED AS REFERENCE FOR STEP 5. 

# # Number of files per directory
# files_per_dir = 1000

# output_base_dir = base_dir +  '/networks/'
# # Ensure the base output directory exists
# os.makedirs(output_base_dir, exist_ok=True)

# # Create and save the networks
# for i, combination in enumerate(district_combinations):
#     if isinstance(combination, int):
#         combination = (combination,)
#     df_copy = gdf_edges_with_highway.copy()
#     df_copy['policy_introduced'] = df_copy['district'].apply(
#         lambda districts: any(d in districts for d in combination)
#     ) & df_copy.apply(
#         lambda row: 'car' in row['modes'] and row['osm:way:highway'] in higher_order_roads, axis=1
#     )
    
#     # Modify freespeed and capacity based on the policy_introduced condition
#     df_copy.loc[df_copy['policy_introduced'], 'capacity'] = df_copy.loc[df_copy['policy_introduced'], 'capacity'] / 2

#     # Determine the subdirectory based on the file index
#     dir_index = (i // files_per_dir) + 1
#     subdir_name = f"networks_{dir_index * files_per_dir}"
#     output_dir = os.path.join(output_base_dir, subdir_name)
    
#     # Create the subdirectory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
#     if len(combination) == 1:
#         filename = f"network_d_{combination[0]}.xml.gz"
#     else:
#         filename = f"network_d_{'_'.join(map(str, combination))}.xml.gz"
    
#     # Convert the DataFrame to XML
#     xml_tree = nio.dataframe_to_xml(df_copy, nodes_dict)
    
#     # Write the XML to a compressed .gz file
#     file_path = os.path.join(output_dir, filename)
#     nio.write_xml_to_gz(xml_tree, file_path)

#     # Clear the dataframe from memory
#     del df_copy
#     gc.collect()

# # Example: Display the file paths of the saved files
# output_files = [os.path.join(root, f) for root, _, files in os.walk(output_base_dir) for f in files]
# print(output_files)
