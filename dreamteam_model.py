### IMPORTS AND MACROS #########################################################

import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st

from pyvis.network import Network
from stvis import pv_static
from k_means_constrained import KMeansConstrained

from beam_search import beam_search

ss = st.session_state


### MISC FUNCTIONS #############################################################

# Assign a color to each team
def team_to_color(cluster_id: int, n_clusters: int) -> str:
    r, g, b, a = ss.COLOR_MAP_NODES(cluster_id/n_clusters)
    return f"rgb({round(255*r)},{round(255*g)},{round(255*b)})"

# Assign a color to each edge
def percentage_to_color(f: float) -> str:
    r, g, b, a = ss.COLOR_MAP_EDGES(f)
    return f"rgb({round(255*r)},{round(255*g)},{round(255*b)})"

# Calculate weight of two interest vectors
def calc_prog_weights(v1: list, v2: list) -> float:
    n = len(v1)
    d = sum([max(v1[i], v2[i]) for i in range(n)])
    return d / (5*n)


### MAKE GRAPH WITH WEIGHTS DEPENDING ON FEATURES ##############################

def make_graph(
    role_mult: float,
    interests_mult: float,
    year_mult: float,
    friend_mult: float,
    challenges_mult: float,
    languages_mult: float,
    objective_mult: float,
    availability_mult: float,
    programming_mult: float,
    edge_threshold: float,
    num_nodes: int,
) -> nx.Graph:

    # Select participants
    participants = ss.participants[0:num_nodes]

    # nodes and edges dataframes to better store their properties
    nodes = pd.DataFrame(data=[[p.id, p.name] for p in participants], 
                         columns=['id', 'label'])
    edges = pd.DataFrame(data=[], # Initially empty
                         columns=['id1', 'id2', 'weight'])

    # For each pair of nodes, compute weight for each feature
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):

            p1 = participants[i]
            p2 = participants[j]

            # year_of_study
            year_weight = (5 - abs(p1.year_of_study - p2.year_of_study)) / 5

            # interests
            interests_weight = 0
            for a in p1.interests:
                for b in p2.interests:
                    if a == b: interests_weight += 1
            interests_weight / max(len(p1.interests), len(p2.interests), 1)

            # preferred_role
            role_weight = 1 if p1.preferred_role != p2.preferred_role else 0

            # friend_registration
            friend_weight = 0
            if p1.id in p2.friend_registration: friend_weight += 1
            if p2.id in p1.friend_registration: friend_weight += 1

            # interest_in_challenges
            challenges_weight = 0.0
            for a in p1.interest_in_challenges:
                for b in p2.interest_in_challenges:
                    if a == b: challenges_weight += 1
            challenges_weight /= 3

            # preferred_languages
            languages_weight = 0
            for a in p1.preferred_languages:
                for b in p2.preferred_languages:
                    if a == b: languages_weight = max(languages_weight, 1)

            # objective
            objective_weight = np.dot(np.array(p1.objective), np.array(p2.objective))

            # availability
            availability_weight = 0
            for (a, b) in p1.availability.items():
                if b and p2.availability[a]: availability_weight += 1
            availability_weight /= 5

            # programming_skills
            programming_weight = calc_prog_weights(p1.programming_skills, p2.programming_skills)

            # Sum all pondered weights
            total_weight = (
                year_mult * year_weight
                + interests_mult * interests_weight
                + role_mult * role_weight
                + friend_mult * friend_weight
                + challenges_mult * challenges_weight
                + languages_mult * languages_weight
                + objective_mult * objective_weight
                + availability_mult * availability_weight
                + programming_mult * programming_weight
            )

            # Add edge to dataframe
            edges.loc[len(edges)] = [p1.id, p2.id, total_weight]

    # Normalize edges and remove small ones
    max_weight = max(edges['weight'])
    if max_weight != 0.0: edges['weight'] = edges['weight'] / max_weight
    edges = edges[edges['weight'] > edge_threshold]

    return nodes, edges


### CLUSTERING FUNCTION TO MAKE TEAMS ##########################################

def make_teams(nodes: pd.DataFrame, edges: pd.DataFrame, n_clusters) -> list[int]:
    # Create inverse edges
    inv_edges = edges.copy()
    inv_edges['weight'] = 1 / edges['weight']

    # Create graph with inverse edges
    inv_graph = nx.Graph()
    inv_graph.add_nodes_from(nodes["id"])
    inv_graph.add_edges_from(
        zip(
            inv_edges["id1"],
            inv_edges["id2"],
            inv_edges[["weight"]].to_dict(orient="records"),
        )
    )

    # Clustering parameters
    clf = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=2,
        size_max=4,
        random_state=0
    )

    # Get adjacency matrix from graph
    adj_matrix = nx.to_numpy_array(inv_graph)

    # Calc clusters
    clusters = clf.fit_predict(adj_matrix)
    return clusters


### FUNCTION TO RENDER RESULTS ON STREAMLIT ####################################

def render_graph(graph: nx.Graph, mode: str) -> None:
    # Create net and inherit from graph
    net = Network(height=f"{ss.HEIGHT}px", width=f"{ss.WIDTH}px")
    net.from_nx(graph)

    # Get positions to put nodes in circular layout
    pos = nx.circular_layout(graph, scale=500)

    # For each node assign position and disable gravity
    for node in net.get_nodes():
        net.get_node(node)["x"] = pos[node][0]
        net.get_node(node)["y"] = -pos[node][1]
        net.get_node(node)["physics"] = False

    pv_static(net, name=mode) # Show graph


### PRINT RESULTS AKA GROUP ASSIGNMENTS ########################################

def print_assignments(nodes: pd.DataFrame, num_nodes: int) -> None:
    cols = st.columns(3)
    for i, col in enumerate(cols):
        with col:
            lb = i * (num_nodes//3)
            ub = (i+1) * (num_nodes//3)
            if i == 2: ub = num_nodes
            col_res = nodes[["label", "cluster"]].iloc[lb:ub]
            st.dataframe(
                col_res.set_index(col_res.columns[0]),
                height=35 * len(col_res) + 38,
                use_container_width=True,
            )


### CALCULATE COST OF SOLUTION #################################################

def calc_cost(nodes: pd.DataFrame, edges: pd.DataFrame) -> float:
    cost = 0.0
    for num_cluster in nodes['cluster'].unique():
        team_nodes = nodes[nodes['cluster'] == num_cluster]['id'].to_list()
        team_edges = edges[(edges['id1'].isin(team_nodes)) & (edges['id2'].isin(team_nodes))]
        cost += sum(team_edges['weight'])
    return cost


### ALL PIPELINE FUNCTION ######################################################

def make_cluster_render_graph(
    role_mult: float,
    interests_mult: float,
    year_mult: float,
    friend_mult: float,
    challenges_mult: float,
    languages_mult: float,
    objective_mult: float,
    availability_mult: float,
    programming_mult: float,
    edge_max_width: int,
    edge_threshold: float,
    num_nodes: int,
    n_clusters: int,
    mode: str
) -> None:
    
    if mode == 'kmeans-constrained':

        nodes, edges = make_graph(
            role_mult=          role_mult,
            interests_mult=     interests_mult,
            year_mult=          year_mult,
            friend_mult=        friend_mult,
            challenges_mult=    challenges_mult,
            languages_mult=     languages_mult,
            objective_mult=     objective_mult,
            availability_mult=  availability_mult,
            programming_mult=   programming_mult,
            edge_threshold=     edge_threshold,
            num_nodes=          num_nodes
        )
        clusters = make_teams(nodes=nodes, edges=edges, n_clusters=n_clusters)


    elif mode == 'beam-search':
        clusters, nodes, edges = beam_search(
            role_mult=          role_mult,
            interests_mult=     interests_mult,
            year_mult=          year_mult,
            friend_mult=        friend_mult,
            challenges_mult=    challenges_mult,
            languages_mult=     languages_mult,
            objective_mult=     objective_mult,
            availability_mult=  availability_mult,
            programming_mult=   programming_mult,
            edge_threshold=     edge_threshold,
            num_nodes=          num_nodes
        )

    # Node properties
    nodes['cluster'] = clusters
    nodes['color'] = nodes['cluster'].apply(team_to_color, args=(n_clusters,))
    nodes['size'] = 20

    # Edge Properies
    edges['color'] = edges['weight'].apply(percentage_to_color)
    edges['weight'] = round(edges['weight'] * edge_max_width, 0)

    if mode == 'beam-search':
        edges[['id1', 'id2']] = edges[['id1', 'id2']].astype('int64')

    # Create graph
    graph = nx.Graph()
    graph.add_nodes_from(zip(nodes["id"], nodes[['label', 'color', 'size']].to_dict(orient="records")))
    graph.add_edges_from(zip(edges['id1'], edges['id2'], edges[['weight', 'color']].to_dict(orient='records')))

    # Display cost
    cost = calc_cost(nodes, edges)
    st.write('Cost of solution:', cost)

    # Render
    render_graph(graph=graph, mode=mode)

    # Print group assignments
    st.subheader('Group Assigments:', divider='red')
    print_assignments(nodes=nodes, num_nodes=num_nodes)
