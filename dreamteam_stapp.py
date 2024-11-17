### IMPORTS ####################################################################

import streamlit as st
import pandas as pd
from matplotlib import colormaps

from participant import load_participants
from dreamteam_model import make_cluster_render_graph, print_assignments


### CONFIG AND MACROS ##########################################################

st.set_page_config(layout='wide')
ss = st.session_state


### SETUP ######################################################################

@st.cache_data
def setup() -> None:
    # Parameters
    ss.HEIGHT = 600
    ss.WIDTH = 1000
    ss.EDGE_MAX_WIDTH = 10
    ss.AV_GROUP_SIZE = 3.2
    ss.N_COLORS = 100
    ss.COLOR_MAP_NODES = colormaps["hsv"]
    ss.COLOR_MAP_EDGES = colormaps["RdYlGn"]
    # Load data
    ss.participants = load_participants("data/datathon_participants_final.json")
    ss.assignments = pd.read_csv("data/results.csv", index_col=False)

setup() # Executed once per run


### MAIN PROGRAM ###############################################################

### SIDEBAR

with st.sidebar:
    st.header('Multipliers:')
    col1, col2 = st.columns(2)

    with col1:
        st.slider("year_of_study", min_value=0.0, max_value=1.0, value=0.5, key="year_mult")
        st.slider("interests", min_value=0.0, max_value=1.0, value=0.5, key="interests_mult")
        st.slider("preferred_role", min_value=0.0, max_value=1.0, value=0.5, key="role_mult")
        st.slider("objective", min_value=0.0, max_value=1.0, value=0.5, key="objective_mult")
        st.slider("programming_skills", min_value=0.0, max_value=1.0, value=0.5, key="programming_mult")

    with col2:
        st.slider("friend_registration", min_value=0.0, max_value=1.0, value=0.5, key="friend_mult")
        st.slider("interest_in_challenges", min_value=0.0, max_value=1.0, value=0.5, key="challenges_mult")
        st.slider("preferred_languages", min_value=0.0, max_value=1.0, value=0.5, key="languages_mult")
        st.slider("availability", min_value=0.0, max_value=1.0, value=0.5, key="availability_mult")

    st.header('Parameters:')
    st.slider("EDGE THRESHOLD", min_value=0.05, max_value=1.0, value=0.5, key="edge_threshold")
    st.slider("NUM NODES", min_value=10, max_value=50, value=25, key="num_nodes")


### MAIN PAGE

tab1, tab2, tab3 = st.tabs(['Kmeans Constrained', 'Beam Search', 'ALL Participants Assignment'])

with tab1:
    st.subheader('Compatibility Graph', divider='red')

    if ss.num_nodes <= 50:
        make_cluster_render_graph(
            year_mult=          ss.year_mult,
            interests_mult=     ss.interests_mult,
            role_mult=          ss.role_mult,
            friend_mult=        ss.friend_mult,
            languages_mult=     ss.languages_mult,
            challenges_mult=    ss.challenges_mult,
            objective_mult=     ss.objective_mult,
            availability_mult=  ss.availability_mult,
            programming_mult=   ss.programming_mult,
            edge_max_width=     ss.EDGE_MAX_WIDTH,
            edge_threshold=     ss.edge_threshold,
            num_nodes=          ss.num_nodes,
            n_clusters=         round(ss.num_nodes / ss.AV_GROUP_SIZE),
            mode=               'kmeans-constrained'
        )
    else:
        st.error('Too many nodes to display!')

with tab2:
    st.subheader("Compatibility Graph", divider="red")

    if ss.num_nodes <= 50:
        make_cluster_render_graph(
            year_mult=          ss.year_mult,
            interests_mult=     ss.interests_mult,
            role_mult=          ss.role_mult,
            friend_mult=        ss.friend_mult,
            languages_mult=     ss.languages_mult,
            challenges_mult=    ss.challenges_mult,
            objective_mult=     ss.objective_mult,
            availability_mult=  ss.availability_mult,
            programming_mult=   ss.programming_mult,
            edge_max_width=     ss.EDGE_MAX_WIDTH,
            edge_threshold=     ss.edge_threshold,
            num_nodes=          ss.num_nodes,
            n_clusters=         round(ss.num_nodes / ss.AV_GROUP_SIZE),
            mode=               'beam-search'
        )
    else:
        st.error("Too many nodes to display!")

with tab3:
    print_assignments(ss.assignments, num_nodes=len(ss.participants))
