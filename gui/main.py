import streamlit as st
import copy
from pipeline_tools import configure_pipeline
import glob

st.set_page_config(page_title="Food Flow Analysis", layout="wide")

# Sidebar
with st.sidebar:
    
    available_pipelines = glob.glob("../ffc_pipeline/*.yaml")
    pipeline_name = st.selectbox("Select a pipeline", available_pipelines)
    
    fs = configure_pipeline(pipeline_name)

    st.title("FFC Pipeline")

    # Toggle switches for each node
    skip = []
    for i, node in enumerate(fs.names):
        col_node_toggle, col_node_name = st.columns([1, 6], vertical_alignment="center")
        with col_node_name:
            st.write(f"Step {i + 1}: {node}")
        with col_node_toggle:
            if not st.toggle(f"Run", value=True, key=f"node_{i}", label_visibility="collapsed"):
                skip.append(i)

    # Run the pipeline without the cached nodes
    fs.run(from_node=5, skip=skip, timing=True)

    st.session_state["datablock"] = fs.datablock

@st.fragment
def page_navigation():
    pg = st.navigation(
        [
            st.Page("pages/1_food_balance_sheet.py", title="Food Balance Sheet"),
            st.Page("pages/2_population.py", title="Population"),
            st.Page("pages/3_land_use.py", title="Land Use"),
            st.Page("pages/4_emission_factors.py", title="Emission Factors"),
            st.Page("pages/5_nutrients.py", title="Nutrients"),
            st.Page("pages/6_inspect_datablock.py", title="Inspect Datablock"),
        ],
        position="top"
    )

    pg.run()

page_navigation()

