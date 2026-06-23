import streamlit as st
import copy
from agrifoodpy.pipeline import Pipeline

st.set_page_config(page_title="Food Flow Analysis", layout="wide")

# Sidebar
with st.sidebar:
    # -----------------
    # Standard pipeline
    # -----------------
    fs = Pipeline.read("../ffc_pipeline/ffc_pipeline.yaml")


    # -------------------------------
    # Food scaling from land pipeline
    # -------------------------------
    # fs = Pipeline.read("../ffc_pipeline/ffc_pipeline_food_scale_from_land.yaml")
    # from agrifoodpy.utils.scaling import linear_scale
    # lin_scale = linear_scale(
    #     y0=2020,
    #     y1=2020,
    #     y2=2050,
    #     y3=2100,
    #     c_init=1.0,
    #     c_end=0.5 
    # ).sel(Year=[2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100])
    # fs.params[10]["fraction"] = lin_scale

    # ------------------------------------------------
    # Shocks
    # ------------------------------------------------

    year_start_prod = st.slider("Select production shock start year", min_value=2020, max_value=2050, value=2030, step=1)
    year_start_imps = st.slider("Select imports shock start year", min_value=2020, max_value=2050, value=2033, step=1)
    severity_prod_val = st.slider("Select production shock severity (0-1)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    severity_imps_val = st.slider("Select imports shock severity (0-1)", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    fs = Pipeline.read("../ffc_pipeline/ffc_pipeline_with_shocks.yaml")
    # severity = 0.5  # 0.5 means a 50% reduction in production
    from agrifoodpy.utils.scaling import linear_scale
    import xarray as xr
    lin_scale = linear_scale(
        y0=2020,
        y1=2020,
        y2=2050,
        y3=2050,
        c_init=1.0,
        c_end=0.5 
    )
    severity_prod = xr.ones_like(lin_scale)
    severity_imps = xr.ones_like(lin_scale)
    severity_prod.loc[{"Year":[year_start_prod, year_start_prod+1, year_start_prod+2, year_start_prod+3, year_start_prod+4]}] = 1-severity_prod_val
    severity_imps.loc[{"Year":[year_start_imps, year_start_imps+1, year_start_imps+2, year_start_imps+3, year_start_imps+4]}] = 1-severity_imps_val
    fs.params[11]["severity"] = severity_prod
    fs.params[12]["severity"] = severity_imps

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

    # Cache the datablock after node 4 (after loading datasets and before processing)
    if "cached_datablock" not in st.session_state:
        print("Caching datablock...")
        fs.run(to_node=4, timing=True)
        st.session_state["cached_datablock"] = fs.datablock

    # Use the cached datablock for subsequent runs to save time
    fs.datablock = copy.deepcopy(st.session_state["cached_datablock"])

    fs.run(from_node=4, skip=skip, timing=True)

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

