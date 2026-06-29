from agrifoodpy.pipeline import Pipeline
import streamlit as st

@st.cache_data
def cache_datablock(pipeline_name, _fs):
    print("Caching datablock...")
    _fs.run(to_node=5, timing=True)
    print("Datablock cached.")
    return _fs


def configure_pipeline(pipeline_name):

    fs = Pipeline.read(pipeline_name)
    
    fs = cache_datablock(pipeline_name, fs)

    # Standard pipeline

    if pipeline_name == "../ffc_pipeline/ffc_pipeline.yaml":
        pass

    # Food scaling from land pipeline
    if pipeline_name == "../ffc_pipeline/ffc_pipeline_food_scale_from_land.yaml":
        
        from agrifoodpy.utils.scaling import linear_scale
        lin_scale = linear_scale(
            y0=2020,
            y1=2020,
            y2=2050,
            y3=2100,
            c_init=1.0,
            c_end=0.5 
        ).sel(Year=[2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100])
        fs.params[10]["fraction"] = lin_scale


    # Pipeline with shocks
    if pipeline_name == "../ffc_pipeline/ffc_pipeline_with_shocks.yaml":

        year_start_prod = st.slider("Select production shock start year", min_value=2020, max_value=2050, value=2030, step=1)
        year_start_imps = st.slider("Select imports shock start year", min_value=2020, max_value=2050, value=2033, step=1)
        severity_prod_val = st.slider("Select production shock severity (0-1)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        severity_imps_val = st.slider("Select imports shock severity (0-1)", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

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

    return fs

