import agrifoodpy
import xarray as xr
from agrifoodpy.pipeline import pipeline_node

@pipeline_node(["fbs", "population", "land", "la_population", "la_land"])
def fbs_la_scaling(
    fbs,
    population,
    land,
    la_population,
    la_land,
    la_id
    ):
    """
    Scale FBS data to match the population and land area.

    Parameters
    ----------
    fbs : xr.Dataset
        Food balance sheet data to be scaled, containing relevant food supply
        elements.
    population : xr.DataArray
        The population data array.
    land : xr.DataArray
        The land data array containing the land use categories per pixel.
    la_population : xr.DataArray
        The local authority population data array containing the population per
        region.
    la_land : xr.DataArray
        The local authority land data array containing the segmentation map to
        identify each region.
    la_id : str
        The local authority ID to be used for scaling.

    Returns
    -------
    tuple of xr.Dataset, xr.DataArray, xr.DataArray
        The scaled FBS data, the scaled land data, and the scaled population
        data.
    """
    # 1.
    # Extract the local authority population and land data mask for the
    # specified ID
    
    # 2.
    # Use the land data mask to filter the land use dataarray and get relative
    # agricultural values: primarily rel_pasture, rel_arable,

    # 3.
    # Scale the FBS data to match the local authority population and land area
    # using the scaling equations from
    # https://docs.google.com/document/d/1wao2BHAf8Z9fbc9hIGZc14oe-6EinfIvEy5zdV5d8sw/edit?tab=t.rlozwpww2649#heading=h.tf6nrdx7e4f
    # Trade will need to be scaled to make sure the FBS is balanced:
    # prod + imports = exports + food + feed + losses + other uses.

    # 4.
    # Return the scaled FBS data, the scaled land data, and the scaled
    # population data making sure they follow the exact same structure as the
    # input data.
    
    return fbs_scaled, land_scaled, population_scaled