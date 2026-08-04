import agrifoodpy
import numpy as np
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
    # create mask for selected LA
    la_land_mask = la_land["Boundaries"].where(la_land["Boundaries"] == int(la_id))
    # get selected LA population
    la_code = la_land['LAD25CD'].where(la_land['ID']==int(la_id), drop=True).item()
    selected_la_population = la_population.sel(Code=la_code)['All ages'].values
    population_uk = population['Principal'].values

        
    # 2.
    # Use the land data mask to filter the land use dataarray and get relative
    # agricultural values: primarily rel_pasture, rel_arable,
    land_use_uk = land["dominant_aggregate"]
    # get land use for selected LA
    selected_la_land_use = land['dominant_aggregate'].where(la_land_mask==int(la_id), drop=True)
    # maybe useful later/for plots
    list_land_types_id = np.arange(1, land.sizes["aggregate_class"] + 1, dtype=int)
    land_types = dict(zip(list_land_types_id, land["aggregate_class"].values))


    # 3.
    # Scale the FBS data to match the local authority population and land area
    # using the scaling equations from
    # https://docs.google.com/document/d/1wao2BHAf8Z9fbc9hIGZc14oe-6EinfIvEy5zdV5d8sw/edit?tab=t.rlozwpww2649#heading=h.tf6nrdx7e4f
    # Trade will need to be scaled to make sure the FBS is balanced:
    # prod + imports = exports + food + feed + losses + other uses.
    ## PRODUCTION ##
    arable_items_uk = fbs.Item.loc[fbs.Item_origin == "Vegetal Products"].values
    production_arable_items_uk = fbs.sel(Item=arable_items_uk)
    
    production_arable_items_la = production_arable_items_uk['production'] \
        * selected_la_land_use.where(selected_la_land_use==3).count() \
            / land_use_uk.where(land_use_uk==3).count()

    animal_items_uk = fbs.Item.loc[fbs.Item_origin == 'Animal Products'].values
    production_animal_items_uk = fbs.sel(Item=animal_items_uk)
    production_animal_items_la = ??
    ## Where do I find headcounts?? I'm assuming animal headcounds (or is it people's?)

    ##  SEED  ##
    seed_la = fbs['seed'] * production_arable_items_la \
        / production_arable_items_uk['production'] 

    ##  FEED  ##
    feed_la = fbs['feed'] * production_animal_items_la \
        / production_animal_items_uk['production'] 

    ## PROCESSING ##
    tot_production_uk = production_animal_items_uk['production'] \
        + production_arable_items_uk['production']
    tot_production_la = production_animal_items_la \
        + production_arable_items_la
    processing_la = fbs['processing'] * tot_production_la / tot_production_uk

    ## ADD: Stock Variation and Tourist consumption ##

    ## LOSSES ##
    losses_la = fbs['losses'] * tot_production_la / tot_production_uk

    ## ADD: Residual ##

    ## FOOD (Retail) ##
    food_la = fbs['food'] * selected_la_population / population_uk

    # 4.
    # Return the scaled FBS data, the scaled land data, and the scaled
    # population data making sure they follow the exact same structure as the
    # input data.
    
    fbs_scaled = fbs.copy()
    fbs_scaled['production'].loc[arable_items_uk] = production_arable_items_la
    fbs_scaled['production'].loc[animal_items_uk] = production_animal_items_la
    fbs_scaled['seed'] = seed_la
    fbs_scaled['feed'] = feed_la
    fbs_scaled['processing'] = processing_la
    fbs_scaled['stock'] = stock_la
    fbs_scaled['tourist'] = tourist_la
    fbs_scaled['losses'] = losses_la
    fbs_scaled['residual'] = residual_la
    fbs_scaled['food'] = food_la

    land_scaled = selected_la_land_use
    population_scaled = selected_la_population 
    # OR population_scaled['All ages'].values = selected_la_population ??

    return fbs_scaled, land_scaled, population_scaled


