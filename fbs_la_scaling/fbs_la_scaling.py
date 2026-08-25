import agrifoodpy
import numpy as np
import xarray as xr
from agrifoodpy.pipeline import pipeline_node


# NOTE: these cathegories are for FAOSTAT data
# need to find a way to generalise them
SCALE_WITH_PASTURE_ITEMS = ['Bovine Meat', 'Mutton & Goat Meat', 'Meat, Other', 
                  'Offals, Edible', 'Fats, Animals, Raw', 'Butter, Ghee', 
                  'Cream', 'Animal Products', 'Meat', 'Offals', 'Animal fats',
                  'Milk - Excluding Butter']
SCALE_WITH_HEADCOUNT_ITEMS = ['Pigmeat', 'Poultry Meat', 'Eggs']


@pipeline_node(["fbs", "population", "land", "la_population", "la_land", 
                "scale_with_pasture", "scale_with_headcounts"])
def fbs_la_scaling(
    fbs,
    population,
    land,
    la_population,
    la_land,
    la_id,
    livestock_la_data, # NOTE: uses agrifoodpy_data.food UK_LIVESTOCK_LAD structure
    scale_with_pasture=SCALE_WITH_PASTURE_ITEMS,
    scale_with_headcounts=SCALE_WITH_HEADCOUNT_ITEMS,
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
    # selected_la_population = la_population.sel(Code=la_code)['All ages'].values
    selected_la_population = la_population["All ages"].reindex(Code=[la_code]).values[0]
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
    ## PRODUCTION - Plants ##
    arable_items_uk = fbs.Item.loc[fbs.Item_origin == "Vegetal Products"].values
    production_arable_items_uk = fbs.sel(Item=arable_items_uk)
    
    production_arable_items_la = production_arable_items_uk['production'] \
        * selected_la_land_use.where(selected_la_land_use==3).count() \
            / land_use_uk.where(land_use_uk==3).count()

    ## PRODUCTION - Animals ##
    animal_items_uk = fbs.Item.loc[fbs.Item_origin == 'Animal Products'].values
    production_animal_items_uk = fbs.sel(Item=animal_items_uk)

    # scale ruminant production (includes dairy) with pasture land use
    ruminant_items_uk = fbs.Item.loc[fbs.Item_name.isin(scale_with_pasture)].values
    production_ruminant_items_uk = fbs.sel(Item=ruminant_items_uk)
    production_ruminant_items_la = production_ruminant_items_uk['production'] \
        * selected_la_land_use.where(selected_la_land_use==2).count() \
            / land_use_uk.where(land_use_uk==2).count()
    
    # scale poultry and pigmeat production with headcount
    poultry_pigmeat_items = fbs.Item.loc[fbs.Item_name.isin(scale_with_headcounts)].values
    production_poultry_pigmeat_items_uk = fbs.sel(Item=poultry_pigmeat_items)

    # read headcounts 
    headcount_la = livestock_la_data['Livestock counts'] \
        .sel(Species=['Poultry', 'Pigs'], Code=la_code) \
        .sum() \
        .item()
    headcount_uk = livestock_la_data['Livestock counts'] \
        .sel(Species=['Poultry', 'Pigs']) \
        .sum() \
        .item()

    production_poultry_pigmeat_items_la = \
        production_poultry_pigmeat_items_uk['production'] \
        * headcount_la / headcount_uk

    # scale remaining animal products with population
    ruminant_and_poultry_pigmeat = np.concatenate([ruminant_items_uk, poultry_pigmeat_items])
    other_animal_items = np.setdiff1d(animal_items_uk, ruminant_and_poultry_pigmeat)
    production_other_animal_items_uk = fbs.sel(Item=other_animal_items)
    production_other_animal_items_la = production_other_animal_items_uk['production'] \
        * selected_la_population / population_uk

    ##  SEED  ##
    # tot animal production uk and la
    production_animal_items_la = xr.concat([production_ruminant_items_la, 
                                            production_poultry_pigmeat_items_la,
                                            production_other_animal_items_la], 
                                            dim="Item")
    production_animal_items_la = production_animal_items_la.reindex(Item=animal_items_uk)
    production_animal_items_uk = xr.concat([production_ruminant_items_uk, 
                                            production_poultry_pigmeat_items_uk,
                                            production_other_animal_items_uk], 
                                            dim="Item")
    production_animal_items_uk = production_animal_items_uk.reindex(Item=animal_items_uk)


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

    ## IMPORTS and EXPORTS ##
    imports_la = fbs['imports'] * selected_la_population / population_uk
    exports_la = fbs['exports'] * tot_production_la / tot_production_uk

    ## Stock Variation and Tourist consumption ##
    # should we use  population_LA_percentage = LA_pop_arr / LA_pop_arr.sum() instead?
    tourist_la = fbs['tourist'] * selected_la_population / population_uk
    stock_la = fbs['domestic'] * selected_la_population / population_uk

    ## LOSSES ##
    losses_la = fbs['losses'] * tot_production_la / tot_production_uk
    ## Residual ##
    residual_la = fbs['residual'] * tot_production_la / tot_production_uk

    ## FOOD (Retail) ##
    food_la = fbs['food'] * selected_la_population / population_uk

    ## OTHER ##
    other_la = fbs['other'] * selected_la_population / population_uk

    # 4.
    # Return the scaled FBS data, the scaled land data, and the scaled
    # population data making sure they follow the exact same structure as the
    # input data.
    fbs_scaled = fbs.copy(deep=True)
    fbs_scaled['production'].loc[arable_items_uk] = production_arable_items_la
    fbs_scaled['production'].loc[animal_items_uk] = production_animal_items_la
    fbs_scaled['imports'] = imports_la
    fbs_scaled['exports'] = exports_la
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

    ## NOTE: check overall balance between production, import and export

    return fbs_scaled, land_scaled, population_scaled


