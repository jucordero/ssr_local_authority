import xarray as xr
import numpy as np

def da_to_map(da, segment):
    """ Generates a segmentation map with values from an input DataArray
    
    Parameters
    ----------

    da : xarray.DataArray
        Input DataArray with values to be used to generate the segmentation map.
    segment : xarray.DataArray
        Input DataArray with segment definitions.

    Returns
    -------
    xarray.DataArray
        Segmentation map with values from the input DataArray.
    """

    # Get the unique values in the segment array
    seg_dict = segment.attrs
    output = xr.zeros_like(segment)

    for reg, id in seg_dict.items():
        reg_mask = (segment != id)

        val = xr.where(reg_mask, 0, da.sel(Region=reg))
        output += val

    output = output.where(segment > 0)

    return output

def da_to_gpd(da, gdb, da_region_col, gdb_region_col):
    """ Appends values to a geopandas dataframe from an input xarray.DataArray.

    Parameters
    ----------

    da : xarray.DataArray
        Input DataArray with values to be appended to the geopandas dataframe.
    gdb : geopandas.GeoDataFrame
        Input geopandas dataframe with a column to match the values in the
        input DataArray.
    """

    name = da.name
    gdb[name] = np.nan

    for reg in da[da_region_col].values:
        reg_mask = (gdb[gdb_region_col] == reg)
        gdb.loc[reg_mask, name] = da.sel(Region=reg).values

    return gdb

