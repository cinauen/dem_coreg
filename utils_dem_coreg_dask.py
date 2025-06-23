

import os
import sys
import logging

import datetime as dt
import numpy as np
import xarray
import rioxarray
import shapely.geometry as shap_geom
import rasterio
from rasterio.enums import Resampling
from rioxarray.merge import merge_arrays
import osgeo
from osgeo import osr
from osgeo import ogr
import geojson
import affine

import matplotlib.pyplot as plt

import cProfile
import pstats

from dask import delayed
from dask import compute


def img_read(FILE_P, NODATA, EPSG_INP, chunks=None):
    """
    Read a raster image using rioxarray.

    Parameters:
    FILE_P (str):
        Path to the directory containing the image file.
    FILE_N (str):
        Name of the image file.
    NODATA (float, optional):
        Value to be considered as no data.
        If NONE, then the nodata values specified in the raster will be
        used (img.rio.nodata).
    EPSG_INP (int):
        EPSG code (e.g. 32603) of the target coordinate reference system. Is only
        used if the image has no specified coorindate system

    Returns:
    xarray.DataArray: The read raster image with nodata values set to
        NaN and CRS set if not already present.
    """
    if chunks is None:
        chunks = 'auto'

    img = rioxarray.open_rasterio(
        FILE_P, masked=False, chunks=chunks)

    if NODATA is not None:
        # set nodata value to if specified
        img = img.rio.write_nodata(NODATA, inplace=True)
    if not np.isnan(img.rio.nodata):
        # set nodata value to nan
        img = set_fill_val_to_nan(img)

    if img.rio.crs is None:
        img = img.rio.write_crs(EPSG_INP)

    return img


def img_preproc(img, res_out, AOI_coords, EPSG_TARGET, NODATA,
                resampling_type='bilinear'):
    """
    Preprocesses an image by adjusting its resolution, clipping it to
    an Area of Interest (AOI), and padding it to match the AOI bounds.

    Parameters:
    img (raster image, rioxarray):
        The input image to be preprocessed.
    res_out (float):
        The desired output resolution.
    AOI_coords (list):
        The coordinates defining the Area of Interest.
        Derived from AOI with read_transl_geojson_AOI_coords_single()
    EPSG_TARGET (int):
        The target EPSG code for reprojection. E.g. 32603
    NODATA (int):
        The value to use for nodata pixels.
    resampling_type (str, optional):
        The resampling type to use. Defaults to 'bilinear'.
        other options cubic nearest etc.)

    Returns:
    raster image: The preprocessed image.
    """
    # adjust resolution and EPSG
    img = img.rio.reproject(
            "EPSG:" + str(EPSG_TARGET), nodata=NODATA,
            resampling=Resampling[resampling_type], resolution=res_out)

    # clip to AOi
    img = clip_to_aoi(
        img, AOI_coords, EPSG_TARGET,
        from_disk=False, drop_na=True)

    # for xdem images need to have same shape therefore pad to box
    img = pad_img_to_bounds(img, AOI_coords)

    return img

@delayed
def img_read_preproc(file_path, img_key, NODATA, res_out, AOI_coords, EPSG_INP,
                     EPSG_TARGET, band_names=None, bands_keep=None,
                     resampling_type='bilinear', from_disk=True,
                     chunks=None, new_origin=None):
    """
    Preprocesses an image by adjusting its resolution, clipping it to
    an Area of Interest (AOI), and padding it to match the AOI bounds.

    Parameters:
    img (raster image, rioxarray):
        The input image to be preprocessed.
    res_out (float):
        The desired output resolution.
    AOI_coords (list):
        The coordinates defining the Area of Interest.
        Derived from AOI with read_transl_geojson_AOI_coords_single()
    EPSG_TARGET (int):
        The target EPSG code for reprojection. E.g. 32603
    NODATA (int):
        The value to use for nodata pixels.
    resampling_type (str, optional):
        The resampling type to use. Defaults to 'bilinear'.
        other options cubic nearest etc.)
    from_disk: !!! should be set forto False for more precise results
        but True is much faster
    chunks: dict
        chunks for dask is not chunking use None for auto chunking
        use 'auto'. Otherwise, specify e.g. as
        {'band': 4, 'x': 5120, 'y': 12400} (best to keep it as large
        as possible but still fitting in memory)
    new_origin: tuple
        new origin fo projection (x_orig, y_orig)

    Returns:
    raster image: The preprocessed image.
    """
    # read file
    img = img_read(file_path, NODATA, EPSG_INP, chunks=chunks)

    if band_names is not None:
        img = img.assign_coords(
            band=band_names['_'.join(img_key.split('_')[:2])])

    if bands_keep is not None:
        img = img.sel(
            band=bands_keep['_'.join(img_key.split('_')[:2])])

    # add band names to attibutes
    img.attrs.update({"long_name": list(img.coords['band'].values)})

    ## adjust resolution and EPSG
    #if img.rio.resolution()[0] != res_out:
    #    img = img.rio.reproject(
    #            "EPSG:" + str(EPSG_TARGET), nodata=NODATA,
    #            resampling=Resampling[resampling_type],
    #            resolution=res_out)

    # adjust resolution and EPSG and transform to same origin
    img = img_transform(img, res_out, EPSG_TARGET,
                        resampling_type=resampling_type,
                        new_origin=new_origin)

    # clip to AOi
    img = clip_to_aoi(
        img, AOI_coords, EPSG_TARGET,
        from_disk=from_disk, drop_na=True)

    # for xdem images need to have same shape therefore pad to box
    img = pad_img_to_bounds(img, AOI_coords)

    return img, img_key


@delayed
def reproject_match(img, ref_img, img_key, resampling_type):
    out = img.rio.reproject_match(
            ref_img,
            Resampling=Resampling[resampling_type])
    return out , img_key


@delayed
def merge_img(img_lst, img_key):
    ''''''
    # ------- merge images
    if len(img_lst) > 1:
        img_merged = merge_arrays(img_lst, method='first')
    else:
        img_merged = img_lst[0]

    return img_merged, img_key



def img_save(img, file_p):
    ''''''
    # ------- saved preprocessed image to file
    img.rio.to_raster(raster_path=file_p)

    return


def get_inp_lst_parallel_proc(ref_path, ref_fname, ref_nodata,
                target_path, target_fname, target_nodata):
    fp_lst = []
    file_n_lst = []
    img_key = []
    img_type = []
    nodata_inp = []
    for i_key, i_fname_lst in ref_fname.items():
        for e, i in enumerate(i_fname_lst):
            fp_lst.append(os.path.join(ref_path, i))
            file_n_lst.append(i)
            img_key.append(f"REF_{i_key}_{e}")
            img_type.append(f"REF_{i_key}")
            nodata_inp.append(ref_nodata[i_key])

    for i_key, i_fname_lst in target_fname.items():
        for e, i in enumerate(i_fname_lst):
            fp_lst.append(os.path.join(target_path, i))
            file_n_lst.append(i)
            img_key.append(f"TAR_{i_key}_{e}")
            img_type.append(f"TAR_{i_key}")
            nodata_inp.append(target_nodata[i_key])

    return fp_lst, file_n_lst, img_key, img_type, nodata_inp


# def merge_reproj_save(ref_lst, target_lst, path_out, ref_name_prefix,
#                       target_name_prefix):
#     # ------- merge images



#     ref_merged = {}
#     for i_key, i_img in ref.items():
#         ref_merged[i_key] = merge_arrays(i_img, method='first')

#     target_merged = {}
#     for i_key, i_img in target.items():
#         target_merged[i_key] = merge_arrays(i_img, method='first')

#     # ---- make sure that the target and the reference image have exactly the
#     # coordinate system by projectoing the target to the same coords as the
#     # reference image
#     for i_key, i_img in target_merged.items():
#         target_merged[i_key] = target_merged[i_key].rio.reproject_match(
#             ref_merged[i_key], Resampling=Resampling.bilinear)

#     # ------- saved preprocessed image to file
#     fpath_ref_out = {}
#     for i_key, i_img in ref_merged.items():
#         fpath_ref_out[i_key] = os.path.join(
#             path_out, f"{ref_name_prefix.split('.')[0]}_{i_key}_preproc.tif")
#         ref_merged[i_key].rio.to_raster(
#             raster_path=fpath_ref_out[i_key], write_nodata=True)

#     fpath_target_out = {}
#     for i_key, i_img in target_merged.items():
#         fpath_target_out[i_key] = os.path.join(
#             path_out, f"{target_name_prefix.split('.')[0]}_{i_key}_preproc.tif")
#         target_merged[i_key].rio.to_raster(
#             raster_path=fpath_target_out[i_key], write_nodata=True)

#     # ----- create a mask
#     # (mask is True where have valid data, is False where eiter the data
#     # on the reference or the target image is missing)

#     # mask_inv is for asorics
#     count = 0
#     for i_key, i_img in ref_merged.items():
#         if count == 0:
#             mask = i_img.where(np.isnan(i_img), False, True)
#             mask_inv = i_img.where(np.isnan(i_img), 1, 0)
#         else:
#             mask = np.where(np.isnan(mask) | np.isnan(i_img.values), False, True)
#             mask_inv = np.where(np.isnan(mask) | np.isnan(i_img.values), 1, 0)
#         count += 1

#     for i_key, i_img in target_merged.items():
#         # mask is already existing
#         mask = np.where(np.isnan(mask) | np.isnan(i_img.values), False, True)
#         mask_inv = np.where(np.isnan(mask) | np.isnan(i_img.values), 1, 0)

#     mask_inv_save_path = os.path.join(path_out, 'arosics_coregmask.tif')
#     mask_inv.rio.to_raster(mask_inv_save_path)

#     return fpath_ref_out, fpath_target_out, mask_inv_save_path


def get_bounds_lbrt(coords):
    '''
    coords can are of form [[x1, y1], [x2, y2], [x3, y3], ...]
    output is [left, bottom, right, top]
    '''

    lb = np.min(np.array(coords), axis=0).tolist()
    tr = np.max(np.array(coords), axis=0).tolist()

    return lb + tr


def pad_img_to_bounds(ref_img, AOI_coords):
    '''
    This function can be useful for reproject match since
    if ref files does not cover the full area then the output image is
    clipped to the extent of the reference image
    '''
    bounds_inp = get_bounds_lbrt(AOI_coords)
    ref_im_out = ref_img.rio.pad_box(
        minx=bounds_inp[0],
        miny=bounds_inp[1],
        maxx=bounds_inp[2],
        maxy=bounds_inp[3])

    return ref_im_out


def read_transl_geojson_AOI_coords_single(file_name, target_crs=None):
    '''
    reads shapefile (but will only get first element --> thus assumes
    that shapefile contains only one polygon
    '''
    with open(file_name) as f:
        gj = geojson.load(f)

    coords = np.squeeze(gj['features'][0]['geometry']['coordinates'])
    crs = gj.crs['properties']['name'].split(':')[-1]  # output is string

    if not isinstance(target_crs, str):
        target_crs = str(target_crs)

    if target_crs == crs or target_crs == 'None':
        coords_out = coords.tolist()
    else:
        os_E, os_N = coord_transformation(
            coords, in_syst=crs, out_syst=target_crs)

        coords_out = np.array([os_E, os_N]).T.tolist()

    poly = shap_geom.Polygon(coords_out)

    return coords_out, poly


def coord_transformation(coord_array, in_syst=4326, out_syst=3857):
    '''
    Convert coordinate to new coordinate system
    use EPSG numbers:
    WGS 84: 4326
    WGS 84 / Pseudo-Mercator -- Spherical Mercator: 3857
    '''
    long, lat = coord_array[:, 0], coord_array[:, 1]
    # input SpatialReference
    inSpatialRef = define_spatial_ref()  # osr.SpatialReference() # Establish its coordinate encoding
    inSpatialRef.ImportFromEPSG(int(in_syst))

    # output SpatialReference
    outSpatialRef = define_spatial_ref()  # osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(int(out_syst))

    # create the CoordinateTransformation
    coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

    lat = np.atleast_2d(lat)
    long = np.atleast_2d(long)
    grid_shape = np.shape(lat)
    os_N = np.zeros(grid_shape)
    os_E = np.zeros(grid_shape)
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):

            point = ogr.CreateGeometryFromWkt('POINT (' + str(long[i, j]) + ' ' \
                                                        + str(lat[i, j]) + ')')
            point.Transform(coordTrans)
            dump = point.ExportToWkt()[point.ExportToWkt().find('(')+1:point.ExportToWkt().find(')')].split(' ')
            os_E[i, j] = float(dump[0])
            os_N[i, j] = float(dump[1])

    return np.squeeze(os_E), np.squeeze(os_N)


def define_spatial_ref():
    '''
    Define spatial reference this is because x & y axis order was
    changed in GDAL version 3 (see:
    https://github.com/OSGeo/gdal/issues/1546)
    '''
    # input SpatialReference
    inSpatialRef = osr.SpatialReference() # Establish its coordinate encoding
    if int(osgeo.__version__[0]) >= 3:
        # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
        inSpatialRef.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    return inSpatialRef


def set_fill_val_to_nan(img):

    # get invalid values mask
    mask_invalid = get_invalid_mask(
        img, how='all_invalid')

    # set invalid values to nan
    img = xarray_change_fill_val(
        img, mask_invalid,
        fill_val_new=np.nan)

    return img


def get_invalid_mask(im_inp, how='all_invalid', fill_val=None):
    '''
    get raster mask masking invalid values
        False = invalid
        True = ok

    Parameters
    ----------
    im_inp : raster image (rioxarray)
        Input raster image.
    how : str, optional
        Method to determine invalid values (default is 'all_invalid').
        Options:
            'all_invalid': Pixel is invalid if all band has an invalid value.
            'any_invalid': Pixel is invalid if any of the bands have an
                           invalid value.
            'single': Return a mask for each band separately.
    fill_val : int or float, optional
        Value to consider as invalid (default is the image's nodata value).

    Returns
    -------
    mask_array : numpy array
        Boolean mask where False indicates invalid values and True
        indicates valid values.

    Notes
    -----
    !!! If the input image does not have a nodata value set, an error
    is raised.
    '''

    if fill_val is None:
        fill_val = im_inp.rio.nodata
    if im_inp.rio.nodata is None:
        sys.exit('!!! ' + str(im_inp.band.values[0])
              + ": rio.nodata not set can NOT get invalid mask")

    if how == 'all_invalid':
        # set to True if the pixel value of at least one band is valid
        # thus pixel is only flagged as invalid if all bands are invalid
        if np.isnan(fill_val):
            mask_array = np.any(~np.isnan(im_inp.values), axis=0)
        else:
            mask_array = np.any(im_inp.values != fill_val, axis=0)
    elif how == 'any_invalid':
        # set to True if the pixel values of all band are valid
        # thus pixel is flagged as invalid one of the bands have an invalid value
        if np.isnan(fill_val):
            mask_array = np.all(~np.isnan(im_inp.values), axis=0)
        else:
            mask_array = np.all(im_inp.values != fill_val, axis=0)
    elif how == 'single':
        if np.isnan(fill_val):
            mask_array = ~np.isnan(im_inp.values)
        else:
            mask_array = im_inp.values != fill_val
    else:
        print('!!!! type of replacement undefined !!!!')

    return mask_array


def xarray_change_fill_val(im_inp, mask_array, fill_val_new=np.nan):
    '''
    change fill value to a different one

    INPUTS:
    imp_inp (raster rioxarray)
        image for which fill value should be changes
    mask_array: (raster mask)
        masks values which should be kept (True). Can be created with
        get_invalid_mask()
    '''

    # fill_val = im_inp._FillValue
    out = im_inp.where(mask_array, other=fill_val_new)
    encoding_orig = im_inp.encoding
    out.rio.update_encoding(encoding_orig, inplace=True)

    # make sure nodata value is propery set (out.rio.nodata)
    out.rio.write_nodata(fill_val_new, inplace=True)

    # make sure that encoding dtype is corect
    out.encoding['rasterio_dtype'] = out.dtype

    return out


def clip_to_aoi(rds, AOI_COORDS, AOI_EPSG=None, from_disk=True,
                drop_na=True):
    '''
    --- clip raster to area of interest defined by edge coordinates (AOI_COORDS)
    INPUTS
    aoi_coords: (list or array)
        Dimension: [[x1, y1], [x2, y2], ...]

    AOI_EPSG: (int)
        epsg of the clip geom if none it is assumed to be the same
        as the dataset (e.g. 32603)

    from_disk (bool):
        if run cliping from disk (thus do not read al intp memory)
        !!! from disk is much faster for large files !! but might cause some
        small inconsistencies at border..
        https://corteva.github.io/rioxarray/stable/examples/clip_geom.html

    drop_na (bool):
        If drop values outside clip area. Otherwise, will get same
        raster but with clipped values masked

    check option with geopandas:
    https://corteva.github.io/rioxarray/stable/examples/clip_geom.html

    '''
    # define shape for clipping
    geometries = [
        {
            'type': 'Polygon',
            'coordinates': [AOI_COORDS]
        }]

    # assign EPSG to the AOI
    if (AOI_EPSG is not None and AOI_EPSG != 'None'
        and AOI_EPSG != 0 and AOI_EPSG != '0'):
        AOI_EPSG = int(AOI_EPSG)
    else:
        AOI_EPSG = rds.rio.crs.to_epsg()

    try:
        clipped = rds.rio.clip(
            geometries, crs="EPSG:" + str(AOI_EPSG),
            from_disk=from_disk, drop=drop_na)
    except:
        return None

    if rds.dtype != clipped.dtype:
        # this can be required as from_disk=True might changes nodata to
        # nan (float)
        clipped = clipped.astype(rds.dtype)

    clipped.rio.update_encoding({'dtype': clipped.dtype}, inplace=True)
    # !!!! encoding needs to be updated first!!!
    # otherwise nodata is not properly updated
    clipped.rio.update_encoding(rds.encoding, inplace=True)
    clipped.rio.write_nodata(rds.rio.nodata, inplace=True)

    return clipped


def init_logging(
        log_file=None, append=True,
        console_loglevel=logging.DEBUG,
        log_level=logging.INFO,
        logging_step_str=''):
    """Set up logging to file and console.

    If want write cusomt messages into logging file then use:
        logging.info('test')

    levels are:
        CRITICAL
        ERROR
        WARNING
        INFO
        DEBUG
        NOTSET
    """
    if append:
        filemode_val = 'a'
    else:
        filemode_val = 'w'

    from logging.handlers import RotatingFileHandler
    # Create a logger
    logger = logging.getLogger('')
    logger.setLevel(log_level)
    handler = RotatingFileHandler(
        log_file,  # file name
        maxBytes=5*1024*1024, # Limit file size to 5 MB (5 * 1024 * 1024 bytes)
        backupCount=2,  # Keep 2 old log files (app.log.1, app.log.2, etc.))
        mode=filemode_val  # with rollover another mode than append doesn't really make sense
        )

    # Set a log format
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(threadName)s %(name)s %(message)s")
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(console_loglevel)
    # set a format which is simpler for console use
    formatter = logging.Formatter("%(message)s")
    console.setFormatter(formatter)

    logger.addHandler(console)

    logging.info('\n\n ====  Proc start {} ====\n Start time: {}'.format(
        logging_step_str, dt.datetime.now().strftime('%Y-%m-%d_%H:%M')))

    return




def setup_time_control():
    """
    Set up time control for profiling.

    Returns:
        cProfile.Profile: A Profiler object that can be used to measure
        execution time.
    """
    prof = cProfile.Profile()
    prof.disable()  # disable profiling if don't want to record time...
    prof.enable()  # profiling back on
    return prof


def save_time_stats(prof, PATH_OUT, FILE_PREFIX):
    """
    Save time statistics for the given profiler.

    Args:
        prof (cProfile.Profile): The profiler object that was initialized at the beginning of the script.
        PATH_OUT (str): The output directory where the statistics will be saved.
        FILE_PREFIX (str): The prefix for the output files.

    Returns:
        None
    """
    # save time measure
    path_stats = os.path.normpath(
        os.path.join(PATH_OUT, f'{FILE_PREFIX}_time_stats.stats'))
    path_stats_txt = os.path.normpath(
        os.path.join(PATH_OUT, f'{FILE_PREFIX}_time_stats.txt'))

    prof.disable()  # don't profile the generation of stats
    prof.dump_stats(path_stats)

    with open(path_stats_txt, 'wt') as output:
        stats = pstats.Stats(path_stats, stream=output)
        stats.sort_stats('cumulative', 'time')
        stats.print_stats()

    return


def create_param_class_instance(module_name: str, class_name: str,
                    base_path, proc_step):
    '''
    dynamic class instance creation

    use e.g. as
    PARAM = create_instance(
        "param_coreg", "UAV_coreg_steps", PATH.PATH_BASE, 1)

    '''
    module = __import__(module_name)
    class_ = getattr(module, class_name)
    return class_(base_path, proc_step)


def check_for_missing_fill_val(img):
    if (img.rio.nodata is None
        and not np.any(np.isnan(img.data))):
        img.rio.write_nodata(0, inplace=True)

    elif img.rio.nodata is None:
        if np.any(np.isnan(img.data)):
            img.rio.write_nodata(np.nan, inplace=True)
        else:
            sys.exit(
                '!!! therem might be an error reading in data. \n'
                + 'nodata is not specified and nans are present in data')
    return img


def resave_tif_to_cog(file_path, resave_rio=True, chunks='auto'):
    '''
    use resave_rio if want to resave the raster after coregistration
    such that all attributes are in the file (by default arosics
    adds this info in a header file, however if only the tif is later
    copied to another locatio it can't be read with rioxarray due to
    missing header.

    !!! Note: with this method the nodata value is not written to file
    even if specified (thus maybe better use tif_to_cog_rasterio())
    '''

    if resave_rio:
        img = rioxarray.open_rasterio(
            file_path, masked=False, chunks=chunks)  # masked=True
        img = check_for_missing_fill_val(img)

        img.rio.to_raster(raster_path=file_path, driver="GTiff")

        img.close()

    tif_to_cog_rasterio(file_path)

    return


def tif_to_cog_rasterio(file_path, out_path=None):
    '''Save tif to cog via rasterio'''
    # Open the source GeoTIFF
    with rasterio.open(file_path, 'r') as src:
        # Define COG creation options
        cog_profile = src.profile.copy()

        blocksize = 256
        padded_data, pad_width, pad_height = pad_raster_to_multiple(
            src, blocksize)
        nodata_val = src.nodata
        cog_profile.update({
            'driver': 'GTiff',
            'BIGTIFF': "YES",
            'compress': 'DEFLATE',
            'blockxsize': blocksize,  # Tile size
            'blockysize': blocksize,  # Tile size
            'tiled': True,
            'width': src.width + pad_width,
            'height': src.height + pad_height,
            'nodata': nodata_val
            })

        if out_path is None:
            out_path = file_path.split('.')[0] + '_cog.tif'
        # Write the COG file
        with rasterio.open(out_path, 'w', **cog_profile) as dst:
            for i in range(1, src.count + 1):
                dst.write(padded_data[i-1], i)
            dst.descriptions = src.descriptions
            dst.update_tags(**src.tags().copy())
    return


def pad_raster_to_multiple(src, blocksize):
    width, height = src.width, src.height
    pad_width = (blocksize - width % blocksize) % blocksize
    pad_height = (blocksize - height % blocksize) % blocksize

    # Pad the raster data
    padded_data = []
    for i in range(1, src.count + 1):
        band = src.read(i)
        padded_band = np.pad(
            band,
            ((0, pad_height), (0, pad_width)),
            mode='constant',
            constant_values=src.nodata
        )
        padded_data.append(padded_band)

    return padded_data, pad_width, pad_height


@delayed
def filter_outliers(img, img_key, min_filt=None, max_filt=None,
                    moving_window_filter=False,
                    window_size=600, std_fact=5,
                    create_plot=False, fp_fig=None):

    # use median filter to create a smoothed image and pad nan values
    # at edges (use here mean as median is much slower)
    #rolled = img.rolling(dict(y=window_size, x=window_size), center=True
    #                       ).construct('window_y', 'window_x')
    #smoothed = rolled.mean(dim=('window_y', 'window_x'))
    img_cleaned = img.copy()
    if min_filt is not None:
        img_cleaned = img_cleaned.where(img > min_filt)

    if max_filt is not None:
        img_cleaned = img_cleaned.where(img < max_filt)

    if moving_window_filter:
        # for faster computation with dask use construct creates a new dimension
        smoothed = img_cleaned.rolling(
            y=window_size, x=window_size, center=True).mean()
        smoothed = smoothed.bfill(dim="x").bfill(dim="y").ffill(dim="x").ffill(dim="y")
        smoothed = smoothed.where(~np.isnan(img_cleaned))

        # get diff from orig'
        img_diff = np.abs(img_cleaned - smoothed)

        # get threshold for filtering
        threshold = img_diff.mean() + std_fact * img_diff.std()

        # remove values outside threshold
        img_cleaned = img_cleaned.where(img_diff <= threshold)

    # -- interpolating the gaps (!!! is very slow !!!)
    #img_cleaned_interp = fill_gaps(img_cleaned, max_gap=10)
    if create_plot:
        create_comparison_plot(
            [img, img_cleaned], ['unfiltered', 'filtered'],
            fp_fig=fp_fig,
            #cbar_title="Elevation change (m)"
            )

    return img_cleaned, img_key


def fill_gaps(img, max_gap=10):
    '''
    use 2D rioxarray (uses scipy groddata in background)
    '''
    # too large gaps ar set to nan
    if img.coords['y'].values[-1] < img.coords['y'].values[0]:
        img = img.sortby('y', ascending=True)
        img_interp1d = (
            img.interpolate_na(dim='x', method='linear', max_gap=max_gap)
            .interpolate_na(dim='y', method='linear', max_gap=max_gap)
            )
        img = img.sortby('y', ascending=False)
        img_interp1d = img_interp1d.sortby('y', ascending=False)
    else:
        img_interp1d = (
            img.interpolate_na(dim='x', method='linear', max_gap=max_gap)
            .interpolate_na(dim='y', method='linear', max_gap=max_gap)
            )

    # do 2D inteprolation with rixarray but this intepolates all
    img_interp2d = img.rio.interpolate_na(method='linear')

    img_interp2d.where(~np.isnan(img_interp1d))

    return img_interp2d


def create_comparison_plot(img_lst, title_lst, fp_fig=None,
                           band_plot='elev', fig_type='png'):

    n_subp = len(img_lst)
    fig, ax = plt.subplots(
        1, n_subp, sharex=True, sharey=True, figsize=(11.7, 8.3))

    for i in range(n_subp):

        img_lst[i].sel(band=band_plot).plot.imshow('x', 'y',
            cmap="RdYlBu", vmin=img_lst[i].min().values*1.02,
            vmax=img_lst[i].max().values*0.98,
            #cbar_title=cbar_title,
            ax=ax[i])
        ax[i].set_title(title_lst[i])

    if fp_fig is not None:
        fig.savefig(f"{fp_fig.split('.')[0]}.{fig_type}",
                    format=fig_type)

    return


def prepare_affine_transform(origin_x, origin_y, res_x, res_y):
    '''
    use this to prepare parameters for changing the grid origin
    can then be used with reproject in transform_raster()

    or can be used with reproject_match
    '''

    # Define the affine transformation matrix with the custom origin
    transform = affine.Affine(
        res_x,  # pixel width (X resolution)
        0,                      # rotation (0 if north-up)
        origin_x,                # top left corner X coordinate
        0,                      # rotation (0 if north-up)
        -res_y,  # pixel height (Y resolution, negative because Y decreases downwards)
        origin_y                 # top left corner Y coordinate
    )

    return transform


def get_new_origin(bounds_lbrt, res):
    '''
    origin is the upper left corner

    bounds: is array with min x, min y, max x, max y (lbrt)
        can come from img.rio.bounds() or poly.bounds (shapely)

    min x, min y, max x, max y.
    '''
    return get_closest_interv(bounds_lbrt[0], res), get_closest_interv(bounds_lbrt[3], res)


def get_closest_interv(x, res):
    return round(x / res) * res


def img_transform(img, res_out, epsg_out, resampling_type='bilinear',
                  new_origin=None):
    '''
    origin_x, origin_y = get_new_origin(bounds_lbrt, res_out)

    '''
    # ----- adjust resolution projection
    change_res = ((res_out is not None and res_out != 0)
                    and (img.rio.resolution()[0] != res_out
                    or img.rio.resolution()[1] != res_out))

    if new_origin is not None:
        affine_trans = prepare_affine_transform(
            *new_origin, res_out, res_out)
    else:
        affine_trans = None

    if img.rio.crs.to_epsg() != epsg_out or change_res:
        if change_res:
            res_target = res_out
        else:
            res_target = None
        img = transform_raster(
            img, epsg_out, nodata_out=img.rio.nodata,
            resampling_type=resampling_type, res=res_target,
            transform_inp=affine_trans)

    return img


def transform_raster(img, EPSG_TARGET, nodata_out=None,
                     resampling_type='bilinear', res=None,
                     transform_inp=None):
    '''
    default rsampling is bilinear
    other options are cubic or nearest (fastest)
    !!! for integer values, use nearest !!!

    to create transform_inp use: prepare_affine_transform() this
    can be used to change grid origin

    '''
    if nodata_out is None:
        nodata_out = img.rio.nodata
    if (img.rio.crs.to_epsg() != EPSG_TARGET or res is not None
        or transform_inp is not None):
        if nodata_out is None:
           nodata_out = img.rio.nodata

        img = img.rio.reproject(  # reproject
            "EPSG:" + str(EPSG_TARGET), nodata=nodata_out,
            resampling=Resampling[resampling_type], resolution=res)

        if transform_inp is not None:
            img = img.rio.reproject(  # reproject
                "EPSG:" + str(EPSG_TARGET), nodata=nodata_out,
                resampling=Resampling[resampling_type],
                transform=transform_inp)

        if nodata_out is not None:
            img.rio.write_nodata(nodata_out, inplace=True)

    return img


