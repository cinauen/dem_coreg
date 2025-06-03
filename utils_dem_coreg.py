

import os
import sys
import logging

import datetime as dt
import numpy as np
import rioxarray
import shapely.geometry as shap_geom
from rasterio.enums import Resampling
import osgeo
from osgeo import osr
from osgeo import ogr

import geojson


def img_read(FILE_P, FILE_N, NODATA, EPSG_INP):
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

    img = rioxarray.open_rasterio(
        os.path.join(FILE_P, FILE_N), masked=False)  # , chunks='auto')

    if NODATA is not None:
        # set nodata value to if specified
        img.rio.set_nodata(NODATA, inplace=True)
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
    out.rio.set_nodata(fill_val_new, inplace=True)

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
    clipped.rio.set_nodata(rds.rio.nodata, inplace=True)

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

