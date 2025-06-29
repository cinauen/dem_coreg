

import os
import json
import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import datetime as dt
import rioxarray
from rasterio.enums import Resampling
from scipy.interpolate import griddata as sci_griddata
import xdem

import utils_preproc


def save_metadata_xdem_coreg_json(meta_data, PATH_OUT, filename_prefix):
    ''' Does not work with dimedelta data!!!'''
    f_path = os.path.join(PATH_OUT, f"{filename_prefix}.json")

    with open(f_path, 'w') as f:
        json.dump(meta_data, f)
    return


def dict_to_text(dict_data, keys_dict=None):
    '''
    Converts dictionaries to text
    (e.g. to save Parameters or metadata to file)

    keys_dict: list of specific keys which should be saved
    '''
    if not keys_dict:
        keys_dict = list(dict_data.keys())

    text = []
    for key, val in dict_data.items():
        if isinstance(val, dict):
            text.append('\n-- dict: ' + key + '\n')
            for key_s, val_s in val.items():
                text.append('%s: %s \n' % (key_s, val_s))
            text.append('--\n')
        else:
            text.append('%s: %s \n' % (key, val))
        if isinstance(val, pd.DataFrame):
            text.append('\n')
    text = ''.join(text)

    return text


def save_to_txt(metadata_save, PATH_OUT, filename_prefix,
                how='a', close=True):

    f_path = os.path.join(PATH_OUT, f"{filename_prefix}.txt")

    meta_text = open(f_path, how)
    meta_text.write(
        '# ==========================================================\n'
        + '#  metadata save: ' + dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        + '\n# ==========================================================\n')
    meta_text.write(metadata_save)

    meta_text.flush()
    if close:
        meta_text.close()
        return

    return meta_text


def get_output_info_as_str(xdem_coreg_class):
    '''
    xdem_coreg_class is e.g. nuth_kaab = xdem.coreg.NuthKaab()
        after fit and apply (nuth_kaab.fit_and_apply(...))
    '''
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        xdem_coreg_class.info()
    return buffer.getvalue()



def apply_xy_shift_to_raster_xdem(file_p_target, file_p_ref, transf_matrix,
                             coreg_name, vertical_crs, resampling_type='bilinear'):
    '''
    Parameters:
    file_p: str
        path to file onto which shift should be applied
        (e.g. target_fname_RGB)
    file_p_ref: str
        reference image to use for reproject_match to ensure that all
        coordinates are the same (use: )
        (e.g. coreg_out_fname[0])
    xy_shifts: dict
        e.g. nuth_kaab._meta['outputs']['affine']
        has {'shift_x': XXX, 'shift_y': XX} are stored in list
        shift_applied_lst
    '''
    # re-read file into xdem format
    dem_aligned = xdem.DEM(file_p_ref)
    #rioxarray.open_rasterio(
    #    file_p_ref, masked=False, chunks='auto')

    # ------ apply shift to RGB ------
    # read RGB onto which should be applied
    img_to_shift = rioxarray.open_rasterio(
        file_p_target, masked=False, chunks=None)
    #img_to_shift['band'] = ('band', list(img_to_shift.attrs['long_name']))
    #b_red = img_to_shift.sel(band='red')
    #b_red.attrs = {}
    #b_red.rio.write_nodata(img_to_shift.rio.nodata, inplace=True)
    #b_red.rio.write_crs(img_to_shift.rio.crs)
    raster_lst = [xdem.DEM.from_xarray(img_to_shift.sel(band=x))
                  for x in img_to_shift.band.values]
    [x.set_vcrs(vertical_crs) for x in raster_lst]
    #src = rasterio.open(file_p_target)
    #band1 = src.read(1)
    #src.close()

    # ---- add vertical refernce ssystem -----
    #dem_aligned.set_vcrs(vertical_crs)
    #img_to_shift.set_vcrs(vertical_crs)

    #-- get and apply horizontal shift
    # for pipeline might have several shifts that must be applied
    # --> therefore loop through list with shifts
    m_inp_lst = []
    for i_matrix in transf_matrix:
        # transfrom xy matrix
        m_inp = i_matrix.copy()
        m_inp[2, :] = [0, 0, 1, 0]  # remove verticalcomponent
        m_inp_lst.append(m_inp)

    # combine all transformation matices into one using matrix multiplication
    # (!!! order matters, matrix multiplication has to be done in reverse order)
    for e_m, i_m in enumerate(m_inp_lst[::-1]):
        if e_m == 0:
            m_combined = i_m
        else:
            m_combined = m_combined @ i_m

    shifted_lst = []
    for i_raster in raster_lst:
        shifted_band = xdem.coreg.apply_matrix(i_raster, m_combined,
                                          resampling='linear')
        shifted_lst.append(shifted_band.to_xarray())


    #img_to_shift.coords["x"], img_to_shift.coords["y"] = extract_2d_transformation_apply(
    #    i_matrix,
    #    img_to_shift.coords["x"].values,
    #    img_to_shift.coords["y"].values)s

    # -- reproject to be aligned with DEM
    #img_to_shift = img_to_shift.rio.reproject_match(
    #    dem_aligned, Resampling=Resampling[resampling_type])

    # ---- save shifted image
    path_RGB_out = os.path.join(
        f"{file_p_target.split('.')[0]}_coreg_{coreg_name}_RGB.tif")
    utils_preproc.img_save(img_shifted, path_RGB_out)

    return


def apply_xy_shift_to_raster(file_p_target, file_p_ref, transf_matrix,
                             coreg_name, resampling_type='cubic'):
    '''
    Parameters:
    file_p: str
        path to file onto which shift should be applied
        (e.g. target_fname_RGB)
    file_p_ref: str
        reference image to use for reproject_match to ensure that all
        coordinates are the same (use: )
        (e.g. coreg_out_fname[0])
    transf_matrix: list with numpy array
        e.g. nuth_kaab._meta['outputs']['affine']
        has {'shift_x': XXX, 'shift_y': XX} are stored in list
        shift_applied_lst
    '''
    # re-read file in xarray
    dem_aligned = rioxarray.open_rasterio(
        file_p_ref, masked=False, chunks='auto')

    # ------ apply shift to RGB ------
    # read RGB onto which should be applied
    img_to_shift = rioxarray.open_rasterio(
        file_p_target, masked=False, chunks='auto')

    #-- get and apply horizontal shift
    # for pipeline might have several shifts that must be applied
    # --> therefore loop through list with shifts
    m_inp_lst = []
    for i_matrix in transf_matrix:
        # transfrom xy matrix
        m_inp = i_matrix.copy()
        m_inp[2, :] = [0, 0, 1, 0]  # remove verticalcomponent
        m_inp_lst.append(m_inp)

    # combine all transformation matices into one using matrix multiplication
    # (!!! order matters, matrix multiplication has to be done in reverse order)
    for e_m, i_m in enumerate(m_inp_lst[::-1]):
        if e_m == 0:
            m_combined = i_m
        else:
            m_combined = m_combined @ i_m

    # apply transformation to xy coordinates on xy plane
    xv, yv = np.meshgrid(img_to_shift.coords['x'].values,
                         img_to_shift.coords['y'].values, indexing='xy')

    # get transformed coordinates
    x_transf, y_transf = extract_2d_transformation_apply(
        m_combined, xv.ravel(), yv.ravel())

    # interpolate transformed irregular points onto regular grid
    data_grid = {}
    for i_band in img_to_shift.band:
        pts = np.stack([x_transf, y_transf,
                        img_to_shift.sel(band=i_band).values.ravel()]).T
        inp_pts = pts[~np.isnan(pts[:, 2]), :]
        data_grid[i_band] = sci_griddata(
            inp_pts[:, :2], inp_pts[:, 2].ravel(),
            (xv.ravel(), yv.ravel()), method=resampling_type)
        print('t')

    # img.loc[{'band': input}] = img_no_val
    # xv, yv = np.meshgrid(img_to_shift.coords['x'].values,
    #                      img_to_shift.coords['y'].values)
    # tt = np.stack([xv.ravel(), yv.ravel()])

    # -- reproject to be aligned with DEM
    img_to_shift = img_to_shift.rio.reproject_match(
        dem_aligned, Resampling=Resampling[resampling_type])

    # ---- save shifted image
    path_RGB_out = os.path.join(
        f"{file_p_target.split('.')[0]}_coreg_{coreg_name}_RGB.tif")
    utils_preproc.img_save(img_to_shift, path_RGB_out)

    return


def extract_2d_transformation_apply(transf_matrix, x_coords, y_coords):
    '''
    extracts 2dtransformation (in xy plane) from 3dtransfromation matrix and
    applies it to the x and y coordinates
    (to apply the xdem transformation e.g. to RGB images)

    x_coords are a 1d numpy arrray extracted with DataArray.coords['x']
    tansf_matrix numpy array 4 x 4
        transformation matrix from e.g. i_coreg_class.to_matrix()
    '''
    # get translation
    #xdem.coreg.AffineCoreg.to_translations
    # get roration
    # xdem.coreg.AffineCoreg.to_rotations


    # extract 2d component
    transf_m2d = transf_matrix[(0, 1, 3), ...][..., (0, 1, 3)].copy()

    # convert coords matrix required for matrix multiplication
    # ones are for translation
    xyt_coords = np.stack([x_coords, y_coords, np.ones(y_coords.shape)])


    xy_transl = transf_m2d @ xyt_coords
    # xy_transl[2, 0] should all be ones

    return xy_transl[0, :], xy_transl[1, :]


def apply_transformation(img_aligned, img_to_transform, t_matrix):

    # apply transformation to xy coordinates on xy plane
    xv, yv = np.meshgrid(img_to_transform.coords['x'].values,
                         img_to_transform.coords['y'].values, indexing='xy')

    # get transformed coordinates
    x_transf, y_transf = extract_2d_transformation_apply(
        t_matrix, xv.ravel(), yv.ravel())

    # interpolate transformed irregular points onto regular grid
    data_grid = {}
    for i_band in img_to_transform.band:
        pts = np.stack([x_transf, y_transf,
                        img_to_transform.sel(band=i_band).values.ravel()]).T
        inp_pts = pts[~np.isnan(pts[:, 2]), :]
        data_grid[i_band] = sci_griddata(
            inp_pts[:, :2], inp_pts[:, 2].ravel(),
            (xv.ravel(), yv.ravel()), method=resampling_type)
        print('t')

    return


def save_xdem_img(img_dem, fpath_out):

    img_dem.save(fpath_out, nodata=np.nan)
    return