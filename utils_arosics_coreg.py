import numpy as np
import pandas as pd
from arosics import COREG_LOCAL, COREG, DESHIFTER
import rioxarray
import xarray
import json

def do_global_coreg(
        im_ref, im_target_lst, im_outname_lst,
        coreg_mask=None,
        apply_correction=True,
        ref_band=1, tar_band=1, nodata=None,
        CPU_num=30, max_shift=50, window_size=None,
        add_coreg_param_dict=None, resampling='cubic',
        **kwargs):
    '''
    uses first file in target list to receive correction
    then appplies forrection to all files in target list
    '''
    im_target = im_target_lst[0]
    path_out = im_outname_lst[0]

    if nodata is None:
        nodata = (0, 0)

    #if coreg_mask is not None:
    #    # creates new clipped tif file to use.....
    #    im_target = create_coreg_mask(
    #        im_target, coreg_mask, EPSG_TARGET)

    kw_dict = {
        'path_out': path_out, 'fmt_out': 'GTIFF',
        'r_b4match': ref_band, 's_b4match': tar_band,
        'mask_baddata_ref': coreg_mask,
        #'mask_baddata_tgt': tgt_mask,
        'CPUs': CPU_num, 'nodata': nodata,
        'max_shift': max_shift,
        'ws': window_size}

    if add_coreg_param_dict is not None:
        kw_dict.update(add_coreg_param_dict)

    kw_dict.update(kwargs)

    # initialize
    #try:
    CR = COREG(im_ref, im_target, **kw_dict)
    CR.calculate_spatial_shifts()


    if apply_correction:
        for i, j in zip(im_target_lst, im_outname_lst):
            apply_coreg(CR, i, j, resampling)
    out = 'global coreg success'
    #except:
    #    print('!!! correg error, failed global coreg!!')
    #    out = 'failed global coreg'
    save_dict_to_text(CR.__dict__, path_out, 'global_coreg_val')
    # SSIM oig and deshifted are e.g.
    #Image similarity within the matching window (SSIM before/after correction): 0.0655 => 0.5220

    return out


def apply_local_coreg(
        im_ref, im_target_lst, im_outname_lst,
        coreg_mask=None, displ=False, apply_correction=True,
        ref_band=1, tar_band=1, nodata=None,
        grid_res_pix=200,
        max_shift=5, window_size=None, CPU_num=30,
        add_coreg_param_dict=None, resampling='cubic',
        get_coreg_after_corr_stats=False,
        **kwargs):

    '''
    uses first file in target list to reive correction
    then appplies forrection to all files in target list

    if from_disk then im_ref must be a path

    match_gsd (bool): True: match the input pixel size to the
        reference pixel size, default = False
    out_gsd (Optional[float, None]): output pixel size in units
        of the reference coordinate system
        (default = pixel size of the input array), given values
        are overridden by match_gsd=True

    r_b4match (int): band of reference image to be used for matching (starts with 1; default: 1)
    s_b4match (int): band of shift image to be used for matching (starts with 1; default: 1)

    mask_baddata_ref (Union[GeoArray, str, None]):
        path to a 2D boolean mask file (or an instance of BadDataMask)
        for the reference image where all bad data pixels (e.g. clouds)
        are marked with True and the remaining pixels with False.
        Must have the same geographic extent and projection like im_ref.
        The mask is used to check if the chosen matching window
        position is valid in the sense of useful data. Otherwise
        this window position is rejected.

    mask_baddata_tgt (Union[GeoArray, str, None]) path to a 2D boolean
        mask file (or an instance of BadDataMask) for the image to
        be shifted where all bad data pixels (e.g. clouds) are marked
        with True and the remaining pixels with False. Must have the
        same geographic extent and projection like im_ref.
        The mask is used to check if the chosen matching window
        position is valid in the sense of useful data. Otherwise
        this window position is rejected.

    footprint_poly_ref (Optional[str, None]):
        footprint polygon of the reference image
        (WKT string or shapely.geometry.Polygon)

    footprint_poly_tgt (Optional[str, None]):
        footprint polygon of the image to be shifted
        (WKT string or shapely.geometry.Polygon)

    data_corners_ref (Optional[list, None]):
        map coordinates of data corners within reference image.
        ignored if footprint_poly_ref is given.

    data_corners_tgt (Optional[list, None]):
        map coordinates of data corners within image to be shifted.
        ignored if footprint_poly_tgt is given.

    outFillVal (int): if given the generated tie point grid is
        filled with this value in case no match could be found
        during co-registration (default: -9999)

    '''
    if window_size is None:
        window_size = (256, 256)
    if nodata is None:
        nodata = (0, 0)

    im_target = im_target_lst[0]
    path_out = im_outname_lst[0]

    #if coreg_mask is not None:
    #    im_target = create_coreg_mask(im_target, coreg_mask, EPSG_TARGET)

    kw_dict = {
        'grid_res': grid_res_pix,
        'path_out': path_out, 'fmt_out': 'GTIFF',
        'r_b4match': ref_band, 's_b4match': tar_band,
        'max_shift': max_shift,
        'window_size': window_size,
        #'footprint_poly_ref': coreg_mask,
        #'footprint_poly_tgt': tgt_mask,
        'CPUs': CPU_num,
        'nodata': nodata,
        'mask_baddata_ref': coreg_mask}

    if add_coreg_param_dict is not None:
        kw_dict.update(add_coreg_param_dict)
    kw_dict.update(kwargs)

    # initialize
    CRL = COREG_LOCAL(
        im_ref, im_target, **kw_dict)

    #try:
    if apply_correction:
        for i, j in zip(im_target_lst, im_outname_lst):
            apply_coreg(CRL, i, j, resampling)
    out = 'local coreg sucess'
    #except:
    #    print('!!! local correg error !!')
    #    out = 'failed local coreg'


    if get_coreg_after_corr_stats:
        CRL_after_corr = COREG_LOCAL(im_ref, CRL.path_out, **kw_dict)

    # display coregistration points
    if displ:
        kw_dict.update({'path_out': 'auto'})

        CRL.view_CoRegPoints(figsize=(15,15), backgroundIm='ref')

        if get_coreg_after_corr_stats:
            # display shifts after coreg
            #CRL_after_corr = COREG_LOCAL(im_ref, CRL.path_out, **kw_dict)
            CRL_after_corr.view_CoRegPoints(
                figsize=(15,15), backgroundIm='ref')

    # save overall stats
    save_tie_point_stats(CRL, path_out, suffix_add='')
    # save tie point grid as shape file and as .txt table
    save_tie_point_grid(
        CRL, path_out, suffix_add='')

    if get_coreg_after_corr_stats:
        # save overall stats
        save_tie_point_stats(
            CRL_after_corr, path_out, suffix_add='_after_coreg')
        # save tie point grid as shape file and as .txt table
        save_tie_point_grid(
            CRL_after_corr, path_out, suffix_add='_after_coreg')

    return out


def save_tie_point_stats(CRL, path_out, suffix_add=''):
    '''
    CRL is local coregistration instance
    path_out is the output path of the target file
    suffix_add: for after coreg stats add suffix e.g. '_after_coreg'

    '''
    # save overall stats
    path_base = remove_suffix(path_out, '.')
    path_txt = f'{path_base}_tie_points_stats{suffix_add}.txt'

    # get stats
    pts_stats = {}
    pts_stats['incl_outliers'] = CRL.tiepoint_grid.calc_overall_stats(
        include_outliers=True)
    pts_stats['without_outliers'] = CRL.tiepoint_grid.calc_overall_stats(
        include_outliers=False)

    # save stats
    pd.DataFrame.from_dict(pts_stats).to_csv(
        path_or_buf=path_txt, sep='\t', header=True)

    return


def save_tie_point_grid(CRL, path_out, suffix_add=''):

    path_base = remove_suffix(path_out, '.')
    # for shapefile
    path_shp = f'{path_base}_tie_points_grid{suffix_add}.shp'
    # for txt
    path_txt = f'{path_base}_tie_points_grid{suffix_add}.txt'

    # for param description check
    # https://danschef.git-pages.gfz-potsdam.de/arosics/doc/arosics.html#arosics.Tie_Point_Grid.Tie_Point_Grid
    try:
        CRL.tiepoint_grid.to_PointShapefile(path_out=path_shp)
    except:
        pass

    CRL.tiepoint_grid.CoRegPoints_table.to_csv(
        path_or_buf=path_txt, sep='\t', header=True)

    return


def check_shift_after_coreg(
        path_ref, path_target,
        coreg_mask=None, displ=False,
        ref_band=1, tar_band=1, nodata=None, grid_res_pix=200,
        max_shift=5, window_size=None, CPU_num=30,
        add_coreg_param_dict=None, file_suffix='',
        **kwargs):

    # use same kwargs as used for coreg
    if window_size is None:
        window_size = (256, 256)
    if nodata is None:
        nodata = (0, 0)

    kw_dict = {
        'grid_res': grid_res_pix,
        'r_b4match': ref_band, 's_b4match': tar_band,
        'max_shift': max_shift,
        'window_size': window_size,
        'CPUs': CPU_num,
        'nodata': nodata,
        'mask_baddata_ref': coreg_mask}
    if add_coreg_param_dict is not None:
        kw_dict.update(add_coreg_param_dict)
    kw_dict.update(kwargs)

    CRL_after_corr = COREG_LOCAL(path_ref, path_target, **kw_dict)
    if displ:
        CRL_after_corr.view_CoRegPoints(
            figsize=(15,15),backgroundIm='ref')

    # save overall stats
    save_tie_point_stats(
        CRL_after_corr, path_target,
        suffix_add=f'{file_suffix}_after_coreg')
    # save tie point grid as shape file and as .txt table
    save_tie_point_grid(
        CRL_after_corr, path_target,
        suffix_add=f'{file_suffix}_after_coreg')

    # !!! note on table output
    #df_valid = CRL_after_corr.tiepoint_grid.CoRegPoints_table.query('OUTLIER!=-9999')
    # ABS_SHIFT is the absolute shift in meter and correspponds to
    # np.sqrt(df_valid['X_SHIFT_M']**2 + df_valid['Y_SHIFT_M']**2)
    # the following corresponds in the stats to the MEAN_ABS_SHIFT incl outliers
    #df_valid['ABS_SHIFT'].mean()
    # and the following to the MEAN_ABS_SHIFT without outliers
    #df_valid.query('OUTLIER == False')['ABS_SHIFT'].mean()
    # get amount of valid tie points with:
    # df_valid.query('OUTLIER == 0').shape
    return


def apply_coreg(CR, im_target, path_out, resampling, align_grids=True):


    # !!!!!!!!! check to add
    # !!! match_gsd (bool) – True: match the input pixel size to the reference pixel size, default = False
    DESHIFTER(im_target, CR.coreg_info, path_out=path_out,
              align_grids=align_grids, resamp_alg=resampling).correct_shifts()
    # !!!!!! do not choose cubic here gives error
    # !!! there is an error with align_grids=True

    #self.meta['proc'] = (self.meta['proc'] + ['coreg_' + coreg_type])
    #self.proc_count += 1

    return



def remove_suffix(inp_str, suffix):
    return inp_str[:-(inp_str[::-1].find(suffix) + 1)]



def save_dict_to_text(dict_data, path_out, file_suffix, keys_dict=None):
    '''
    Converts dictionaries to text
    (e.g. to save Parameters to file)

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

    f_path = f"{path_out.split('.')[0]}_{file_suffix}.txt"
    with open(f_path, 'w') as f:
        f.write(text)
    f_path = f"{path_out.split('.')[0]}_{file_suffix}_coreg_info.json"
    with open(f_path, 'w') as f:
        json.dump(dict_data['_coreg_info'], f)

    return text