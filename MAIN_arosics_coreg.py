
'''
Coregister a DEM to a reference DEM (using xdem)

inputs are raster files (.tif)

Steps:
run horizontal coregistration

nohup python ./MAIN_arosics_coreg.py &>./arosics_coreg.log &
'''

import sys
import os
import argparse
import numpy as np
import pandas as pd
import logging
import datetime as dt

import param_coreg
import utils_dem_coreg
import utils_arosics_coreg

# get path to current directory to load modules
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import control_file_templ


def main(PARAM_CLASS="UAV_coreg_steps_no_merge"):
    PATH = control_file_templ.get_proc_paths()

    # ------------ get param -----------
    PARAM = utils_dem_coreg.create_param_class_instance(
        "param_coreg", PARAM_CLASS, PATH.PATH_BASE, 2)

    # -------------- setup time control ----------------
    prof = utils_dem_coreg.setup_time_control()

    # === create log file and initialize errors to be written to console
    utils_dem_coreg.init_logging(
        log_file=os.path.join(PARAM.PATH_IO,
        f'{PARAM.PROJ_NAME}_arosics_coreg_log_errors.log'),
        log_level=logging.DEBUG)
    logging.info(
        '==== Proc start ====:'
        + dt.datetime.now().strftime('%Y-%m-%d_%H:%M'))

    # ====== read AOI (used for clipping)
    AOI_FILE = os.path.join(
        PARAM.PATH_IO, PARAM.PROCESSING_AOI)
    AOI_coords, AOI_poly = utils_dem_coreg.read_transl_geojson_AOI_coords_single(
        AOI_FILE, PARAM.EPSG_OUT)

    # ===== get names of preprocessed files
    df_merged = pd.read_csv(PARAM.preproc_files_df, sep='\t',
                            header=0).set_index('img_key')

    mask_inv_save_path = PARAM.mask_file

    # do pre-coregistration in horizontal direction based on RGB image
    # do global coregistration

    # uses first file in target list to get correction parameters
    # then appplies forrection to all files in target list

    logging.info('==== Start global coreg: '
                + dt.datetime.now().strftime('%Y-%m-%d_%H%M'))


    df_merged['f_path_global_coreg'] = df_merged.apply(
        lambda x: f'{x['f_path'].split('.')[0]}_coreg_global.tif'
        if 'REF' not in x['f_path'] else x['f_path'], axis=1)
    # local coregistration
    df_merged['f_path_local_coreg'] = df_merged.apply(
        lambda x: f'{x['f_path'].split('.')[0]}_coreg_local.tif'
        if 'REF' not in x['f_path'] else x['f_path'], axis=1)

    utils_arosics_coreg.do_global_coreg(
        df_merged.loc[PARAM.coreg_ref, 'f_path'],
        df_merged.loc[PARAM.coreg_target, 'f_path'].tolist(),
        df_merged.loc[PARAM.coreg_target, 'f_path_global_coreg'].tolist(),
        coreg_mask=None,#mask_inv_save_path,
        CPU_num=PARAM.COREG_CPU_NUM,
        max_shift=PARAM.COREG_GLOBAL_MAX_SHIFT,
        window_size=(PARAM.COREG_GLOBAL_WINDOW_SIZE,
                    PARAM.COREG_GLOBAL_WINDOW_SIZE),  # is given as tuple
        add_coreg_param_dict=PARAM.add_coreg_param_dict,
        nodata=(np.nan, np.nan),  # nodata for target and reference
        resampling=PARAM.RESAMPLING_TYPE)

    for i in df_merged.loc[PARAM.coreg_target, 'f_path_global_coreg'].tolist():
        utils_dem_coreg.resave_tif_to_cog(i, chunks=PARAM.dask_chunks)


    utils_arosics_coreg.apply_local_coreg(
        df_merged.loc[PARAM.coreg_ref, 'f_path'],
        df_merged.loc[PARAM.coreg_target, 'f_path_global_coreg'].tolist(),
        df_merged.loc[PARAM.coreg_target, 'f_path_local_coreg'].tolist(),
        coreg_mask=None,#mask_inv_save_path,
        nodata=(np.nan, np.nan),  # nodata for target and reference
        grid_res_pix=PARAM.GRID_RES_PIX_LOCAL,
        max_shift=PARAM.COREG_LOCAL_MAX_SHIFT,
        window_size=(PARAM.COREG_LOCAL_WINDOW_SIZE,
                    PARAM.COREG_LOCAL_WINDOW_SIZE),  # is given as tuple
        CPU_num=PARAM.COREG_CPU_NUM,
        add_coreg_param_dict=PARAM.add_coreg_param_dict,
        resampling=PARAM.RESAMPLING_TYPE
        )
    for i in df_merged.loc[PARAM.coreg_target, 'f_path_local_coreg'].tolist():
        utils_dem_coreg.resave_tif_to_cog(i)

    df_merged.to_csv(
        os.path.join(PARAM.PATH_IO, f'{PARAM.PROJ_NAME}_arosics_coreg_files.txt'),
        sep='\t', header=True)

    utils_dem_coreg.save_time_stats(
        prof, PARAM.PATH_IO, f'{PARAM.PROJ_NAME}_arosics_coreg')

    print('----- FINISH ----')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Do horizontal co-registration based on mosaic')
    parser.add_argument(
        '--PARAM_CLASS', type=str,
        help=('name of Parameter class'), default="UAV_coreg_steps_no_merge")
    args = parser.parse_args()

    main(**vars(args))