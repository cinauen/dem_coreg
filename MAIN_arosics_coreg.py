
'''
Coregister UAV data in horiyontal plane using agisoft using just RGB or
combind with DEM

inputs are:
 - preprocessed raster files (.tif), reference and target (need to have same
   coordinate system and extent)
 - nodata mask (if defined to be used)
parameters:
 - all required parameters must be defined as a class in param_coreg.py

Steps:
run horizontal coregistration using arosics

------
usage: MAIN_arosics_coreg.py [-h] [--PARAM_CLASS PARAM_CLASS]

Do horizontal co-registration based on mosaic
options:
  -h, --help            show this help message and exit
  --PARAM_CLASS PARAM_CLASS
                        name of class with parameters
                        (default: UAV_coreg_steps_no_merge)

or run in background with
nohup python ./MAIN_arosics_coreg.py --PARAM_CLASS UAV_coreg_steps_no_merge &>./arosics_coreg.log &
'''

import sys
import os
import argparse
import numpy as np
import pandas as pd
import logging
import datetime as dt

# get path to current directory to load modules
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import param_coreg
import utils_preproc
import utils_arosics_coreg

import control_file


def main(PARAM_CLASS="UAV_coreg_steps_no_merge"):
    # %% ======= setup PATHS and parameters
    PATH = control_file.get_proc_paths()

    # ------------ get param -----------
    PARAM = utils_preproc.create_param_class_instance(
        "param_coreg", PARAM_CLASS, PATH.PATH_BASE, 2)

    # -------------- setup time control ----------------
    prof = utils_preproc.setup_time_control()

    # --- create log file and initialize errors to be written to console
    utils_preproc.init_logging(
        log_file=os.path.join(PARAM.PATH_IO,
        f'{PARAM.PROJ_NAME}_arosics_coreg_log_errors.log'),
        log_level=logging.DEBUG)
    logging.info(f'=== Proc start ===: {utils_preproc.get_timenow_str()}')

    # ----- get names of preprocessed files and mask
    df_merged = pd.read_csv(PARAM.preproc_files_df, sep='\t',
                            header=0).set_index('img_key')

    if PARAM.use_mask_file:
        mask_inv_save_path = PARAM.mask_file
    else:
        mask_inv_save_path = None

    # ---- define output paths for coregistration
    df_merged['f_path_global_coreg'] = df_merged.apply(
        lambda x: f'{x['f_path'].split('.')[0]}_coreg_global.tif'
        if 'REF' not in x['f_path'] else x['f_path'], axis=1)
    # local coregistration
    df_merged['f_path_local_coreg'] = df_merged.apply(
        lambda x: f'{x['f_path'].split('.')[0]}_coreg_local.tif'
        if 'REF' not in x['f_path'] else x['f_path'], axis=1)


    # %% ====== global coregistration
    # First file in target list is used to get correction parameters.
    # The corrections are then then applied to all remaning files in
    # target list
    logging.info(f'==== Start global coreg: {utils_preproc.get_timenow_str()}')
    utils_arosics_coreg.do_global_coreg(
        df_merged.loc[PARAM.coreg_ref, 'f_path'],
        df_merged.loc[PARAM.coreg_target, 'f_path'].tolist(),
        df_merged.loc[PARAM.coreg_target, 'f_path_global_coreg'].tolist(),
        coreg_mask=mask_inv_save_path,
        CPU_num=PARAM.COREG_CPU_NUM,
        max_shift=PARAM.COREG_GLOBAL_MAX_SHIFT,
        window_size=(PARAM.COREG_GLOBAL_WINDOW_SIZE,
                    PARAM.COREG_GLOBAL_WINDOW_SIZE),  # is given as tuple
        add_coreg_param_dict=PARAM.add_coreg_param_dict,
        nodata=(np.nan, np.nan),  # nodata for target and reference
        resampling=PARAM.RESAMPLING_TYPE)

    # save as .cog for faster display and r-save as .tif (since direct
    # output of arosics required header file)
    logging.info(f'==== Save global coreg: {utils_preproc.get_timenow_str()}')
    for i in df_merged.loc[PARAM.coreg_target, 'f_path_global_coreg'].tolist():
        utils_preproc.resave_tif_to_cog(i, chunks=PARAM.dask_chunks)

    # %% ====== local coregistration
    # First file in target list is used to get correction parameters.
    # The corrections are then then applied to all remaning files in
    # target list
    logging.info(f'==== Start local coreg: {utils_preproc.get_timenow_str()}')
    utils_arosics_coreg.apply_local_coreg(
        df_merged.loc[PARAM.coreg_ref, 'f_path'],
        df_merged.loc[PARAM.coreg_target, 'f_path_global_coreg'].tolist(),
        df_merged.loc[PARAM.coreg_target, 'f_path_local_coreg'].tolist(),
        coreg_mask=mask_inv_save_path,
        nodata=(np.nan, np.nan),  # nodata for target and reference
        grid_res_pix=PARAM.GRID_RES_PIX_LOCAL,
        max_shift=PARAM.COREG_LOCAL_MAX_SHIFT,
        window_size=(PARAM.COREG_LOCAL_WINDOW_SIZE,
                    PARAM.COREG_LOCAL_WINDOW_SIZE),  # is given as tuple
        CPU_num=PARAM.COREG_CPU_NUM,
        add_coreg_param_dict=PARAM.add_coreg_param_dict,
        resampling=PARAM.RESAMPLING_TYPE
        )
    logging.info(f'==== Save local coreg: {utils_preproc.get_timenow_str()}')
    for i in df_merged.loc[PARAM.coreg_target, 'f_path_local_coreg'].tolist():
        utils_preproc.resave_tif_to_cog(i)

    # ====== save ouput file information and time statistics
    df_merged.to_csv(
        os.path.join(PARAM.PATH_IO, f'{PARAM.PROJ_NAME}_arosics_coreg_files.txt'),
        sep='\t', header=True)
    utils_preproc.save_time_stats(
        prof, PARAM.PATH_IO, f'{PARAM.PROJ_NAME}_arosics_coreg')

    logging.info(f'==== finished ====: {utils_preproc.get_timenow_str()}')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Do horizontal co-registration based on mosaic')
    parser.add_argument(
        '--PARAM_CLASS', type=str,
        help=('name of class with parameters (default: UAV_coreg_steps_no_merge)'),
        default="UAV_coreg_steps_no_merge")
    args = parser.parse_args()

    main(**vars(args))

