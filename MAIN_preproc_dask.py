
'''
Peprocessing to later coregister UAV data (DEM and RGB)

inputs are:
 - raster files (.tif), RGB and DEM
 - AOI with area of interest
parameters:
 - all required parameters must be defined as a class in param_coreg.py

Steps: run preprocessing with dask
1) read files (reference: file to which the dem should be coregistered to,
               target: the file which has to be coregistered)
2) preprocess files: clip to same extent and reproject to same
               resolution (min resolution) and merge subfiles if required
3) save pre-processed files

------
usage: MAIN_preproc_dask.py [-h] [--PARAM_CLASS PARAM_CLASS]

Preprocess imagery for co-registration

options:
  -h, --help    show this help message and exit
  --PARAM_CLASS PARAM_CLASS
                name of class with parameters
                (default: UAV_coreg_steps_no_merge)

or run in background
nohup python ./MAIN_preproc_dask.py --PARAM_CLASS UAV_coreg_steps_no_merge &>./Preproc_dask.log &
'''

import sys
import os
import argparse
import logging
import numpy as np
import pandas as pd
import xarray
from joblib import Parallel, delayed
import gc
from dask import compute, persist
from dask.distributed import Client, LocalCluster

import param_coreg
import utils_preproc

# get path to current directory to load modules
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import control_file


def main(PARAM_CLASS="UAV_coreg_steps_no_merge"):
    # %% ======= setup PATHS and parameters
    PATH = control_file.get_proc_paths()

    # ------------ get param -----------
    PARAM = utils_preproc.create_param_class_instance(
        "param_coreg", PARAM_CLASS, PATH.PATH_BASE, proc_step=1)

    # -------------- setup time control ----------------
    prof = utils_preproc.setup_time_control()

    # --- create log file and initialize errors to be written to console
    utils_preproc.init_logging(
        log_file=os.path.join(PARAM.PATH_IO,
        f'{PARAM.PROJ_NAME}_preproc_log_errors.log'))
    logging.info(f'==== Proc start ====: {utils_preproc.get_timenow_str()}')


    # %% ====== read AOI (used for clipping)
    AOI_FILE = os.path.join(
        PARAM.PATH_IO, PARAM.PROCESSING_AOI)
    AOI_coords, AOI_poly = utils_preproc.read_transl_geojson_AOI_coords_single(
        AOI_FILE, PARAM.EPSG_OUT)
    # get new origin for image transform (this makes sure that the
    # coord systems of all all images fit exactly, thus no need for
    # reproject match)
    new_orig = utils_preproc.get_new_origin(
        AOI_poly.bounds, PARAM.RESOLUTION_OUT)

    # %% ============= 1) read and preprocess files
    # preprocess raster files
    # clipping all to same extent and transform to same resolution
    # (for xdem they must have the same projection extent, bounds and
    #  resolutiom (thus exact same cooridnates))
    file_lst, file_n_lst, img_key, img_type, nodata_inp = utils_preproc.get_inp_lst_parallel_proc(
        PARAM.REF_PATH, PARAM.REF_FNAME, PARAM.REF_NODATA_INP,
        PARAM.TARGET_PATH, PARAM.TARGET_FNAME, PARAM.TARGET_NODATA_INP,
        )
    # create DataFrame with file names
    df_meta = pd.DataFrame(
        np.stack([img_type, img_key, file_lst, file_n_lst]).T,
        columns=['pID', 'img_key', 'file_p', 'file_n']).groupby('pID').aggregate(list)

    # preprocess files in parallel using dask. (Similalry could use
    # joblib see \additional_versions\MAIN_preproc.py)
    logging.info(f'==== read preproc ====: {utils_preproc.get_timenow_str()}')
    task_w1 = [utils_preproc.img_read_preproc(
        *i, PARAM.RESOLUTION_OUT, AOI_coords,
        PARAM.EPSG_INP, PARAM.EPSG_OUT, band_names=PARAM.band_names,
        bands_keep=PARAM.bands_keep, chunks=PARAM.dask_chunks,
        new_origin=new_orig)
            for i in zip(file_lst, img_key, nodata_inp)]

    w1_futures = persist(*task_w1)
    w1_futures = compute(*w1_futures)  # compute is required due creating
    # dict in to next line  (therfore compute could actually also be run
    # directly instead of task persist --> w1 = compute(*[utils_sem...])
    img_lst, img_key = zip(*w1_futures)

    # free up memory
    del w1_futures, task_w1
    gc.collect()

    # ----- do outlier filtering (here for DEM)
    # easiest is just to use min and max for upper and lower bounds
    # !!! window based filtering is very slow and needs to be tested with
    # different window sizes and std threshold options.
    logging.info(f'=== filter outliers ===: {utils_preproc.get_timenow_str()}')
    img_dict = {x: y for x, y in zip(img_key, img_lst)}

    # get list of files that should be filtered (DSM)
    key_lst = [x for x in img_key if PARAM.filter_img_key in x]
    task_w11 = [utils_preproc.filter_outliers(
        img_dict[i_key], i_key,
        min_filt=PARAM.filter_min, max_filt=PARAM.filter_max,
        window_size=PARAM.filter_window_size_px,
        std_fact=PARAM.filter_std_fact,
        create_plot=True,
        fp_fig=os.path.join(
            PARAM.PATH_IO,
            f'{i_key}_outlier_filter.pdf')) for i_key in key_lst]

    w11_futures = persist(*task_w11)
    w11_futures = compute(*w11_futures)
    img_lst11, img_key11 = zip(*w11_futures)
    img_dict.update({x: y for x, y in zip(img_key11, img_lst11)})

    # free up memory
    del w11_futures, img_lst11, img_key11, task_w11
    gc.collect()

    # %% ====== merge files if there are several subfiles
    logging.info(f'==== merge ====: {utils_preproc.get_timenow_str()}')
    img_type_index = df_meta.index.tolist()
    img = {x: [img_dict[xx] for xx in df_meta.loc[x, 'img_key']]
           for x in img_type_index}
    w3_futures = compute(*[utils_preproc.merge_img(
            img[i_ID], i_ID) for i_ID in img_type_index])
    img_m_lst, img_m_key = zip(*w3_futures)

    # free up memory
    del w3_futures
    gc.collect()

    # ---- if selected, merge RGB and DEM bands into one image (xarray)
    # such that all band are used for horizontal coreg with arosics
    img_m_lst_out = []
    img_m_key_out = []
    img_m_file_out = []
    if PARAM.save_files_separate:
        # option for no merge
        img_m_lst_out.append(img_m_lst)
        img_m_key_out.append(img_m_key)
        img_m_file_out.append(
            [os.path.join(
                PARAM.PATH_IO,
                f"{df_meta.loc[i_key, 'file_n'][0].split('.')[0]}_preproc.tif")
            for i_key in img_m_key])
    if PARAM.merge_bands is not None:
        # option for merging
        logging.info(f'==== merge bands ====: {utils_preproc.get_timenow_str()}')
        img_m_lst_out = []
        img_m_key_out = []
        img_m_file_out = []
        for i_key, i_lst in PARAM.merge_bands.items():
            img_m = [img_m_lst[img_m_key.index(x)] for x in i_lst]
            img_concat = xarray.concat(img_m, dim='band')
            img_concat.attrs.update({'long_name':
                                     list(img_concat.coords['band'].values)})
            img_m_lst_out.append(img_concat)
            img_m_key_out.append(i_key)
            img_m_file_out.append(
                os.path.join(PARAM.PATH_IO, f"{i_key}_preproc.tif"))

    # %% ====== save files and metadata ====
    # ------- create DataFrame with saved files
    df_merged = pd.DataFrame(
        np.stack([img_m_key_out, img_m_file_out]).T,
        columns=['img_key', 'f_path']).set_index('img_key')
    # save
    df_merged.to_csv(
        os.path.join(PARAM.PATH_IO,
                     f'{PARAM.PROJ_NAME}_preproc_files.txt'),
        sep='\t', header=True)

    # ------- save files in parallel
    logging.info(f'==== save merged ====: {utils_preproc.get_timenow_str()}')
    n_jobs = len(img_m_file_out)
    Parallel(n_jobs=n_jobs, verbose=0, backend='loky')(delayed(
        utils_preproc.img_save)(i_img, i_file) for i_img, i_file in zip(
            img_m_lst_out, img_m_file_out))

    # %% ======= create and save mask (used foe arosics and xdem)
    logging.info(f'==== create mask ====: {utils_preproc.get_timenow_str()}')
    check_mask = xarray.concat(img_m_lst, dim='band')
    mask_arosics = (check_mask.isnull()).any(
        dim='band', keep_attrs=False).astype(int)
    mask_inv_save_path = os.path.join(PARAM.PATH_IO,
                                      'arosics_coregmask.tif')
    mask_arosics.rio.to_raster(mask_inv_save_path)

    # %% ==== save time stats
    utils_preproc.save_time_stats(
        prof, PARAM.PATH_IO, f'{PARAM.PROJ_NAME}_preproc')

    logging.info(f'==== finished ====: {utils_preproc.get_timenow_str()}')

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess imagery for co-registration')
    parser.add_argument(
        '--PARAM_CLASS', type=str,
        help=('name of class with parameters (default: UAV_coreg_steps_no_merge)'),
        default="UAV_coreg_steps_no_merge")
    args = parser.parse_args()
    use_client = True

    if use_client:
        # if use dask local cluster
        cluster = LocalCluster(n_workers=6, threads_per_worker=6)
        client = Client(cluster)
    main(**vars(args))