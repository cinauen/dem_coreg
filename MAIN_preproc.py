
'''
Coregister a DEM to a reference DEM (using xdem)

inputs are raster files (.tif)

Steps:
1) read files (reference: file to which the dem should be coregistered to,
               target: the file which has to be coregistered)
2) preprocess files: clip to same extent and reproject to same
               resolution (min resolution) and merge subfiles if required
3) save pre-processed files

#4) read files into xdem format
#5) get difference between the initial dems
#6) run co-registration with two different pipelines
#7) get difference per pipeline and create plot
#8) save coregistered image

run with
nohup python ./MAIN_preproc.py &>./Preproc.log &
'''

import sys
import os
import argparse
import numpy as np
import pandas as pd
import logging
import datetime as dt
from joblib import Parallel, delayed, cpu_count
from dask.distributed import Client, LocalCluster
import gc
import xarray

import param_coreg
import utils_dem_coreg


# get path to current directory to load modules
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import control_file


def main(PARAM_CLASS="UAV_coreg_steps"):

    PATH = control_file.get_proc_paths()

    # ------------ get param -----------
    PARAM = utils_dem_coreg.create_param_class_instance(
        "param_coreg", PARAM_CLASS, PATH.PATH_BASE, proc_step=1)

    # -------------- setup time control ----------------
    prof = utils_dem_coreg.setup_time_control()

    # === create log file and initialize errors to be written to console
    utils_dem_coreg.init_logging(
        log_file=os.path.join(PARAM.PATH_IO,
        f'{PARAM.PROJ_NAME}_preproc_log_errors.log'))
    logging.info(
        '==== Proc start ====:'
        + dt.datetime.now().strftime('%Y-%m-%d_%H:%M'))

    # ====== read AOI (used for clipping)
    AOI_FILE = os.path.join(
        PARAM.PATH_IO, PARAM.PROCESSING_AOI)
    AOI_coords, AOI_poly = utils_dem_coreg.read_transl_geojson_AOI_coords_single(
        AOI_FILE, PARAM.EPSG_OUT)

    # %% ===== 1) read and preprocess files
    # preprocess raster files
    # clipping all to same extent and transform to
    # same resolution
    # (for xdem they must have the same
    # projection extent, bounds and resolutiom (thus exact same cooridnates))

    file_lst, file_n_lst, img_key, img_type, nodata_inp = utils_dem_coreg.get_inp_lst_parallel_proc(
        PARAM.REF_PATH, PARAM.REF_FNAME, PARAM.REF_NODATA_INP,
        PARAM.TARGET_PATH, PARAM.TARGET_FNAME, PARAM.TARGET_NODATA_INP,
        )
    df_meta = pd.DataFrame(
        np.stack([img_type, img_key, file_lst, file_n_lst]).T,
        columns=['pID', 'img_key', 'file_p', 'file_n']).groupby('pID').aggregate(list)

    logging.info(
        '==== read preproc ====:'
        + dt.datetime.now().strftime('%Y-%m-%d_%H:%M'))
    n_jobs = min(int(cpu_count()-10), len(img_key))
    w1 = Parallel(n_jobs=n_jobs, verbose=0, backend='loky')(delayed(
        utils_dem_coreg.img_read_preproc)(
            *i,
            PARAM.RESOLUTION_OUT, AOI_coords,
            PARAM.EPSG_INP, PARAM.EPSG_OUT, band_names=PARAM.band_names,
            bands_keep=PARAM.bands_keep)
                for i in zip(file_lst, img_key, nodata_inp))

    img_lst, img_key = zip(*w1)
    #df_merged =
    gc.collect()
    logging.info(
        '==== filter outliers ====:'
        + dt.datetime.now().strftime('%Y-%m-%d_%H:%M'))
    img_dict = {x: y for x, y in zip(img_key, img_lst)}
    img_dict['REF_DSM_0'] = utils_dem_coreg.filter_outliers(
        img_dict['REF_DSM_0'], window_size=6, std_fact=3,
        create_plot=False,
        fp_fig=os.path.join(PARAM.PATH_IO,
                            f'REF_DSM_outlier_fill.pdf'))

    logging.info(
        '==== reproject ====:'
        + dt.datetime.now().strftime('%Y-%m-%d_%H:%M'))
    w2 = Parallel(n_jobs=max(n_jobs - 1, 1), verbose=0, backend='loky')(delayed(
        utils_dem_coreg.reproject_match)(
            i_img, img_lst[0], i_key, PARAM.RESAMPLING_TYPE)
                for i_img, i_key in zip(img_lst[1:], img_key[1:]))

    img_lst0, img_key0 = zip(*w2)
    img_lst = [img_lst[0]] + list(img_lst0)
    img_key = [img_key[0]] + list(img_key0)
    gc.collect()

    img_dict = {x: y for x, y in zip(img_key, img_lst)}

    logging.info(
        '==== merge ====:'
        + dt.datetime.now().strftime('%Y-%m-%d_%H:%M'))
    if PARAM.merge_bands is None:
        save_merged = True
    else:
        save_merged = False
    img_type_index = df_meta.index.tolist()
    img = {x: [img_dict[xx] for xx in df_meta.loc[x, 'img_key']]
           for x in img_type_index}
    w3 = Parallel(n_jobs=max(n_jobs - 1, 1), verbose=0, backend='loky')(delayed(
        utils_dem_coreg.merge_save)(
            img[i_ID], i_ID, PARAM.PATH_IO,
            df_meta.loc[i_ID, 'file_n'][0].split('.')[0], save=save_merged)
                for i_ID in img_type_index)
    img_m_lst, img_m_key, fp_merge = zip(*w3)
    gc.collect()

    if PARAM.merge_bands is None:
        df_merged = pd.DataFrame(
            np.stack([img_m_key, fp_merge]).T,
            columns=['img_key', 'f_path']).set_index('img_key')
    else:
        logging.info(
        '==== merge bands ====:'
        + dt.datetime.now().strftime('%Y-%m-%d_%H:%M'))
        img_m_lst_out = []
        img_m_key_out = []
        img_m_file_out = []
        for i_key, i_lst in PARAM.merge_bands.items():
            img_m = [img_m_lst[img_m_key.index(x)] for x in i_lst]
            img_m_lst_out.append(xarray.concat(img_m, dim='band'))
            img_m_key_out.append(i_key)
            img_m_file_out.append(
                os.path.join(PARAM.PATH_IO, f"{i_key}_preproc.tif"))

        n_jobs = len(img_m_file_out)
        Parallel(n_jobs=n_jobs, verbose=0)(delayed(
            utils_dem_coreg.img_save)(*i) for i in zip(img_m_lst_out,
                                                       img_m_file_out))
        gc.collect()
        df_merged = pd.DataFrame(
            np.stack([img_m_key_out, img_m_file_out]).T,
            columns=['img_key', 'f_path']).set_index('img_key')

    df_merged.to_csv(
        os.path.join(PARAM.PATH_IO,
                     f'{PARAM.PROJ_NAME}_preproc_files.txt'),
        sep='\t', header=True)

    # ---- create and save mask
    logging.info(
        '==== create mask ====:'
        + dt.datetime.now().strftime('%Y-%m-%d_%H:%M'))
    check_mask = xarray.concat(img_m_lst, dim='band')
    mask_xdem = (~check_mask.isnull()).any(dim='band', keep_attrs=True)
    mask_arosics = (~mask_xdem).astype(int)

    mask_inv_save_path = os.path.join(PARAM.PATH_IO,
                                      'arosics_coregmask.tif')
    mask_arosics.rio.to_raster(mask_inv_save_path)

    utils_dem_coreg.save_time_stats(
        prof, PARAM.PATH_IO, f'{PARAM.PROJ_NAME}_preproc')

    logging.info(
        '==== finished ====:'
        + dt.datetime.now().strftime('%Y-%m-%d_%H:%M'))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess imagery for co-registration')
    parser.add_argument(
        '--PARAM_CLASS', type=str,
        help=('name of Parameter class'), default="UAV_coreg_steps")
    args = parser.parse_args()
    use_client = True
    if use_client:
        cluster = LocalCluster(n_workers=12, threads_per_worker=1)
        client = Client(cluster)
    main(**vars(args))