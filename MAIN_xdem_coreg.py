'''
Coregister a DEM to a reference DEM using xdem and apply corrections
to the RGB or other raster files

inputs are:
 - preprocessed raster files (.tif), reference and target (need to have same
   coordinate system and extent)
 - nodata mask (if defined to be used)
parameters:
 - all required parameters must be defined as a class in param_coreg.py


Steps:
run xdem coregistraton on DEM and apply corrections to other raster (e.g. RGB)

------
usage: MAIN_xdem_coreg.py [-h] [--PARAM_CLASS PARAM_CLASS]

Do horizontal 3d co-registration based on DSM
options:
  -h, --help            show this help message and exit
  --PARAM_CLASS PARAM_CLASS
                        name of Parameter class

run in background with
nohup python ./MAIN_xdem_coreg.py --PARAM_CLASS UAV_coreg_steps_no_merge &>./xdem_coreg.log &
'''

import sys
import os
import argparse
import numpy as np
import pandas as pd
import rioxarray
import logging
import datetime as dt
import xdem
import matplotlib.pyplot as plt
from rasterio.transform import Affine as rasterio_affine
from rasterio.enums import Resampling
from joblib import Parallel, delayed

# get path to current directory to load modules
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
import param_coreg
import utils_preproc
import utils_xdem_coreg


import control_file

def main(PARAM_CLASS="UAV_coreg_steps_no_merge"):
    # %% ======= setup PATHS and parameters
    PATH = control_file.get_proc_paths()
    # ------------ get param -----------
    PARAM = utils_preproc.create_param_class_instance(
        "param_coreg", PARAM_CLASS, PATH.PATH_BASE, proc_step=3)

    # -------------- setup time control ----------------
    prof = utils_preproc.setup_time_control()

    # ---- create log file and initialize errors to be written to console
    utils_preproc.init_logging(
        log_file=os.path.join(PARAM.PATH_IO,
        f'{PARAM.PROJ_NAME}_xdem_coreg_log_errors.log'))
    logging.info(f'=== Proc start ===: {utils_preproc.get_timenow_str()}')

    # ----- get names of preprocessed files
    df_merged = pd.read_csv(PARAM.preproc_files_df, sep='\t',
                            header=0).set_index('img_key')
    # --- get mask
    mask_inv_save_path = PARAM.mask_file
    mask = rioxarray.open_rasterio(
            mask_inv_save_path, masked=False, chunks='auto')
    mask = ~mask.astype(bool).values


    # %% ========== 4 - 7) SIMPLE 3d co-registration using xdem ============
    # example taken from
    # https://xdem.readthedocs.io/en/stable/basic_examples/plot_nuth_kaab.html#sphx-glr-basic-examples-plot-nuth-kaab-py

    # ------ read pre-processed images from files into xdem format
    ref_fname_DEM = df_merged.loc[*PARAM.ref_file_keys]
    target_fname_DEM = df_merged.loc[*PARAM.target_file_keys]
    reference_DEM = xdem.DEM(ref_fname_DEM)
    target_DEM = xdem.DEM(target_fname_DEM)

    # get names of RGB (or other band files) onto which the corrections
    # should be applied as well
    target_fname_RGB = df_merged.loc[*PARAM.apply_file_keys]

    # ---- add vertical refernce ssystem -----
    reference_DEM.set_vcrs(PARAM.vertical_crs)
    target_DEM.set_vcrs(PARAM.vertical_crs)

    # make sure that rasters have same grid and crs
    target_DEM = target_DEM.reproject(reference_DEM)

    # ----- initialize figure (figure with 3 subplots in one row) -----
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True,
                           figsize=(11.7, 8.3))

    # ----- plot difference between DEMs before coregistration -----
    diff_before = reference_DEM - target_DEM
    diff_before.plot(cmap="RdYlBu", vmin=-5, vmax=5,
                     cbar_title="Elevation change (m)", ax=ax[0])
    ax[0].set_title('uncorrected')

    # %% ========= xdem co-registration =========

    count = 0
    coreg_out_lst = []
    coreg_out_fname_lst = []
    coreg_matrix_lst = []
    coreg_name_lst = []
    for i_coreg_name, i_coreg_class in PARAM.xdem_coreg_algo.items():

        logging.info(f'=== Coreg start ({i_coreg_name}) ===: {utils_preproc.get_timenow_str()}')

        # --- apply coreg
        coreg_dem_out = i_coreg_class.fit_and_apply(
            reference_DEM, target_DEM, mask)

        # ---- save coregistration metadata
        matrix_lst = []
        if 'pipeline' in i_coreg_class.__dict__:
            # for combined coregistration pipleine
            for e_pipe, i_pipe in enumerate(i_coreg_class.pipeline):
                meta_txt = utils_xdem_coreg.dict_to_text(
                    i_pipe.meta['outputs'])
                utils_xdem_coreg.save_to_txt(
                    meta_txt, PARAM.PATH_IO,
                    f'{target_fname_DEM.split('.')[0]}_{i_coreg_name}_{e_pipe:02d}_metadata_outputs')
                #utils_xdem_coreg.save_metadata_xdem_coreg(
                #    i_pipe.meta['outputs'], PARAM.PATH_IO,
                #    f'{target_fname_DEM.split('.')[0]}_{i_coreg_name}_{e_pipe:02d}_metadata_outputs')
                #shift_lst.append(i_pipe.meta['outputs']['affine'])
                matrix_lst.append(i_pipe.to_matrix())
        else:
            meta_txt = utils_xdem_coreg.dict_to_text(
                i_coreg_class.meta['outputs'])
            utils_xdem_coreg.save_to_txt(
                meta_txt, PARAM.PATH_IO,
                f'{target_fname_DEM.split('.')[0]}_{i_coreg_name}_metadata_outputs')
            #utils_xdem_coreg.save_metadata_xdem_coreg(
            #    i_coreg_class.meta['outputs'], PARAM.PATH_IO,
            #    f'{target_fname_DEM.split('.')[0]}_{i_coreg_name}_metadata_outputs')
            #shift_lst.append(i_coreg_class.meta['outputs']['affine'])
            matrix_lst.append(i_coreg_class.to_matrix())

        #meta_info = utils_xdem_coreg.get_output_info_as_str(i_coreg_class)
        #utils_xdem_coreg.save_to_txt(
        #    meta_info, PARAM.PATH_IO,
        #    f'{target_fname_DEM.split('.')[0]}_{i_coreg_name}_metadata_info')

        #logging.info(meta_info)

        # --- plot remaining shift (elev diff)
        diff_after_coreg = reference_DEM - coreg_dem_out
        diff_after_coreg.plot(
            cmap="RdYlBu", vmin=-5, vmax=5,
            cbar_title="Elevation change (m)", ax=ax[1])
        ax[count + 1].set_title(i_coreg_name)

        # --- add file to list for parallel saving
        # !!! output images are in xdem format
        coreg_out_lst.append(coreg_dem_out)
        coreg_out_fname_lst.append(
            os.path.join(
                f"{target_fname_DEM.split('.')[0]}_coreg_{i_coreg_name}_DSM.tif"))
        coreg_matrix_lst.append(matrix_lst)
        coreg_name_lst.append(i_coreg_name)

        count += 1

    ## %% ==== test other combined corregistration pipeline =====
    ## combine two coregistration methods
    #coreg_pipeline2 = xdem.coreg.ICP() + xdem.coreg.NuthKaab()

    ## --- apply coreg
    #aligned_dem2 = coreg_pipeline2.fit_and_apply(
    #    reference_DEM, target_DEM, mask)

    ## ---- save coregistration outputs
    #utils_xdem_coreg.save_metadata_xdem_coreg(
    #    coreg_pipeline2.meta['outputs'], PARAM.PATH_IO,
    #    f'{target_fname_DEM.split('.')[0]}_ICPNuthKaab_metadata_outputs')
    #meta_info = utils_xdem_coreg.get_output_info_as_str(coreg_pipeline2)
    #utils_xdem_coreg.save_to_txt(
    #    meta_info, PARAM.PATH_IO,
    #    f'{target_fname_DEM.split('.')[0]}_ICPNuthKaab_metadata_info')
    #print([coreg_pipeline2.meta["outputs"]["affine"][s] for s in
    #    ["shift_x", "shift_y", "shift_z"]])

    ## --- plot remaining shift (elev diff)
    #diff_after2 = reference_DEM - aligned_dem2
    #diff_after2.plot(cmap="RdYlBu", vmin=-5, vmax=5,
    #                cbar_title="Elevation change (m)", ax=ax[2])
    #ax[2].set_title('ICP and NuthKaab')


    # --------- save corrected DEM and apply orrecton to RGB---------
    # save files in parallel
    logging.info(f'==== save coreg DEMs ====: {utils_preproc.get_timenow_str()}')
    n_jobs = len(coreg_out_lst)
    Parallel(n_jobs=n_jobs, verbose=0, backend='loky')(delayed(
        utils_xdem_coreg.save_xdem_img)(i_img, i_file) for i_img, i_file in zip(
            coreg_out_lst, coreg_out_fname_lst))

    logging.info(f'==== apply shift to RGB and save ====: {utils_preproc.get_timenow_str()}')
    n_jobs = len(coreg_out_lst)
    Parallel(n_jobs=n_jobs, verbose=0, backend='loky')(delayed(
        utils_xdem_coreg.apply_xy_shift_to_raster_xdem)(
            target_fname_RGB, i_fp_ref, i_shifts, i_name, PARAM.vertical_crs)
                for i_fp_ref, i_shifts, i_name in zip(
                    coreg_out_fname_lst, coreg_matrix_lst,
                    coreg_name_lst))


    # ---- apply coregistration to RGB
    #path_DEM_out = os.path.join(
    #    f"{target_fname_DEM.split('.')[0]}_coreg_NuthKaab.tif")
    #aligned_dem1.save(path_DEM_out, nodata=np.nan)
    #del aligned_dem1

    # re-read file in xarray
    #dem_aligned_1 = rioxarray.open_rasterio(
    #    path_DEM_out, masked=False, chunks='auto')

    ## ------ apply shift to RGB ------
    ## read RGB
    #RGB_shift = rioxarray.open_rasterio(
    #    target_fname_RGB, masked=False, chunks='auto')
    ##-- get horizontal shift
    #shift_x = nuth_kaab._meta['outputs']['affine']['shift_x']
    #shift_y = nuth_kaab._meta['outputs']['affine']['shift_y']

    ## shift cooridnates
    #RGB_shift.coords["x"] = RGB_shift.coords["x"] + shift_x
    #RGB_shift.coords["y"] = RGB_shift.coords["y"] + shift_y

    ## reproject to be aligned with DEM
    #RGB_shift = RGB_shift.rio.reproject_match(
    #    dem_aligned_1, Resampling=Resampling['bilinear'])

    ## ---- save coregistered DEMs
    #path_RGB_out = os.path.join(
    #    f"{target_fname_RGB.split('.')[0]}_coreg_NuthKaab_RGB.tif")
    #utils_preproc.img_save(RGB_shift, path_RGB_out)


    ## Get the current affine transform
    ##transform_curr = RGB_shift.rio.transform()

    ## Apply the shifts
    ##shifted_transform = rasterio_affine(
    ##    transform_curr.a, transform_curr.b, transform_curr.c + shift_x,
    ##
    ## write transform t raster
    ## Assign the new transform to the raster
    ##RGB_shift.rio.write_transform(shifted_transform, inplace=True)

    ## %% ======== 8) save outputs
    ## ---- save figure with elevation differences
    #path_out = os.path.join(
    #    PARAM.PATH_IO,
    #    f"{target_fname_DEM.split('.')[0]}_coreg_figure.pdf")
    #fig.savefig(path_out, format='pdf')

    path_out = os.path.join(
        PARAM.PATH_IO,
        f"{target_fname_DEM.split('.')[0]}_coreg_{'_'.join(coreg_name_lst)}_fig.pdf")
    fig.savefig(path_out, format='pdf')

    # --- display plot
    # !!! (on linux $DIPLAY environmental variable must have been defined
    # in the current terminal)
    plt.show()

    # ----- save time statistics
    utils_preproc.save_time_stats(
        prof, PARAM.PATH_IO, f'{PARAM.PROJ_NAME}_xdem_coreg')

    logging.info(f'==== finished ====: {utils_preproc.get_timenow_str()}')

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Do horizontal 3d co-registration based on DSM')
    parser.add_argument(
        '--PARAM_CLASS', type=str,
        help=('name of Parameter class'), default="UAV_coreg_steps_no_merge")
    args = parser.parse_args()

    main(**vars(args))