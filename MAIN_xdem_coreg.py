
'''
Coregister a DEM to a reference DEM (using xdem)

inputs are raster files (.tif)

Steps:
run xdem oreg

run with
nohup python ./MAIN_xdem_coreg.py &>./xdem_coreg.log &
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

import param_coreg
import utils_dem_coreg


# get path to current directory to load modules
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import control_file_templ

def main(PARAM_CLASS="UAV_coreg_steps_no_merge"):
    PATH = control_file_templ.get_proc_paths()

    # ------------ get param -----------
    PARAM = utils_dem_coreg.create_param_class_instance(
        "param_coreg", PARAM_CLASS, PATH.PATH_BASE, proc_step=3)


    # -------------- setup time control ----------------
    prof = utils_dem_coreg.setup_time_control()

    # === create log file and initialize errors to be written to console
    utils_dem_coreg.init_logging(
        log_file=os.path.join(PARAM.PATH_IO,
        f'{PARAM.PROJ_NAME}_xdem_coreg_log_errors.log'))
    logging.info(
        '==== Proc start ====:'
        + dt.datetime.now().strftime('%Y-%m-%d_%H:%M'))

    # ====== read AOI (used for clipping)
    AOI_FILE = os.path.join(
        PARAM.PATH_IO, PARAM.PROCESSING_AOI)
    AOI_coords, AOI_poly = utils_dem_coreg.read_transl_geojson_AOI_coords_single(
        AOI_FILE, PARAM.EPSG_OUT)

    df_merged = pd.read_csv(PARAM.preproc_files_df, sep='\t',
                            header=0).set_index('img_key')

    mask_inv_save_path = PARAM.mask_file
    mask = rioxarray.open_rasterio(
            mask_inv_save_path, masked=False, chunks='auto')
    mask = ~mask.astype(bool).values


    # %% ========== 4 - 7) SIMPLE 3d co-registration using xdem ============
    # example taken from
    #  https://xdem.readthedocs.io/en/stable/basic_examples/plot_nuth_kaab.html#sphx-glr-basic-examples-plot-nuth-kaab-py

    # ------ read preprocessed images from files into xdem format
    ref_fname_inp = df_merged.loc['REF_DSM', 'f_path']
    target_fname_inp = df_merged.loc['TAR_DSM', 'f_path_local_coreg']
    reference_dem = xdem.DEM(ref_fname_inp)
    target_dem = xdem.DEM(target_fname_inp)

    # ---- add vertical refernce ssystem
    reference_dem.set_vcrs("Ellipsoid")
    target_dem.set_vcrs("Ellipsoid")

    # make sure that rasters have same grid and crs
    target_dem = target_dem.reproject(reference_dem)


    # ------ initlaize figure (figure with 3 subplots in one row)
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True,
                        figsize=(11.7, 8.3))

    # ------- plot difference between DEMs before coregistration
    diff_before = reference_dem - target_dem
    diff_before.plot(cmap="RdYlBu", vmin=-5, vmax=5,
                     cbar_title="Elevation change (m)", ax=ax[0])
    ax[0].set_title('uncorrected')

    # -------- nuth kaab co-registration -----
    # aset-up NuthKaab coregistration
    nuth_kaab = xdem.coreg.NuthKaab()
    # apply coreg
    aligned_dem1 = nuth_kaab.fit_and_apply(
        reference_dem, target_dem, mask)
    print([nuth_kaab.meta["outputs"]["affine"][s] for s in
        ["shift_x", "shift_y", "shift_z"]])
    # check for remaining shift (elev diff)
    diff_after1 = reference_dem - aligned_dem1
    diff_after1.plot(cmap="RdYlBu", vmin=-5, vmax=5,
                    cbar_title="Elevation change (m)", ax=ax[1])
    ax[1].set_title('NuthKaab')

    # ---- other combined corregistration pipeline
    # combine two coregistration methods
    coreg_pipeline2 = xdem.coreg.ICP() + xdem.coreg.NuthKaab()
    # apply coreg
    aligned_dem2 = coreg_pipeline2.fit_and_apply(
        reference_dem, target_dem, mask)
    # check for remaining shift (elev diff)
    diff_after2 = reference_dem - aligned_dem2
    diff_after2.plot(cmap="RdYlBu", vmin=-5, vmax=5,
                    cbar_title="Elevation change (m)", ax=ax[2])
    ax[2].set_title('ICP and NuthKaab')

    # %% ======== 8) save outputs
    # ---- save figure with elevation differences
    path_out = os.path.join(
        PARAM.PATH_IO,
        f"{PARAM.TARGET_FNAME_LST[0].split('.')[0]}_coregister_figure.pdf")
    fig.savefig(path_out, format='pdf')

    # ---- save coregistered DEMs
    path_out = os.path.join(
        f"{PARAM.TARGET_FNAME[0].split('.')[0]}_coregister_NuthKaab.tif")
    aligned_dem1.save(path_out, nodata=np.nan)

    path_out = os.path.join(
        PARAM.PATH_IO,
        f"{PARAM.TARGET_FNAME_LST[0].split('.')[0]}_coregister_ICP_NuthKaab.tif")
    aligned_dem2.save(path_out, nodata=np.nan)

    # --- display plot
    # !!! (on linux $DIPLAY environmental variable must have been defined
    # in the current terminal)
    plt.show()

    utils_dem_coreg.save_time_stats(
        prof, PARAM.PATH_IO, f'{PARAM.PROJ_NAME}_xdem_coreg')


    print('----- FINISH ----')

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Do horizontal 3d co-registration based on DSM')
    parser.add_argument(
        '--PARAM_CLASS', type=str,
        help=('name of Parameter class'), default="UAV_coreg_steps_no_merge")
    args = parser.parse_args()

    main(**vars(args))