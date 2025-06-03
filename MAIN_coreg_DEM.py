
'''
Coregister a DEM to a reference DEM (using xdem)

inputs are raster files (.tif)

Steps:
1) read files (reference: file to which the dem should be coregistered to,
               target: the file which has to be coregistered)
2) preprocess files: clip to same extent and reproject to same
               resolution (min resolution) and merge subfiles if required
3) save pre-processed files
4) read files into xdem format
5) get difference between the initial dems
6) run co-registration with two different pipelines
7) get difference per pipeline and create plot
8) save coregistered image
'''

import sys
import os
import numpy as np
import pandas as pd
import rioxarray
import logging
import datetime as dt
import xdem
from rioxarray.merge import merge_arrays
from rasterio.enums import Resampling
import matplotlib.pyplot as plt

import param_coreg
import utils_dem_coreg

# get path to current directory to load modules
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import control_file
PATH = control_file.get_proc_paths()

# %% ================== set input =============================
# input output path (AOI file and for coreg ouput)
PARAM = param_coreg.LiBP_coreg(PATH)


# === create log file and initialize errors to be written to console
utils_dem_coreg.init_logging(
    log_file=os.path.join(PATH.PATH_IO,
    f'{PARAM.PROJ_NAME}_log_errors.log'))
logging.info(
    '==== Proc start ====:'
    + dt.datetime.now().strftime('%Y-%m-%d_%H:%M'))

# ====== read AOI (used for clipping)
AOI_FILE = os.path.join(
    PATH.PATH_IO, PARAM.PROCESSING_AOI)
AOI_coords, AOI_poly = utils_dem_coreg.read_transl_geojson_AOI_coords_single(
    AOI_FILE, PARAM.EPSG_TARGET)

# %% ===== 1) read DEM files
# --- read target files
target_lst = []
for i_img in PARAM.TARGET_FNAME_LST:
    target_lst.append(
        utils_dem_coreg.img_read(PARAM.TARGET_PATH, i_img,
        PARAM.TARGET_NODATA, PARAM.EPSG_INP))

# ---- read ref files
ref_lst = []
for i_img in PARAM.REF_FNAME_LST:
    ref_lst.append(
        utils_dem_coreg.img_read(PARAM.REF_PATH, i_img,
        PARAM.REF_NODATA, PARAM.EPSG_INP))

# %% ===== 2) pre-process dem fiels (for xdem they must have the same
# projection extent, bounds and resolutiom (thus exact same cooridnates))
# --- get minimum resolution of the target and the refernece image
res_target = min([x.rio.resolution()[0] for x in target_lst])
res_ref = min([x.rio.resolution()[0] for x in ref_lst])
res_out = min(res_target, res_ref)

# --- proc target files
# (clipping all to same extent and transform to
# same resolution)
target_proc = []
for i_img in target_lst:
    target_proc.append(
        utils_dem_coreg.img_preproc(i_img, res_out, AOI_coords,
                                    PARAM.EPSG_TARGET, np.nan))
del target_lst

# ------- pre-proc ref files
# (clipping all to same extent and transform to
# same resolution)
ref_proc = []
for i_img in ref_lst:
    ref_proc.append(
        utils_dem_coreg.img_preproc(i_img, res_out, AOI_coords,
                                    PARAM.EPSG_TARGET, np.nan))
del ref_lst

# ------- merge images if have more then one image item
if len(target_proc) > 1:
    target_img = merge_arrays(target_proc, method='first')
else:
    target_img = target_proc[0]
del target_proc

if len(ref_proc) > 1:
    ref_img = merge_arrays(ref_proc, method='first')
else:
    ref_img = ref_proc[0]
del ref_proc

# ---- make sure that the target and the reference image have exactly the
# coordinate system by projectoing the target to the same coords as the
# reference image
target_img = target_img.rio.reproject_match(
    ref_img, Resampling=Resampling.bilinear)

# ------- saved preprocessed image to file
# create output path
target_path_edit = os.path.join(
    PATH.PATH_IO, PARAM.TARGET_FNAME_LST[0].split('.')[0] + '_edit.tif')
# save target
target_img.rio.to_raster(raster_path=target_path_edit, write_nodata=True)

# create output path
ref_path_edit = os.path.join(
    PATH.PATH_IO, PARAM.REF_FNAME_LST[0].split('.')[0] + '_edit.tif')
# save reference
ref_img.rio.to_raster(raster_path=ref_path_edit, write_nodata=True)

# ----- create a mask
# (mask is True where have valid data, is False where eiter the data
# on the reference or the target image is missing)
mask = target_img.where(np.isnan(target_img), False, True)
mask = np.where(np.isnan(mask.values) | np.isnan(ref_img.values), False, True)


# %% ========== 4 - 7) SIMPLE 3d co-registration using xdem ============
# example taken from
#  https://xdem.readthedocs.io/en/stable/basic_examples/plot_nuth_kaab.html#sphx-glr-basic-examples-plot-nuth-kaab-py

# ------ read preprocessed images from files into xdem format
reference_dem = xdem.DEM(ref_path_edit)
target_dem = xdem.DEM(target_path_edit)

# ---- add vertical refernce ssystem
reference_dem.set_vcrs("Ellipsoid")
target_dem.set_vcrs("Ellipsoid")


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
    PATH.PATH_IO,
    f"{PARAM.TARGET_FNAME_LST[0].split('.')[0]}_coregister_figure.pdf")
fig.savefig(path_out, format='pdf')

# ---- save coregistered DEMs
path_out = os.path.join(
    PATH.PATH_IO,
    f"{PARAM.TARGET_FNAME_LST[0].split('.')[0]}_coregister_NuthKaab.tif")
aligned_dem1.save(path_out, nodata=np.nan)

path_out = os.path.join(
    PATH.PATH_IO,
    f"{PARAM.TARGET_FNAME_LST[0].split('.')[0]}_coregister_ICP_NuthKaab.tif")
aligned_dem2.save(path_out, nodata=np.nan)

# --- display plot
# !!! (on linux $DIPLAY environmental variable must have been defined
# in the current terminal)
plt.show()


print('----- FINISH ----')