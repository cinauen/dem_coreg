
import os
from joblib import cpu_count
import numpy as np



class LiBP_coreg:
    def __init__(self, PATH):
        # ============ DEFINE INPUT ==================
        # === define project parameters =====
        PATH.PATH_IO = os.path.join(
            PATH.PATH_BASE, 'Fieldwork', 'Alaska_processing',
            '2_proc', 'LiBP', 'BAP24-S', 'coreg_test')

        # -- project name (is used for file prefix e.g. for logging file)
        self.PROJ_NAME = 'LiBP_coreg'

        # ======= define input imagery ========
        # ---------- target imagery ----------
        # paths to target files
        self.TARGET_PATH = os.path.join(
            PATH.PATH_BASE, 'Fieldwork', 'Alaska_processing',
            '2_proc', 'LiBP', 'BAP24-S', 'coreg_test')
        # target files
        self.TARGET_FNAME_LST = [
            'BAP24S_LiBP_merged_raster_v1.tif']
        # nodata value of target imagery
        self.TARGET_NODATA = 36.86

        # --------- reference imagery ---------
        # path to refernece imagery
        self.REF_PATH = os.path.join(
            PATH.PATH_BASE, 'Fieldwork', 'Alaska_processing',
            '2_proc', 'LiBP', 'BAP24-S', 'coreg_test', 'MACS_reference')
        # reference imagery
        self.REF_FNAME_LST = [
            'WA_Kotzebue_20240711_15cm_01_DSM_7_2.tif',
            'WA_Kotzebue_20240711_15cm_01_DSM_7_3.tif']
        self.REF_NODATA = -32768

        # --- AOI area which should be used for processing
        # (file must be saved in target paths)
        self.PROCESSING_AOI = 'BAPS_LiBP_AOI.geojson'

        self.EPSG_INP = 32603
        self.EPSG_TARGET = 32603


class UAV_coreg:
    def __init__(self, PATH_BASE):
        # ============ DEFINE INPUT ==================
        # === define project parameters =====
        self.PATH_IO = os.path.join(
            PATH_BASE, 'Fieldwork', 'Alaska_processing',
            '2_proc', 'UAV', 'pix4d_proc', 'coreg_test')

        # -- project name (ise used for file prefix e.g. for logging file)
        self.PROJ_NAME = 'UAV_coreg'

        # ======= define input imagery ========
        # ---------- target imagery ----------
        # paths to target files
        self.TARGET_PATH = os.path.join(
            PATH_BASE, 'Fieldwork', 'Alaska_processing',
            '2_proc', 'UAV', 'pix4d_proc', 'coreg_test')
        # target files
        self.TARGET_FNAME = {
            'RGB': ['BAP23B_v02_EW_transparent_mosaic_group1.tif'],
            'DSM': ['BAP23B_v02_EW_dsm.tif']}

        # nodata value of target imagery
        self.TARGET_NODATA_INP = {'RGB': 0, 'DSM': -1000}

        # --------- reference imagery ---------
        # path to refernece imagery
        self.REF_PATH = os.path.join(
            PATH_BASE, 'Fieldwork', 'Alaska_processing',
            '2_proc', 'UAV', 'pix4d_proc', 'coreg_test')
        # reference imagery
        self.REF_FNAME = {
            'RGB': ['BAP22B_v03_EW_transparent_mosaic_group1.tif'],
            'DSM': ['BAP22B_v03_EW_dsm.tif']}

        self.REF_NODATA_INP = {'RGB': 0, 'DSM': -1000}

        # --- AOI area which should be used for processing
        # (file must be saved in target paths)
        self.PROCESSING_AOI = 'AOI_coreg.geojson'

        self.RESOLUTION_OUT = 0.05

        self.EPSG_INP = 32603
        self.EPSG_OUT = 32603


        self.COREG_CPU_NUM = int(cpu_count()-10)

        self.COREG_GLOBAL_MAX_SHIFT = 50  # given in pixel
        # default value is 5
        self.COREG_GLOBAL_WINDOW_SIZE = 256  # matching window size
        # in pixels default is (256, 256)

        self.COREG_LOCAL = True
        self.GRID_RES_PIX_LOCAL = 200  # !!!! this needs to be adjusted according
        # to the image resolution/size if too small get Memory error
        self.COREG_LOCAL_MAX_SHIFT = 50  # given in pixel
        # default value is 5
        self.COREG_LOCAL_WINDOW_SIZE = 512  # matching window size
        # in pixels default is (256, 256)

        # needs to be given as string because is evaluated according to
        # AOI poly !!!!! needs to be fully in image area !!!
        self.add_coreg_param_dict = None
        # "{'footprint_poly_ref': AOI_poly, 'footprint_poly_tgt':  AOI_poly, 'max_iter': 10}"

        self.SAVE_AS_COG = True  # is only applied is coreg files
        # (thus only aplied to final output)

        self.RESAMPLING_TYPE = 'cubic'  # 'linear', 'nearest' 'bilinear'


class UAV_coreg_steps:
    def __init__(self, PATH_BASE, proc_step):
        # ============ DEFINE INPUT ==================
        # === define project parameters =====
        self.PATH_IO = os.path.join(
            PATH_BASE, 'Fieldwork', 'Alaska_processing',
            '2_proc', 'UAV', 'pix4d_proc', 'coreg_test')

        # -- project name (ise used for file prefix e.g. for logging file)
        self.PROJ_NAME = 'UAV_coreg'

        # ======= define input imagery ========
        # ---------- target imagery ----------
        # paths to target files
        self.TARGET_PATH = os.path.join(
            PATH_BASE, 'Fieldwork', 'Alaska_processing',
            '2_proc', 'UAV', 'pix4d_proc', 'coreg_test')
        # target files
        self.TARGET_FNAME = {
            'RGB': ['BAP23B_v02_EW_transparent_mosaic_group1.tif'],
            'DSM': ['BAP23B_v02_EW_dsm.tif']}

        # nodata value of target imagery
        self.TARGET_NODATA_INP = {'RGB': 0, 'DSM': -10000}

        # --------- reference imagery ---------
        # path to refernece imagery
        self.REF_PATH = os.path.join(
            PATH_BASE, 'Fieldwork', 'Alaska_processing',
            '2_proc', 'UAV', 'pix4d_proc', 'coreg_test')
        # reference imagery
        self.REF_FNAME = {
            'RGB': ['BAP22B_v03_EW_transparent_mosaic_group1.tif'],
            'DSM': ['BAP22B_v03_EW_dsm.tif']}

        self.REF_NODATA_INP = {'RGB': 0, 'DSM': -10000}
        self.band_names = {'RGB': ['red', 'green', 'blue', 'alpha'],
                           'DSM': ['elev']}
        self.bands_keep = {'RGB': ['red', 'green', 'blue'],
                           'DSM': ['elev']}

        # merge RGB and DEM into one for preprocessing arosics coreg
        self.merge_bands = {'REF_BAP22B': ['REF_RGB', 'REF_DSM'],
                            'TAR_BAP23B': ['TAR_RGB', 'TAR_DSM']
                            }
        self.coreg_ref = 'REF_BAP22B'
        self.coreg_target = ['TAR_BAP23B']  # if list then first is used to get
        # coregistration, whic is the applied to all the other images

        # --- AOI area which should be used for processing
        # (file must be saved in target paths)
        self.PROCESSING_AOI = 'AOI_coreg.geojson'

        self.RESOLUTION_OUT = 0.05

        self.EPSG_INP = 32603
        self.EPSG_OUT = 32603

        self.dask_chunks = {'band': 4, 'x': 5120, 'y': 5120}

        self.RESAMPLING_TYPE = 'cubic'  # 'linear', 'nearest' 'bilinear'



        # filter file
        self.filter_img_key = 'DSM'
        # --- remove below or above min max
        self.filter_min = None
        self.filter_max = 100  # if None then nothing is done
        # --- filter according to moving window stats
        self.fitler_moving_window_outlier = False
        self.filter_window_size_px = 600
        self.filter_std_fact = 6

        if proc_step == 2:

            self.mask_file = os.path.join(
                self.PATH_IO, 'arosics_coregmask.tif')
            self.preproc_files_df = os.path.join(
                self.PATH_IO, f'{self.PROJ_NAME}_preproc_files.txt')

            self.COREG_CPU_NUM = 30  # int(cpu_count()-10)

            self.COREG_GLOBAL_MAX_SHIFT = 50  # given in pixel
            # default value is 5
            self.COREG_GLOBAL_WINDOW_SIZE = 256  # matching window size
            # in pixels default is (256, 256)

            self.COREG_LOCAL = True
            self.GRID_RES_PIX_LOCAL = 25  # !!!! this needs to be adjusted according
            # to the image resolution/size if too small get Memory error
            self.COREG_LOCAL_MAX_SHIFT = 50  # given in pixel
            # default value is 5
            self.COREG_LOCAL_WINDOW_SIZE = 512  # matching window size
            # in pixels default is (256, 256)

            # needs to be given as string because is evaluated according to
            # AOI poly !!!!! needs to be fully in image area !!!
            self.add_coreg_param_dict = None
            # "{'footprint_poly_ref': AOI_poly, 'footprint_poly_tgt':  AOI_poly, 'max_iter': 10}"

            self.SAVE_AS_COG = True  # is only applied is coreg files
            # (thus only aplied to final output)


        if proc_step == 3:
            self.mask_file = os.path.join(
                self.PATH_IO, 'arosics_coregmask.tif')
            self.preproc_files_df = os.path.join(
                self.PATH_IO,
                f'{self.PROJ_NAME}_arosics_coreg_files.txt')



class UAV_coreg_steps_no_merge:
    def __init__(self, PATH_BASE, proc_step):
        # ============ DEFINE INPUT ==================
        # === define project parameters =====
        self.PATH_IO = os.path.join(
            PATH_BASE, '1B_proc', '3_gully_sites',
            'UAV_coreg_test')
            #PATH_BASE, 'Fieldwork', 'Alaska_processing',
            #'2_proc', 'UAV', 'pix4d_proc', 'coreg_test')

        # -- project name (ise used for file prefix e.g. for logging file)
        self.PROJ_NAME = 'UAV_coreg_NO_merge'

        # ======= define input imagery ========
        # ---------- target imagery ----------
        # paths to target files
        self.TARGET_PATH = os.path.join(
            PATH_BASE, '1B_proc', '3_gully_sites',
            'UAV_coreg_test')
            #PATH_BASE, 'Fieldwork', 'Alaska_processing',
            #'2_proc', 'UAV', 'pix4d_proc', 'coreg_test')
        # target files
        self.TARGET_FNAME = {
            'RGB': ['BAP23B_v02_EW_transparent_mosaic_group1.tif'],
            'DSM': ['BAP23B_v02_EW_dsm.tif']}

        # nodata value of target imagery
        self.TARGET_NODATA_INP = {'RGB': 0, 'DSM': -10000}

        # --------- reference imagery ---------
        # path to refernece imagery
        self.REF_PATH = os.path.join(
            PATH_BASE, '1B_proc', '3_gully_sites',
            'UAV_coreg_test')
            #PATH_BASE, 'Fieldwork', 'Alaska_processing',
            #'2_proc', 'UAV', 'pix4d_proc', 'coreg_test')
        # reference imagery
        self.REF_FNAME = {
            'RGB': ['BAP23B_UAV_test_RGB.tif'],
            'DSM': ['BAP23B_UAV_test_elev.tif']}

        self.REF_NODATA_INP = {'RGB': 0, 'DSM': 0}  # {'RGB': 0, 'DSM': -10000}
        self.band_names = {'REF_RGB': ['red', 'green', 'blue'],
                           'REF_DSM': ['elev'],
                           'TAR_RGB': ['red', 'green', 'blue', 'alpha'],
                           'TAR_DSM': ['elev']}
        self.bands_keep = {'REF_RGB': ['red', 'green', 'blue'],
                           'REF_DSM': ['elev'],
                           'TAR_RGB': ['red', 'green', 'blue'],
                           'TAR_DSM': ['elev']}

        # merge RGB and DEM into one for preprocessing arosics coreg
        self.merge_bands = None
        self.coreg_ref = 'REF_RGB'
        self.coreg_target = ['TAR_RGB', 'TAR_DSM']  # if list then first is used to get
        # coregistration, whic is the applied to all the other images

        # --- AOI area which should be used for processing
        # (file must be saved in target paths)
        self.PROCESSING_AOI = 'AOI_coreg.geojson'

        self.RESOLUTION_OUT = 0.05

        self.EPSG_INP = 32603
        self.EPSG_OUT = 32603

        self.dask_chunks = {'band':3, 'x': 5120, 'y': 5120}

        self.RESAMPLING_TYPE = 'cubic'  # 'linear', 'nearest' 'bilinear'



        # filter file
        self.filter_img_key = 'DSM'
        # --- remove below or above min max
        self.filter_min = None
        self.filter_max = 100  # if None then nothing is done
        # --- filter according to moving window stats
        self.fitler_moving_window_outlier = False
        self.filter_window_size_px = 600
        self.filter_std_fact = 6

        if proc_step == 2:

            self.mask_file = os.path.join(
                self.PATH_IO, 'arosics_coregmask.tif')
            self.preproc_files_df = os.path.join(
                self.PATH_IO, f'{self.PROJ_NAME}_preproc_files.txt')

            self.COREG_CPU_NUM = 30  # int(cpu_count()-10)

            self.COREG_GLOBAL_MAX_SHIFT = 50  # given in pixel
            # default value is 5
            self.COREG_GLOBAL_WINDOW_SIZE = 256  # matching window size
            # in pixels default is (256, 256)

            self.COREG_LOCAL = True
            self.GRID_RES_PIX_LOCAL = 400  # !!!! this needs to be adjusted according
            # to the image resolution/size if too small get Memory error
            self.COREG_LOCAL_MAX_SHIFT = 50  # given in pixel
            # default value is 5
            self.COREG_LOCAL_WINDOW_SIZE = 512  # matching window size
            # in pixels default is (256, 256)

            # needs to be given as string because is evaluated according to
            # AOI poly !!!!! needs to be fully in image area !!!
            self.add_coreg_param_dict = None
            # "{'footprint_poly_ref': AOI_poly, 'footprint_poly_tgt':  AOI_poly, 'max_iter': 10}"

            self.SAVE_AS_COG = True  # is only applied is coreg files
            # (thus only aplied to final output)


        if proc_step == 3:
            self.mask_file = os.path.join(
                self.PATH_IO, 'arosics_coregmask.tif')
            self.preproc_files_df = os.path.join(
                self.PATH_IO,
                f'{self.PROJ_NAME}_arosics_coreg_files.txt')