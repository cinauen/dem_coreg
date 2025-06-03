
import os

class LiBP_coreg:
    def __init__(self, PATH):
        # ============ DEFINE INPUT ==================
        # === define project parameters =====
        PATH.PATH_IO = os.path.join(
            PATH.PATH_BASE, 'Fieldwork', 'Alaska_processing',
            '2_proc', 'LiBP', 'BAP24-S', 'coreg_test')

        # -- project name (ise used for file prefix e.g. for logging file)
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

