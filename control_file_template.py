
'''
NON tracked control params
Here defines the base paths to the project. Paths are different
for linux and windows
'''
import os

class get_proc_paths:
    def __init__(self, system='windows'):
        if system == 'windows':
            # for windows
            self.PATH_BASE = os.path.normpath(
                r'//ADJUST_PATHS/')