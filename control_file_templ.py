
'''
NON tracked control params
Here defines the base paths to the project. Paths are different
for linux and windows
'''
import os

class get_proc_paths:
    def __init__(self, system='linux'):
        if system == 'linux':
            # for linux
            self.PATH_BASE = os.path.normpath(
                r'/isipd/projects/Response/GIS_RS_projects/USER/')
        elif system == 'windows':
            # for windows
            self.PATH_BASE = os.path.normpath(
                r'//smb.isipd.dmawi.de/projects/Response/GIS_RS_projects/USER/')