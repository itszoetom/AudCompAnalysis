"""
Show an image with the recording locations for all subjects in the database.

USAGE:
- You first need to run database_cell_locations and save the resulting database.
- You also need to run this script inside the allensdk environment.

python figure_recording_locations.py [subject] [session]
python figure_recording_locations.py [subject]
python figure_recording_locations.py
"""

import os
import sys
sys.path.append('..')
import numpy as np
from jaratoolbox import celldatabase
from jaratoolbox import histologyanalysis as ha  # This requires the Allen SDK
import matplotlib.pyplot as plt
from jaratoolbox import settings
from jaratoolbox import extraplots
import studyparams


SAVE_FIGURE = 0

if len(sys.argv)>1:
    subject = sys.argv[1]
else:
    subject = None  #studyparams.SUBJECTS
if len(sys.argv)>2:
    session = sys.argv[2]
else:
    session = None
    
dbPath = os.path.join(settings.DATABASE_PATH, studyparams.STUDY_NAME)
dbCoordsFilename = os.path.join(dbPath,f'celldb_{studyparams.STUDY_NAME}_coords.h5')
celldb = celldatabase.load_hdf(dbCoordsFilename)

if subject is not None:
    celldb = celldb.query('subject==@subject')
if session is not None:
    celldb = celldb.query('date==@session')

# -- Show all recording sites --
aa = ha.AllenAverageCoronalAtlas()
aa.add_points_from_db(celldb)
aa.show_all_sites(nRows=None, areas=['AUDp','AUDv','AUDd','AUDpo'])
 
if SAVE_FIGURE:
    suffix = '_{subject}' if subject is not None else ''
    figFilename = f'all_sites{suffix}'
    #extraplots.save_figure(figFilename, 'jpg', [16, 9], outputDir=dbPath, facecolor='k')
    extraplots.save_figure(figFilename, 'jpg', [25, 13], outputDir=dbPath, facecolor='k')

