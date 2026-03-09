"""
Find the location of each recorded cell for a given subject (in CCF coordinates).

This script will update the celldb_STUDYNAME_coords.h5 database with cell locations
for one subject. If that database doesn't exist yet, it will create one based on
the basic database created by database_generation.py

You need to specify the subject to analyze, as in:
python database_cell_locations.py test000
"""

import os
import sys
import numpy as np
from jaratoolbox import celldatabase
from jaratoolbox import histologyanalysis as ha  # This requires the Allen SDK
import matplotlib.pyplot as plt
from jaratoolbox import settings
from jaratoolbox import extraplots
studyparams = __import__('2025acpop.studyparams').studyparams

from importlib import reload
reload(ha)
reload(celldatabase)

if len(sys.argv)==2:
    subjects = [sys.argv[1]]
else:
    subjects = studyparams.SUBJECTS
    #raise ValueError('You need to specify which subject to process.')

#nRows = {'feat014':1, 'feat015':1, 'feat016':, 'feat017', 'feat018', 'feat019'}

histoColumns = ['recordingSiteName', 'x_coord', 'y_coord', 'z_coord']
histoColumnsDefaultValue = [None, np.nan, np.nan, np.nan]

dbPath = os.path.join(settings.DATABASE_PATH, studyparams.STUDY_NAME)
dbCoordsFilename = os.path.join(dbPath, f'celldb_{studyparams.STUDY_NAME}_coords.h5')
if os.path.isfile(dbCoordsFilename):
    celldb = celldatabase.load_hdf(dbCoordsFilename)
else:
    dbBasicFilename = os.path.join(dbPath, f'celldb_{studyparams.STUDY_NAME}_basic.h5')
    celldb = celldatabase.load_hdf(dbBasicFilename)
    histoColumnsDict = dict(zip(histoColumns, histoColumnsDefaultValue))
    celldb = celldb.assign(**histoColumnsDict)

for subject in subjects:
    print(f'Processing {subject}')
    celldbOneSubject = celldb[celldb.subject==subject]
    celldbOneSubjectCoords = ha.cell_locations(celldbOneSubject)

    # Unfortunately, there is a bug with update https://github.com/pandas-dev/pandas/issues/55509
    #celldb.update(celldbOneSubjectCoords) # The bug would change ints to floats
    celldb = celldb.combine_first(celldbOneSubjectCoords)

    celldatabase.save_hdf(celldb, dbCoordsFilename)

    # -- Show all recording sites for this subject --
    aa = ha.AllenAverageCoronalAtlas()
    aa.add_points_from_db(celldb)  # Show for all subjects
    aa.add_points_from_db(celldbOneSubjectCoords)
    aa.show_all_sites(nRows=None, areas=['AUDp','AUDv','AUDd','AUDpo'])

    plt.show()

    if 0:
        figFilename = f'all_sites_{subject}'
        extraplots.save_figure(figFilename, 'jpg', [16, 9], outputDir=dbPath, facecolor='k')
