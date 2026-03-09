"""
Generate and save database containing basic information, stats, and indices for each cell.
"""

import os
from scipy import stats
from jaratoolbox import celldatabase
from jaratoolbox import spikesorting
from jaratoolbox import settings
studyparams = __import__('2025acpop.studyparams').studyparams


def calculate_base_stats(db):
    return db


def calculate_indices(db):
    return db


def calculate_cell_locations(db):
    pass

if __name__ == "__main__":
    # -- Generate cell database (this function excludes bad cells by default, see docs) --
    print(f"subjects: {studyparams.SUBJECTS}")

    celldb = celldatabase.generate_cell_database_from_subjects(studyparams.SUBJECTS)
    #subjectToAnalyze = 'feat018'
    #celldb = celldatabase.generate_cell_database_from_subjects(subjectToAnalyze)

    # -- Compute the base stats and indices for each cell --
    celldb = calculate_base_stats(celldb)  # Calculated for all cells
    celldb = calculate_indices(celldb)     # Calculated for a selected subset of cells

    dbPath = os.path.join(settings.DATABASE_PATH, studyparams.STUDY_NAME)
    dbFilename = os.path.join(dbPath,'celldb_{}_basic.h5'.format(studyparams.STUDY_NAME))
    if os.path.isdir(dbPath):
        celldatabase.save_hdf(celldb, dbFilename)
        print('Saved database to {}'.format(dbFilename))
    else:
        print('{} does not exist. Please create this folder.'.format(dbPath))
