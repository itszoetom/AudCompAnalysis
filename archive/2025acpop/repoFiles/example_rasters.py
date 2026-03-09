"""
Show responses to natural sounds for a few neurons.
"""

import os
import numpy as np
from jaratoolbox import celldatabase
import matplotlib.pyplot as plt
from jaratoolbox import settings
from jaratoolbox import extraplots
from jaratoolbox import ephyscore
from jaratoolbox import behavioranalysis
import sys
sys.path.append('../')
import studyparams
import studyutils

subject = 'feat018'
session = '2024-06-14'

# -- Load cell database --
dbPath = os.path.join(settings.DATABASE_PATH, studyparams.STUDY_NAME)
dbCoordsFilename = os.path.join(dbPath,f'celldb_{studyparams.STUDY_NAME}_coords.h5')
celldb = celldatabase.load_hdf(dbCoordsFilename)

# -- Select a single session --
celldb = celldb.query('subject==@subject & date==@session')
print(celldb.groupby('recordingSiteName').size())

# -- Select a sound type to analyze --
stimType = 'naturalSound'
stimVar = 'soundID'
timeRange = [-2, 6]  # In seconds

ensemble = ephyscore.CellEnsemble(celldb)
ephysData, bdata = ensemble.load(stimType)
currentStim = bdata[stimVar]

nTrials = len(currentStim)
eventOnsetTimes = ephysData['events']['stimOn'][:nTrials] # Ignore trials not in bdata 

spikeTimesFromEventOnsetAll, trialIndexForEachSpikeAll, indexLimitsEachTrialAll = \
    ensemble.eventlocked_spiketimes(eventOnsetTimes, timeRange)

possibleStim = np.unique(currentStim)
trialsEachCond = behavioranalysis.find_trials_each_type(currentStim, possibleStim)
condEachSortedTrial, sortedTrials = np.nonzero(trialsEachCond.T)
sortingInds = np.argsort(sortedTrials)

# Interesting cells from feat018: 2024-06-14
someCells = [16, 87, 283,  145, 151, 137,  186, 187, 188, 189, 197, 303,  225, 232, 373]

probeDepth = celldb.pdepth.iloc[someCells[0]]
fig = plt.figure(figsize=(12, 14))
axs = fig.subplots(int(np.ceil(len(someCells)/3)), 3, sharex=True, sharey=True)
for count, indcell in enumerate(someCells):
    sortedIndexForEachSpike = sortingInds[trialIndexForEachSpikeAll[indcell]]
    axs.flat[count].plot(spikeTimesFromEventOnsetAll[indcell], sortedIndexForEachSpike, '.k', ms=1)
    axs.flat[count].set_xlabel('Time (s)')
    axs.flat[count].set_ylabel(f'[{indcell}] Sorted trials')
    recSiteName = studyutils.simplify_site_name(celldb.iloc[indcell].recordingSiteName)
    axs.flat[count].set_title(f'{recSiteName}')
plt.suptitle(f'{subject} {session} {probeDepth}um',
             fontweight='bold')
plt.tight_layout();
plt.show()
