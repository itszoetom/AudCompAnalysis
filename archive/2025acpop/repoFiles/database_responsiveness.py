"""
Add columns to database with responsiveness to natural sounds.

It took about 7 min to process all 10486 cells.
"""

import os
import sys
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from jaratoolbox import settings
from jaratoolbox import celldatabase
from jaratoolbox import ephyscore
from jaratoolbox import behavioranalysis
from jaratoolbox import spikesanalysis
from jaratoolbox import extraplots
studyparams = __import__('2025acpop.studyparams').studyparams
from importlib import reload
reload(studyparams)

# -- Load database of cells --
dbPath = os.path.join(settings.DATABASE_PATH, studyparams.STUDY_NAME)
dbCoordsFilename = os.path.join(dbPath, f'celldb_{studyparams.STUDY_NAME}_coords.h5')
celldb = celldatabase.load_hdf(dbCoordsFilename)

# -- Define what stimulus to use --
CASE = 0
if CASE == 0:
    stimType = 'naturalSound'
    stimVar = 'soundID'
    timeRange = [-2, 6]  # In seconds
elif CASE == 1:
    stimType = 'AM'
    stimVar = 'currentFreq'
    timeRange = [-0.5, 1.5]

'''
if 0:    
    subject = 'feat015'
    sessionDate = '2024-03-20'
    probeDepth = 2413
    celldb = celldb[(celldb.subject==subject) & (celldb.date==sessionDate) & (celldb.pdepth==probeDepth)]
if 1:    
    subject = 'feat016'
    sessionDate = '2024-03-22'
    probeDepth = 3000
    celldb = celldb[(celldb.subject==subject) & (celldb.date==sessionDate) & (celldb.pdepth==probeDepth)]
'''

nCells = len(celldb)
nCategories = len(studyparams.SOUND_CATEGORIES)

periodsName = ['base', 'respOnset', 'respSustained']
allPeriods = [ [-1, 0], [0, 0.5] , [1, 4] ]
periodDuration = [x[1]-x[0] for x in allPeriods]
#meanFiringEachPeriodEachCell = np.empty((nCells, len(allPeriods)))

pValsEachCellOnset = np.empty((nCells, nCategories))
pValsEachCellSustain = np.empty((nCells, nCategories))
minPvalEachCellOnset = np.full(nCells, np.nan)
minPvalEachCellSustain = np.full(nCells, np.nan)
minPvalIndexEachCellOnset = np.full(nCells, -1)
minPvalIndexEachCellSustain = np.full(nCells, -1)

firingRateEachCellBase = np.full(nCells, np.nan)
bestFiringRateEachCellOnset = np.full(nCells, np.nan)
bestFiringRateEachCellSustain = np.full(nCells, np.nan)
bestIndexEachCellOnset = np.full(nCells, -1)
bestIndexEachCellSustain = np.full(nCells, -1)
maxFiringRateEachCellOnset = np.full(nCells, np.nan)
minFiringRateEachCellOnset = np.full(nCells, np.nan)
maxFiringRateEachCellSustain = np.full(nCells, np.nan)
minFiringRateEachCellSustain = np.full(nCells, np.nan)


indCell = -1
for indRow, dbRow in celldb.iterrows():
    indCell += 1
    #if indCell != 12: continue  # DEBUG (using only one cell)
    oneCell = ephyscore.Cell(dbRow)
    ephysData, bdata = oneCell.load(stimType)

    spikeTimes = ephysData['spikeTimes']
    eventOnsetTimes = ephysData['events']['stimOn']
    currentStim = bdata[stimVar]

    # -- Test if trials from behavior don't match ephys --
    if (len(currentStim) > len(eventOnsetTimes)) or \
       (len(currentStim) < len(eventOnsetTimes)-1):
        print(f'[{indRow}] Warning! BevahTrials ({len(currentStim)}) and ' +
              f'EphysTrials ({len(eventOnsetTimes)})')
        continue
    if len(currentStim) == len(eventOnsetTimes)-1:
        eventOnsetTimes = eventOnsetTimes[:len(currentStim)]

    possibleStim = np.unique(currentStim)
    trialsEachInstance = behavioranalysis.find_trials_each_type(currentStim, possibleStim)
    nTrialsEachInstance = trialsEachInstance.sum(axis=0)  # Not used, but in case you need it

    # -- Identify trials per category --
    nInstances = len(possibleStim)//nCategories
    trialsEachCateg = np.zeros((trialsEachInstance.shape[0], nCategories), dtype=bool)
    for indc in range(len(possibleStim)):
        trialsEachCateg[:, indc//nInstances] |= trialsEachInstance[:, indc]
    nTrialsEachCateg = trialsEachCateg.sum(axis=0)
    
    (spikeTimesFromEventOnset,trialIndexForEachSpike,indexLimitsEachTrial) = \
        spikesanalysis.eventlocked_spiketimes(spikeTimes, eventOnsetTimes, timeRange)

    meanFiringEachPeriod = np.empty(len(allPeriods))
    spikesEachTrialEachPeriod = []
    for indPeriod, period in enumerate(allPeriods):
        spikeCountMat = spikesanalysis.spiketimes_to_spikecounts(spikeTimesFromEventOnset,
                                                                 indexLimitsEachTrial, period)
        spikesEachTrial = spikeCountMat[:,0]
        spikesEachTrialEachPeriod.append(spikesEachTrial)

    firingRateEachCellBase[indCell] = spikesEachTrialEachPeriod[0].mean()/periodDuration[0]
    
    meanFiringRateBase = np.empty(nCategories)
    meanFiringRateOnset = np.empty(nCategories)
    pValEachCondOnset = np.empty(nCategories)
    meanFiringRateSustain = np.empty(nCategories)
    pValEachCondSustain = np.empty(nCategories)
    for indcond in range(nCategories):
        trialsThisCond = trialsEachCateg[:,indcond]
        firingRateBase = spikesEachTrialEachPeriod[0][trialsThisCond]/periodDuration[0]
        firingRateOnset = spikesEachTrialEachPeriod[1][trialsThisCond]/periodDuration[1]
        firingRateSustain = spikesEachTrialEachPeriod[2][trialsThisCond]/periodDuration[2]
        try:
            wStat, pValThisCond = stats.wilcoxon(firingRateBase, firingRateOnset)
        except ValueError:
            pValThisCond = 1
        pValEachCondOnset[indcond] = pValThisCond
        try:
            wStat, pValThisCond = stats.wilcoxon(firingRateBase, firingRateSustain)
        except ValueError:
            pValThisCond = 1
        pValEachCondSustain[indcond] = pValThisCond
        meanFiringRateOnset[indcond] = firingRateOnset.mean()
        meanFiringRateSustain[indcond] = firingRateSustain.mean()

    if 0:
        printopts = np.get_printoptions()
        np.set_printoptions(suppress=True,precision=4)
        print(f'Cell {indCell} meanFiringRateBase: {firingRateEachCellBase[indCell]:0.2f}')
        print(f'Cell {indCell} meanFiringRateOnset: {np.round(meanFiringRateOnset,2)}' +
              f'\t{np.round(pValEachCondOnset,4)}')
        print(f'Cell {indCell} meanFiringRateSust : {np.round(meanFiringRateSustain,2)}' +
              f'\t{np.round(pValEachCondSustain,4)}')
        print('--')
        np.set_printoptions(**printopts)
    
    indMinPvalOnset = np.argmin(pValEachCondOnset)
    minPvalIndexEachCellOnset[indCell] = indMinPvalOnset
    minPvalEachCellOnset[indCell] = pValEachCondOnset[indMinPvalOnset]
    # "BEST" indicates the maximum absolute value difference between the mean firing
    #  rate for a given modulation rate and baseline firing rate for each cell.
    indBestOnset = np.argmax(np.abs(meanFiringRateOnset-firingRateEachCellBase[indCell]))
    bestIndexEachCellOnset[indCell] = indBestOnset
    bestFiringRateEachCellOnset[indCell] = meanFiringRateOnset[indBestOnset]
    maxFiringRateEachCellOnset[indCell] = np.max(meanFiringRateOnset)
    minFiringRateEachCellOnset[indCell] = np.min(meanFiringRateOnset)

    indMinPvalSustain = np.argmin(pValEachCondSustain)
    minPvalIndexEachCellSustain[indCell] = indMinPvalSustain
    minPvalEachCellSustain[indCell] = pValEachCondSustain[indMinPvalSustain]
    indBestSustain = np.argmax(np.abs(meanFiringRateSustain-firingRateEachCellBase[indCell]))
    bestIndexEachCellSustain[indCell] = indBestSustain
    bestFiringRateEachCellSustain[indCell] = meanFiringRateSustain[indBestSustain]
    maxFiringRateEachCellSustain[indCell] = np.max(meanFiringRateSustain)
    minFiringRateEachCellSustain[indCell] = np.min(meanFiringRateSustain)

    if indCell % 50 == 0:
        print(f'{indCell}/{nCells}')

celldb['nsMinPvalOnset'] = minPvalEachCellOnset
celldb['nsIndexMinPvalOnset'] = minPvalIndexEachCellOnset
celldb['nsMinPvalSustain'] = minPvalEachCellSustain
celldb['nsIndexMinPvalOnset'] = minPvalIndexEachCellSustain

celldb['nsFiringRateBaseline'] = firingRateEachCellBase
celldb['nsFiringRateBestOnset'] = bestFiringRateEachCellOnset
celldb['nsIndexBestOnset'] = bestIndexEachCellOnset
celldb['nsFiringRateBestSustain'] = bestFiringRateEachCellSustain
celldb['nsIndexBestSustain'] = bestIndexEachCellSustain

celldb['nsFiringRateMaxOnset'] = maxFiringRateEachCellOnset
celldb['nsFiringRateMinOnset'] = minFiringRateEachCellOnset
celldb['nsFiringRateMaxSustain'] = maxFiringRateEachCellSustain
celldb['nsFiringRateMinSustain'] = minFiringRateEachCellSustain

# -- Show the comparison of p-values for onset and sustained responses --
if 0:
    alpha = 0.01
    onset = celldb.nsMinPvalOnset<(alpha/nCategories)
    sust = celldb.nsMinPvalSustain<(alpha/nCategories)
    comb = np.vstack((np.arange(nCells),onset,sust)).T
    print(comb)

dbResponsiveFilename = os.path.join(dbPath, f'celldb_{studyparams.STUDY_NAME}_responsive.h5')
celldatabase.save_hdf(celldb, dbResponsiveFilename)
