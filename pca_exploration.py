#%% Imports
import os
import studyparams
import numpy as np
import pandas as pd
from jaratoolbox import celldatabase
from jaratoolbox import settings
from jaratoolbox import spikesanalysis
from jaratoolbox import ephyscore
from jaratoolbox import behavioranalysis
from scipy import stats
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from jaratoolbox import extraplots

# TODO: Filter dataframe by grouped brain region to load the ephys and bdata. Calculate the population spike counts as
#  before, but grouping all stimuli together. Feed that into sklearn.decomp.PCA. single_values_ gives eigenvalues and
#  components_ gives eigenvector. Make scree plot from values (eigenvalue is y-axis, component number is x-axis)

#%% Loading data and setting time bins
figSavePath = "/Users/Matt/Desktop/Research/Murray/data/images"
fontSizeLabels = 10

databaseDir = os.path.join(settings.DATABASE_PATH, studyparams.STUDY_NAME)  # Note: Change these to use PathLib instead
subject = "feat007"
dbPath = os.path.join(databaseDir, f'{subject}_paspeech_speech_pval.h5')
mouseDB = celldatabase.load_hdf(dbPath)

simpleSiteNames = mouseDB["recordingSiteName"].str.split(',').apply(lambda x: x[0])
mouseDB["recordingSiteName"] = simpleSiteNames


# recordingDate = mouseDB.date.unique()[0]
recordingDate = "2022-03-10"
# targetSiteName = "Primary auditory area"
# targetSiteName = "Dorsal auditory area"
targetSiteNames = ["Primary auditory area", "Dorsal auditory area",
                   "Ventral auditory area"]
leastCellsArea = 10000
for testName in targetSiteNames:
    celldbSubset = mouseDB[(mouseDB.date == recordingDate) & (mouseDB.recordingSiteName == testName)]
    leastCellsArea = np.min([leastCellsArea, celldbSubset.shape[0]])

for targetSiteName in targetSiteNames:
    celldbSubset = mouseDB[(mouseDB.date == recordingDate) & (mouseDB.recordingSiteName == targetSiteName)]

    ensemble = ephyscore.CellEnsemble(celldbSubset)
    ephysDataSpeech, bdataSpeech = ensemble.load("FTVOTBorders")
    # ephysDataAM, bdataAM = ensemble.load("AM")
    # ephysDataPT, bdataPT = ensemble.load("pureTones")

    periodsNameSpeech = ['base200', 'respOnset', 'respSustained']
    allPeriodsSpeech = [[-0.2, 0], [0, 0.12], [0.12, 0.24]]  # try with shorter period for onset response.
    timeRangeSpeech = [allPeriodsSpeech[0][0], allPeriodsSpeech[-1][-1]]

    # periodsNameAM = ['respOnset', 'respSustained']
    # allPeriodsAM = [[0, 0.1], [0.1, 0.5]]

    # Speech start
    spikeTimesSpeech = ephysDataSpeech['spikeTimes']
    eventOnsetTimesSpeech = ephysDataSpeech['events']['stimOn']

    spikeTimesFromEventOnsetAll, trialIndexForEachSpikeAll, indexLimitsEachTrialAll = \
        ensemble.eventlocked_spiketimes(eventOnsetTimesSpeech, timeRangeSpeech)

    FTParamsEachTrial = bdataSpeech['targetFTpercent']
    possibleFTParams = np.unique(FTParamsEachTrial)
    nTrials = len(FTParamsEachTrial)

    VOTParamsEachTrial = bdataSpeech['targetVOTpercent']
    possibleVOTParams = np.unique(VOTParamsEachTrial)

    trialsEachCond = behavioranalysis.find_trials_each_combination(VOTParamsEachTrial, possibleVOTParams, FTParamsEachTrial,
                                                                   possibleFTParams)

    binSize = 0.005  # 5 ms spike time bins
    binEdges = np.arange(allPeriodsSpeech[1][0], allPeriodsSpeech[1][1], binSize)  # Using the onset time, so [0, 0.12]
    spikeCounts = ensemble.spiketimes_to_spikecounts(binEdges)  # (nCells, nTrials, nBins)
    nCells = spikeCounts.shape[0]

    # avgFRTrial = spikeCounts.mean(axis=1)
    # sumEvokedFR = avgFRTrial.sum(axis=1)
    sumEvokedFR = spikeCounts.sum(axis=2)  # Sum across the bins so now dims are (nCells, nTrials)
    spikesPerSecEvoked = sumEvokedFR/(allPeriodsSpeech[1][1] - allPeriodsSpeech[1][0])  # Divide by time width to get per sec

    # PCA Speech Start

    # Non-normalized PCA
    # pcaSpeech = PCA()
    # PCSpeech = pcaSpeech.fit(spikesPerSecEvoked)
    #
    # features = range(pcaSpeech.n_components_)
    # speechEigen = pcaSpeech.explained_variance_ratio_
    #
    # plt.figure(figsize=(10, 6))
    # plt.bar(features, speechEigen, color='black')
    # plt.xlabel('PCA features')
    # plt.ylabel('variance %')
    # plt.xticks(features)
    # plt.title("Scree Plot, non-normalized, primary aud area")
    # plt.show()


    # PCA with firing rates normalzied by subtracting trial averages
    trialMeans = spikesPerSecEvoked.mean(axis=1)
    spikesPerSecEvokedNormalized = spikesPerSecEvoked.T - trialMeans  # (nTrials, nCells)

    # Now to subset to have an equal number of cells for each area
    if spikesPerSecEvokedNormalized.shape[1] > leastCellsArea:
        subsetIndex = np.random.choice(spikesPerSecEvokedNormalized.shape[1],
                                       leastCellsArea, replace=False)
        smallestCommonSubset = spikesPerSecEvokedNormalized[:, subsetIndex]
    else:
        smallestCommonSubset = spikesPerSecEvokedNormalized

    pcaSpeech = PCA()
    PCSpeech = pcaSpeech.fit(smallestCommonSubset)

    features = range(pcaSpeech.n_components_)
    speechEigen = pcaSpeech.explained_variance_ratio_

    particRatio = ((np.sum(speechEigen))**2) / np.sum(speechEigen**2)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.bar(features, speechEigen, color='black')
    plt.xlabel('PCA features')
    plt.ylabel('variance %')
    plt.xticks(features)
    plt.title(f"Scree Plot, normalized, {targetSiteName}")
    plt.text(0.87, 0.9, f"Participation Ratio =\n {particRatio:0.4}",
             horizontalalignment='center', verticalalignment='center',
             fontsize=12, transform=ax.transAxes)
    figSaveName = figSavePath + f"/normalized_Scree_{targetSiteName}"
    extraplots.save_figure(figSaveName, 'svg', [10, 6])
    plt.show()

#%%
    # Scatter plot with eigenvalues
    speechComponents = pcaSpeech.components_
    speechVar = pcaSpeech.explained_variance_

    transformedSubset = pcaSpeech.transform(smallestCommonSubset)

    # plt.scatter(smallestCommonSubset[:, 10], smallestCommonSubset[:, 8], alpha=0.3)
    plt.scatter(transformedSubset[:, 10], transformedSubset[:, 8], alpha=0.3)

    plt.ylabel('PC 1', fontsize=fontSizeLabels, fontweight='bold')
    plt.xlabel('PC 2', fontsize=fontSizeLabels, fontweight='bold')

    extraplots.save_figure("firing_rates_pca", 'svg', [6.4, 4.8],
                           "/Users/Matt/Desktop/Research/Murray/data/images/proposal_images/")
    plt.show()
