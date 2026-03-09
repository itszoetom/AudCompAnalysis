# Plots Scree Plots and Calculates Participation Ratio for each mouse - date combination.
# 3x3 plots with each brain area - sound type combination

#%% Imports
import os
from .. import studyparams
import numpy as np
from jaratoolbox import celldatabase, settings, ephyscore, behavioranalysis, extraplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# TODO: Filter dataframe by grouped brain region to load the ephys and bdata. Calculate the population spike counts as
#  before, but grouping all stimuli together. Feed that into sklearn.decomp.PCA. single_values_ gives eigenvalues and
#  components_ gives eigenvector. Make scree plot from values (eigenvalue is y-axis, component number is x-axis)

#%% Constants
databaseDir = os.path.join(settings.DATABASE_PATH, studyparams.STUDY_NAME)

subject_list = ['feat004', 'feat005', 'feat006', 'feat007', 'feat008', 'feat009', 'feat010']
#fullDbPath = f'fulldb_speech_tuning.h5'
#fullPath = os.path.join(databaseDir, fullDbPath)
#fullDb = celldatabase.load_hdf(fullPath)
dbPathSpeech = f'feat007_paspeech_speech_pval.h5'
dbPathAM = f'feat007_paspeech_am_pval.h5'
dbPathPT = f'feat007_paspeech_tones_pval.h5'

# %% Loading data and initializing figure
fullPathSpeech = os.path.join(databaseDir, dbPathSpeech)
mouseDBSpeech = celldatabase.load_hdf(fullPathSpeech)
simpleSiteNames = mouseDBSpeech["recordingSiteName"].str.split(',').apply(lambda x: x[0])
mouseDBSpeech["recordingSiteName"] = simpleSiteNames

fullPathAM = os.path.join(databaseDir, dbPathAM)
mouseDBAM = celldatabase.load_hdf(fullPathAM)
simpleSiteNames = mouseDBAM["recordingSiteName"].str.split(',').apply(lambda x: x[0])
mouseDBAM["recordingSiteName"] = simpleSiteNames

fullPathPT = os.path.join(databaseDir, dbPathPT)
mouseDBPT = celldatabase.load_hdf(fullPathPT)
simpleSiteNames = mouseDBPT["recordingSiteName"].str.split(',').apply(lambda x: x[0])
mouseDBPT["recordingSiteName"] = simpleSiteNames

celldbSubsetSpeech = mouseDBSpeech[(mouseDBSpeech.date == '2022-03-10') & (mouseDBSpeech.recordingSiteName == 'Primary auditory area')]
celldbSubsetAM = mouseDBAM[(mouseDBAM.date == '2022-03-10') & (mouseDBAM.recordingSiteName == 'Primary auditory area')]
celldbSubsetPT = mouseDBPT[(mouseDBPT.date == '2022-03-10') & (mouseDBPT.recordingSiteName == 'Primary auditory area')]


for subject in subject_list:
    dbPathSpeech = f'{subject}_paspeech_speech_pval.h5'
    dbPathAM = f'{subject}_paspeech_am_pval.h5'
    dbPathPT = f'{subject}_paspeech_tones_pval.h5'

    for date in studyparams.recordingDate_list[subject]:
        #%% Loading data and initializing figure
        fullPathSpeech = os.path.join(databaseDir, dbPathSpeech)
        mouseDBSpeech = celldatabase.load_hdf(fullPathSpeech)
        simpleSiteNames = mouseDBSpeech["recordingSiteName"].str.split(',').apply(lambda x: x[0])
        mouseDBSpeech["recordingSiteName"] = simpleSiteNames

        fullPathAM = os.path.join(databaseDir, dbPathAM)
        mouseDBAM = celldatabase.load_hdf(fullPathAM)
        simpleSiteNames = mouseDBAM["recordingSiteName"].str.split(',').apply(lambda x: x[0])
        mouseDBAM["recordingSiteName"] = simpleSiteNames

        fullPathPT = os.path.join(databaseDir, dbPathPT)
        mouseDBPT = celldatabase.load_hdf(fullPathPT)
        simpleSiteNames = mouseDBPT["recordingSiteName"].str.split(',').apply(lambda x: x[0])
        mouseDBPT["recordingSiteName"] = simpleSiteNames

        fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharey='row')
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        fig.suptitle(f"Scree Plots for Auditory Areas and Sound Types for Mouse {subject} on {date}", fontsize=16)

        #%% Process Each Target Site
        for i, targetSiteName in enumerate(studyparams.targetSiteName):
            celldbSubsetSpeech = mouseDBSpeech[(mouseDBSpeech.date == date) & (mouseDBSpeech.recordingSiteName == targetSiteName)]
            celldbSubsetAM = mouseDBAM[(mouseDBAM.date == date) & (mouseDBAM.recordingSiteName == targetSiteName)]
            celldbSubsetPT = mouseDBPT[(mouseDBPT.date == date) & (mouseDBPT.recordingSiteName == targetSiteName)]

            # Check if all subsets are empty
            if celldbSubsetSpeech.empty and celldbSubsetAM.empty and celldbSubsetPT.empty:
                print(f"No data in {targetSiteName} on {date} for Speech, AM, and PT.")
                continue

            if not celldbSubsetSpeech.empty:
                # Load Speech data for the mouse
                ensembleSpeech = ephyscore.CellEnsemble(celldbSubsetSpeech)
                ephysDataSpeech, bdataSpeech = ensembleSpeech.load("FTVOTBorders")
            else:
                print(f"No data in {targetSiteName} on {date} for Speech.")

            if not celldbSubsetAM.empty:
                # Load AM data for the mouse
                ensembleAM = ephyscore.CellEnsemble(celldbSubsetAM)
                ephysDataAM, bdataAM = ensembleAM.load("AM")
            else:
                print(f"No data in {targetSiteName} on {date} for AM.")

            if not celldbSubsetPT.empty:
                # Load PT data for the mouse
                ensemblePT = ephyscore.CellEnsemble(celldbSubsetPT)
                ephysDataPT, bdataPT = ensemblePT.load("pureTones")
            else:
                print(f"No data in {targetSiteName} on {date} for PT.")

        #%% AM and PT Start
            for sessionType in ["AM", 'pureTones']:
                if sessionType == ('pureTones'):
                    # Gets spike times and event onset times (i.e. when the sounds were presented or the stim turns on)
                    spikeTimesPT = ephysDataPT['spikeTimes']
                    nTrials = len(bdataPT['currentFreq'])
                    eventOnsetTimesPT = ephysDataPT['events']['stimOn'][:nTrials]

                    # Lines up event times and spike times
                    spikeTimesFromEventOnsetPT, trialIndexForEachSpikePT, indexLimitsEachTrialPT = \
                        ensemblePT.eventlocked_spiketimes(eventOnsetTimesPT,[studyparams.evoked_start, studyparams.evoked_end])

                    spikeCountMatPT = ensemblePT.spiketimes_to_spikecounts(studyparams.binEdges)
                    spikeCountMatPT = spikeCountMatPT.sum(axis=2)
                    spike_ratePT = spikeCountMatPT / (studyparams.evoked_end - studyparams.evoked_start)

                    spikeRatePT = spike_ratePT.T

                    # Perform PCA on the data:
                    pcaPT = PCA()
                    pcaPT.fit((spikeRatePT - np.mean(spikeRatePT, axis=0)))

                if sessionType == ('AM'):
                    spikeTimesAM = ephysDataAM['spikeTimes']
                    nTrials = len(bdataAM['currentFreq'])
                    eventOnsetTimesAM = ephysDataAM['events']['stimOn'][:nTrials]

                    spikeTimesFromEventOnsetAM, trialIndexForEachSpikeAM, indexLimitsEachTrialAM = \
                    ensembleAM.eventlocked_spiketimes(eventOnsetTimesAM, [studyparams.evoked_start, studyparams.evoked_end])

                    spikeCountMatAM = ensembleAM.spiketimes_to_spikecounts(studyparams.binEdges)
                    spikeCountMatAM = spikeCountMatAM.sum(axis=2)
                    spike_rateAM = spikeCountMatAM / (studyparams.evoked_end - studyparams.evoked_start)

                    spikeRateAM = spike_rateAM.T

                    pcaAM = PCA()
                    pcaAM.fit((spikeRateAM - np.mean(spikeRateAM, axis=0)))

        #%% Speech start
            spikeTimesSpeech = ephysDataSpeech['spikeTimes']
            eventOnsetTimesSpeech = ephysDataSpeech['events']['stimOn']

            spikeTimesFromEventOnsetAll, trialIndexForEachSpikeAll, indexLimitsEachTrialAll = \
                ensembleSpeech.eventlocked_spiketimes(eventOnsetTimesSpeech, studyparams.timeRangeSpeech)

            FTParamsEachTrial = bdataSpeech['targetFTpercent']
            possibleFTParams = np.unique(FTParamsEachTrial)
            nTrials = len(FTParamsEachTrial)

            VOTParamsEachTrial = bdataSpeech['targetVOTpercent']
            possibleVOTParams = np.unique(VOTParamsEachTrial)

            trialsEachCond = behavioranalysis.find_trials_each_combination(VOTParamsEachTrial, possibleVOTParams, FTParamsEachTrial,
                                                                           possibleFTParams)

            spikeCounts = ensembleSpeech.spiketimes_to_spikecounts(studyparams.binEdges)  # (nCells, nTrials, nBins)
            nCells = spikeCounts.shape[0]

            # avgFRTrial = spikeCounts.mean(axis=1)
            # sumEvokedFR = avgFRTrial.sum(axis=1)
            sumEvokedFR = spikeCounts.sum(axis=2)  # Sum across the bins so now dims are (nCells, nTrials)
            spikesPerSecEvoked = sumEvokedFR/(studyparams.allPeriodsSpeech[1][1] - studyparams.allPeriodsSpeech[1][0])  # Divide by time width to get per sec

            # PCA with firing rates normalzied by subtracting trial averages
            trialMeans = spikesPerSecEvoked.mean(axis=1)
            spikesPerSecEvokedNormalized = spikesPerSecEvoked.T - trialMeans  # (nTrials, nCells)

            pcaSpeech = PCA()
            pcaSpeech.fit((spikesPerSecEvokedNormalized - np.mean(spikesPerSecEvokedNormalized, axis=0)))

            # Now to subset to have an equal number of cells for each area
            if spikesPerSecEvokedNormalized.shape[1] > studyparams.leastCellsArea:
                subsetIndex = np.random.choice(spikesPerSecEvokedNormalized.shape[1],
                                               studyparams.leastCellsArea, replace=False)
                smallestCommonSubset = spikesPerSecEvokedNormalized[:, subsetIndex]
            else:
                smallestCommonSubset = spikesPerSecEvokedNormalized

            pcaSpeech = PCA()
            pcaSpeech = pcaSpeech.fit(smallestCommonSubset)

        #%% Plotting
            speechFeatures = range(pcaSpeech.n_components_)
            featuresPT = range(pcaPT.n_components_)
            featuresAM = range(pcaAM.n_components_)

            speechEigen = pcaSpeech.explained_variance_ratio_
            particRatioSpeech = ((np.sum(speechEigen)) ** 2) / np.sum(speechEigen ** 2)

            eigenPT = pcaPT.explained_variance_ratio_
            particRatioPT = ((np.sum(eigenPT)) ** 2) / np.sum(eigenPT ** 2)

            eigenAM = pcaAM.explained_variance_ratio_
            particRatioAM = ((np.sum(eigenAM)) ** 2) / np.sum(eigenAM ** 2)

            # Set consistent y-axis limit
            y_max = max(max(speechEigen), max(eigenPT), max(eigenAM)) * 1.1

            # Define x-axis limit
            x_min = 0
            x_max = 13

            # Plotting Speech
            ax = axes[i, 0]
            ax.bar(speechFeatures, speechEigen, color='black')
            ax.set_xlabel('PCA features', fontsize=12)
            ax.set_ylabel('variance %', fontsize=12)
            ax.set_xticks(range(x_max))
            ax.set_xlim(x_min, x_max)
            ax.set_title(f"{targetSiteName} - Speech", fontsize=14)
            ax.text(0.87, 0.85, f"Participation Ratio = {particRatioSpeech:.3f}",
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, transform=ax.transAxes)
            ax.set_ylim(0, y_max)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

            # Plotting PT
            ax = axes[i, 1]
            ax.bar(featuresPT, eigenPT, color='black')
            ax.set_xlabel('PCA features', fontsize=12)
            ax.set_ylabel('variance %', fontsize=12)
            ax.set_xticks(range(x_max))
            ax.set_xlim(x_min, x_max)
            ax.set_title(f"{targetSiteName} - PT", fontsize=14)
            ax.text(0.87, 0.85, f"Participation Ratio = {particRatioPT:.3f}",
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, transform=ax.transAxes)
            ax.set_ylim(0, y_max)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

            # Plotting AM
            ax = axes[i, 2]
            ax.bar(featuresAM, eigenAM, color='black')
            ax.set_xlabel('PCA features', fontsize=12)
            ax.set_ylabel('variance %', fontsize=12)
            ax.set_xticks(range(x_max))
            ax.set_xlim(x_min, x_max)
            ax.set_title(f"{targetSiteName} - AM", fontsize=14)
            ax.text(0.87, 0.85, f"Participation Ratio = {particRatioAM:.3f}",
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, transform=ax.transAxes)
            ax.set_ylim(0, y_max)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

            extraplots.save_figure(f"scree_plots_{subject}_{date}_pca", 'png', [12.4, 10.8],
                                   "/Users/zoetomlinson/Desktop/NeuroAI/Figures/Singular Mouse Plots/"
                                   "date_pca_visualizations")
        plt.show()