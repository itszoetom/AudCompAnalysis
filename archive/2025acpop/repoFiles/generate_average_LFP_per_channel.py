"""
Load LFP and calculate average LFP for each channel.
This is used to get a better estimate of the location of electrodes.

USAGE:
python generate_average_LFP_per_channel.py [subject]

If no subject is specified, it will use a default subject+session.
"""

import os
import sys
import numpy as np
from jaratoolbox import settings
from jaratoolbox import celldatabase
from jaratoolbox import loadneuropix
from jaratoolbox import loadbehavior
from jaratoolbox import behavioranalysis
from matplotlib import pyplot as plt


SAVE_DATA = True

if len(sys.argv)>1:
    subject = sys.argv[1]
    if len(sys.argv)>2:
        sessionDate = sys.argv[2]
    else:
        sessionDate = None  # Processed all dates
    DEBUG = False
else:
    subject = 'feat017'
    sessionDate = '2024-04-12'
    DEBUG = True
    
sessionType = 'pureTones' #'naturalSound'  #'pureTones'
siteInd = 0  # Use the first site (assuming a single site per penetration)
dataStream = 'Neuropix-PXI-100.1'
recordNode = 'Record Node 101'

timeRange = [-0.1, 0.3]
baselineRange = [timeRange[0], 0]

# -- Find corresponding ephys and behavior session in inforec --
inforecFile = os.path.join(settings.INFOREC_PATH, f'{subject}_inforec.py')
inforec = celldatabase.read_inforec(inforecFile)
behavSession = None  # In case no session is found
found = False
if DEBUG:
    for experiment in inforec.experiments:
        if experiment.date == sessionDate:
            experimentsToProcess = [experiment]
else:
    experimentsToProcess = inforec.experiments

for experiment in experimentsToProcess:
    sessionDate = experiment.date
    for oneSession in experiment.sites[siteInd].sessions:
        if oneSession.sessiontype == sessionType:
            sessionTime = oneSession.timestamp
            behavSession = sessionDate.replace('-','')+oneSession.behavsuffix
            paradigm = oneSession.paradigm

            rawDataPath = os.path.join(settings.RAW_NEUROPIX_PATH, subject,
                                       f'{sessionDate}_{sessionTime}')
            xmlfile = os.path.join(rawDataPath, recordNode, 'settings.xml')

            # -- Load behavior data --
            behavFile = loadbehavior.path_to_behavior_data(subject, paradigm, behavSession)
            print(f'Loading behavior data: {behavFile} ...')
            bdata = loadbehavior.BehaviorData(behavFile)
            if 'currentFreq' in bdata:
                stimEachTrial = bdata['currentFreq']
            else:
                stimEachTrial = bdata['soundID']
            nTrials = len(stimEachTrial)
            possibleStim = np.unique(stimEachTrial)
            nStim = len(possibleStim)

            # -- Load raw data --
            print(f'Loading raw data from {rawDataPath} ...')
            contData = loadneuropix.Continuous(rawDataPath, dataStream)
            rawdata = contData.data
            sampleRate = contData.sampleRate
            nChannels = contData.nChannels
            bitVolts = contData.bitVolts
            probeMap = loadneuropix.ProbeMap(xmlfile)

            # -- Load events from ephys data --
            print('Loading events data...')
            events = loadneuropix.RawEvents(rawDataPath, dataStream)
            eventOnsetTimes = events.get_onset_times()  # In samples
            eventOnsetTimes -= events.firstSample  # Convert to 0-based

            # If the ephys data has one more event than bdata, delete the last ephys trial
            if len(stimEachTrial) == len(eventOnsetTimes)-1:
                eventOnsetTimes = eventOnsetTimes[:len(stimEachTrial)]
            assert len(stimEachTrial) == len(eventOnsetTimes), \
                "Number of trials in behavior and ephys do not match"

            # -- Calculate event-locked LFP --
            sampleRange = [int(timeRange[0]*sampleRate), int(timeRange[1]*sampleRate)]
            timeVec = np.arange(sampleRange[0], sampleRange[1])/sampleRate
            nSamplesToExtract = sampleRange[1] - sampleRange[0]
            eventlockedLFP = np.empty((nTrials, nSamplesToExtract, nChannels), dtype=np.int16)
            # -- Calculate event-locked LFP --
            print('Calculating eventlockedLFP...')
            for indt, evSample in enumerate(eventOnsetTimes):
                eventlockedLFP[indt, :, :] = rawdata[evSample+sampleRange[0]:evSample+sampleRange[1], :]

            # -- Calculate average LFP for each stimulus condition --
            print('Calculating average LFP for each stimulus...')
            trialsEachCond = behavioranalysis.find_trials_each_type(stimEachTrial, possibleStim)
            avgLFP = np.empty((nStim, nSamplesToExtract, nChannels))
            labelsAvgLFP = ['Stimulus', 'Time', 'Channel']
            for indstim, stimFreq in enumerate(possibleStim):
                avgLFP[indstim, :, :] = np.mean(eventlockedLFP[trialsEachCond[:, indstim], :, :], axis=0)
            avgLFP *= bitVolts  # Convert to uV

            # -- Sort channels according to depth --
            # Note that in NP1.0 channel 191 (starting from 0) does not get a position in the XML file.
            chanOrder = np.argsort(probeMap.channelID)
            yPosNewOrder = probeMap.ypos[chanOrder]
            sortingOrder = np.argsort(yPosNewOrder)
            sortedChannels = probeMap.channelID[sortingOrder]

            # -- Subtract baseline --
            baselineSamples = np.logical_and(timeVec >= baselineRange[0], timeVec < baselineRange[1])
            baseline = np.mean(avgLFP[:, baselineSamples, :], axis=1)
            avgLFPnobase = avgLFP - baseline[:, np.newaxis, :]

            sortedAvgLFPnobase = avgLFPnobase[:, :, sortedChannels]

            varEachStim = np.var(sortedAvgLFPnobase, axis=(1,2))
            maxVarStim = np.argmax(varEachStim)

            colorLimit = max(abs(avgLFPnobase.max()), abs(avgLFPnobase.min()))
            clim = [-colorLimit, colorLimit]

            outputDir = os.path.join(settings.EPHYS_NEUROPIX_PATH, f'{subject}_processed')
            if not os.path.exists(outputDir):
                os.makedirs(outputDir)
            outputFile = os.path.join(outputDir, f'{subject}_{sessionDate}_{sessionTime}_avgLFP.npz')
            if SAVE_DATA:
                print('Saving average LFP data to:', os.path.join(outputDir, outputFile))
                np.savez(outputFile, avgLFPnobase=avgLFPnobase, timeVec=timeVec, baseline=baseline,
                         possibleStim=possibleStim, sampleRate=sampleRate, bitVolts=bitVolts,
                         channelID=probeMap.channelID, electrodeYpos=probeMap.ypos,
                         sortedChannels=sortedChannels, maxVarStim=maxVarStim)

            # -- Plot baseline-subtracted average LFP for each stimulus condition --
            if 1:
                print('Plotting responses for all stimuli...')
                plt.clf()
                for indStim in range(len(possibleStim)):
                    plt.subplot(4, 4, indStim+1)
                    plt.imshow(sortedAvgLFPnobase[indStim, :, :].T, aspect='auto', origin='lower',
                               extent=[timeVec[0], timeVec[-1], 0, nChannels], clim=clim)
                    plt.colorbar()
                    plt.title(f'{possibleStim[indStim]} Hz')
                plt.suptitle(f'Avg LFP (baseline-subtracted) for {subject} {sessionDate}_{sessionTime}')
                plt.axis('tight')
                plt.show()

            # -- Plot baseline-subtracted average LFP for best stimulus --
            if 0:
                print('Plotting responses for stimulus with max response...')
                plt.clf()
                ax1 = plt.gca()
                imExtent = [timeVec[0], timeVec[-1], probeMap.ypos.min(), probeMap.ypos.max()]
                plt.imshow(sortedAvgLFPnobase[maxVarStim, :, :].T, aspect='auto', origin='lower',
                           extent=imExtent, clim=clim)
                ax1.set_ylabel('Distance from tip (um)')
                ax2 = ax1.twinx()
                ax2.set_ylim([sortedChannels[0], sortedChannels[-1]])
                ax2.set_ylabel('Channel (sorted by depth)', rotation=270, labelpad=16)
                plt.colorbar(pad=0.08)
                plt.title(f'{possibleStim[maxVarStim]} Hz')
                plt.suptitle(f'Avg LFP (baseline-subtracted) for {subject} {sessionDate}_{sessionTime}')
                plt.axis('tight')
                plt.show()
                plt.pause(0.01)
            break  # Skip to the next experiment once the right session is found
    if DEBUG:
        print('Debug mode: stopping after the first experiment.')
        break
