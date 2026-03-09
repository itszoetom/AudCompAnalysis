"""
Show the average LFP and the cell locations on the atlas.
This script requires the Allen SDK to be installed in the environment.

USAGE:
python report_lfp_and_histology.py [subject]

If no subject is specified, it will use a default subject.
"""

import os
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from jaratoolbox import settings
from jaratoolbox import celldatabase
from jaratoolbox import histologyanalysis as ha  # This requires the Allen SDK
from jaratoolbox import extraplots
import studyparams

SAVE = True

if len(sys.argv)>1:
    subject = sys.argv[1]
else:
    subject = 'feat017'

outputDir = os.path.join(settings.FIGURES_DATA_PATH, studyparams.STUDY_NAME, 'lfp')
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

pageSize = [20, 12]  # Size of the figure in inches    
sessionsPerPage = 4
processedDir = os.path.join(settings.EPHYS_NEUROPIX_PATH, f'{subject}_processed')

responseRange = [0.02, 0.03]  # Range to estimate response on each channel
earlyResponseRange = [0.015, 0.02]  # Range to estimate response on each channel

# -- Get a list of all files in processedDir that end in avgLFP.npz --
allFiles = os.listdir(processedDir)
allFiles = [f for f in allFiles if f.endswith('avgLFP.npz')]
allFiles = sorted(allFiles)  # Sort the files to ensure consistent order

# -- Extract session date and time from the filenames --
sessionDates = []
sessionTimes = []
for oneFile in allFiles:
    parts = oneFile.split('_')
    sessionDates.append(parts[1])
    sessionTimes.append(parts[2].replace('.npz', ''))

# -- Load the database of cells --    
dbPath = os.path.join(settings.DATABASE_PATH, studyparams.STUDY_NAME)
dbCoordsFilename = os.path.join(dbPath,f'celldb_{studyparams.STUDY_NAME}_coords.h5')
celldb = celldatabase.load_hdf(dbCoordsFilename)

if subject is not None:
    celldb = celldb.query('subject==@subject')

# -- Load mouse atlas --
aa = ha.AllenAverageCoronalAtlas()

# -- Plot the average LFP and slice for each session --
fig = plt.gcf()  #figure(figsize=pageSize)
nPages = int(np.ceil(len(sessionDates) / sessionsPerPage))
for indpage in range(nPages):
    fig.clf()
    # -- Main gridspec --
    gsMain = gridspec.GridSpec(sessionsPerPage, 3, width_ratios=[0.4, 0.25, 0.35])
    gsMain.update(left=0.05, right=0.95, top=0.94, bottom=0.05, wspace=0.1) #, hspace=0.3)

    indSessionsToPlot = range(indpage*sessionsPerPage, (indpage+1)*sessionsPerPage)
    for indplot, inds in enumerate(indSessionsToPlot):
        if inds >= len(sessionDates):
            break
        sessionDate = sessionDates[inds]
        sessionTime = sessionTimes[inds]
        lfpFile = os.path.join(processedDir, f'{subject}_{sessionDate}_{sessionTime}_avgLFP.npz')
        lfpData = np.load(lfpFile)

        avgLFPnobase = lfpData['avgLFPnobase']
        timeVec = lfpData['timeVec']
        possibleStim = lfpData['possibleStim']
        maxVarStim = lfpData['maxVarStim']
        electrodeYpos = lfpData['electrodeYpos']
        sortedChannels = lfpData['sortedChannels']

        sortedAvgLFPnobase = avgLFPnobase[:, :, sortedChannels]

        colorLimit = max(abs(avgLFPnobase.max()), abs(avgLFPnobase.min()))
        clim = [-colorLimit, colorLimit]

        # -- Plot the average LFP for the best stimulus --
        stimIndToUse = maxVarStim # maxVarStim # 7
        axLFP = plt.subplot(gsMain[indplot,0])
        imExtent = [1e3*timeVec[0], 1e3*timeVec[-1], electrodeYpos.min(), electrodeYpos.max()]
        axLFP.imshow(sortedAvgLFPnobase[stimIndToUse, :, :].T, aspect='auto', origin='lower',
                   extent=imExtent, clim=clim)
        axTitle = f'{sessionDate} ({possibleStim[stimIndToUse]:0.0f} Hz)'
        axLFP.set_ylabel(f'{axTitle}\nDistance from tip (um)')
        axLFPextraY = axLFP.twinx()
        axLFPextraY.set_ylim([sortedChannels[0], sortedChannels[-1]])
        #axLFPextraY.axvline(0, color='w', linestyle='-', alpha=0.5)
        axLFPextraY.axvline(1e3*0.02, color='k', linestyle='-', alpha=0.1)
        axLFPextraY.set_xlim([1e3*timeVec[0], 1e3*0.2])
        axLFPextraY.set_ylabel('Channel', rotation=270, labelpad=16)
        #plt.colorbar(pad=0.08)
        if (indplot == sessionsPerPage-1) or (indplot == len(sessionDates)%sessionsPerPage-1):
            axLFP.set_xlabel('Time from stim (ms)')
        else:
            #axLFP.set_xticklabels([])
            pass
            
        # -- Plot atlas slice for the session --
        axAtlas = plt.subplot(gsMain[indplot,1])
        plt.sca(axAtlas)
        celldbSubset = celldb.query('date==@sessionDate')
        aa.reset_points()
        aa.add_points_from_db(celldbSubset)
        aa.show_single_slice(areas=['AUDp','AUDv','AUDd','AUDpo'])

        # -- Plot histogram of best channels and avg response on each channel --
        axHist = plt.subplot(gsMain[indplot,2])
        axHist.hist(celldbSubset.bestChannel, bins=20, fc='firebrick', ec='orangered')
        axHist.set_xlim([sortedChannels[0], sortedChannels[-1]])
        axHist.set_ylabel('N cells', color='firebrick')
        if (indplot == sessionsPerPage-1) or (indplot == len(sessionDates)%sessionsPerPage-1):
            axHist.set_xlabel('Best channel')
        respRangeSamples = np.logical_and(timeVec>=responseRange[0], timeVec<=responseRange[1])
        earlyRespRangeSamples = np.logical_and(timeVec>=earlyResponseRange[0],
                                               timeVec<=earlyResponseRange[1])
        avgResponse = sortedAvgLFPnobase[stimIndToUse,respRangeSamples].mean(axis=0)
        avgEarlyResponse = sortedAvgLFPnobase[stimIndToUse,earlyRespRangeSamples].mean(axis=0)
        axHist3 = axHist.twinx()
        axHist3.plot(sortedChannels, avgEarlyResponse, color='c', lw=2)
        axHist3.axis('off')
        earlyRespRangeStr = (f'{1e3*earlyResponseRange[0]:0.0f}-' +
                             f'{1e3*earlyResponseRange[1]:0.0f} ms')
        axHist3.text(0.95, 0.2, earlyRespRangeStr, transform=axHist3.transAxes,
                     color='c', fontsize=10, ha='right', va='top')
        axHist2 = axHist.twinx()
        responseColor = plt.colormaps['viridis'](0.3)
        axHist2.plot(sortedChannels, avgResponse, color=responseColor, lw=2)
        respRangeStr = f'{1e3*responseRange[0]:0.0f}-{1e3*responseRange[1]:0.0f} ms'
        axHist2.text(0.95, 0.12, respRangeStr, transform=axHist3.transAxes,
                     color=responseColor, fontsize=10, ha='right', va='top')
        axHist2.set_ylabel(f'Avg response [{respRangeStr}] (uV)',
                           color=responseColor, rotation=270, labelpad=16)

    plt.suptitle(f'{subject} [{indpage+1}/{nPages}]', fontsize=16, x=0.53)
    plt.show()

    # -- Save the figure --
    if SAVE:
        outputFilename = f'{subject}_avgLFP_{indpage+1}'
        extraplots.save_figure(outputFilename, 'png', pageSize, outputDir=outputDir,
                               facecolor='w')

    plt.pause(0.1)        
    #plt.waitforbuttonpress()
    #break
