"""
Estimate the areas covered by a particular electrode penetration.

Based on histologyanalysis.cell_locations()
"""
from typing_extensions import Unpack

import os
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.gridspec as gridspec
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from jaratoolbox import settings
from jaratoolbox import histologyanalysis as ha
from jaratoolbox import celldatabase
from jaratoolbox import extraplots
import studyparams
import studyutils

SAVE = True

if len(sys.argv)>1:
    subject = sys.argv[1]
else:
    subject = 'feat017'
    
outputDir = os.path.join(settings.FIGURES_DATA_PATH, studyparams.STUDY_NAME, 'penetrations')
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

pageSize = [14, 8]  # Size of the figure in inches    
sessionsPerPage = 1

# NOTE: The first electrode in NP1.0 is actually 175um from tip, but we use 100 here.
from_tip = lambda x: 20*(x//2)+100  # Assumes channels start at 0
channelDistanceFromTip = [from_tip(ch) for ch in range(0,385)]
shank = 1  # NP1.0 has a single shank, so we use 1 and the shankID

# -- Define the location where the SDK will download files --
ALLEN_SDK_MANIFEST = os.path.join(settings.ALLEN_SDK_CACHE_PATH, 'manifest.json')
resolution = 25 # In microns/pixel
mcc = MouseConnectivityCache(resolution=resolution, manifest_file=ALLEN_SDK_MANIFEST)
rsp = mcc.get_reference_space()
rspAnnotationVolumeRotated = np.rot90(rsp.annotation, 1, axes=(2, 0))

if 0:
    plt.clf()
    plt.imshow(rspAnnotationVolumeRotated[:,:,210].T, clim=[0,1600])
    plt.axis('equal')

# -- Read tracks file with penetration information --
fileNameInfohist = os.path.join(settings.INFOHIST_PATH, '{}_tracks.py'.format(subject))
tracks = ha.read_tracks_file(fileNameInfohist).tracks
tracksDF = pd.DataFrame(tracks)

# -- Read inforec file with session information --
inforecFile = os.path.join(settings.INFOREC_PATH, f'{subject}_inforec.py')
inforec = celldatabase.read_inforec(inforecFile)

# -- Plot the atlas slice and penegration for each session --
fig = plt.gcf()  #figure(figsize=pageSize)
nPages = int(np.ceil(len(inforec.experiments) / sessionsPerPage))
for indpage in range(nPages):
    fig.clf()
    # -- Main gridspec --
    gsMain = gridspec.GridSpec(sessionsPerPage, 2, width_ratios=[0.8, 0.2])
    gsMain.update(left=0.01, right=0.95, top=0.94, bottom=0.05, wspace=0.01) #, hspace=0.3)
    
    indSessionsToPlot = range(indpage*sessionsPerPage, (indpage+1)*sessionsPerPage)
    for indplot, inds in enumerate(indSessionsToPlot):
        if inds >= len(inforec.experiments):
            break
        experiment = inforec.experiments[inds]

        outsideBrain = np.max(channelDistanceFromTip)-experiment.maxDepth
        recordingTrack = experiment.recordingTrack
        brainArea = experiment.brainArea
        maxDepth = experiment.maxDepth
        selection= 'brainArea==@brainArea & recordingTrack==@recordingTrack'
        thisTrack = tracksDF.query(selection).iloc[0]
        histImage = thisTrack['histImage']
        filenameSVG = ha.get_filename_registered_svg(subject, brainArea, histImage,
                                                     recordingTrack, shank)
        depths = np.linspace(0, maxDepth, 100)
        brainSurfCoords, tipCoords, siteCoords = ha.get_coords_from_svg(filenameSVG,
                                                                        depths,
                                                                        maxDepth)
        # Find point outside brain extending in direction brainSurfCoord and tipCoords
        brainSurfCoords = np.array(brainSurfCoords)
        tipCoords = np.array(tipCoords)
        electrodeVector = (brainSurfCoords-tipCoords)
        electrodeVector = electrodeVector / np.linalg.norm(electrodeVector)
        outsideBrainPix = outsideBrain/resolution
        topElectrodeCoords = brainSurfCoords + outsideBrainPix*electrodeVector

        atlasZ = thisTrack['atlasZ']
        eachCoord = np.hstack((np.array(siteCoords), np.tile(atlasZ, (len(depths),1))))
        eachCoordID = rspAnnotationVolumeRotated[Unpack[eachCoord.T.astype(int)]] # CHANGED by zoe on 8/18/25
        structDict = rsp.structure_tree.get_structures_by_id(eachCoordID)
        
        transitions = (np.diff(eachCoordID,prepend=1) != 0)
        #transitions[-1] = True
        transitionsInds = np.flatnonzero(transitions)
        transitionsDepth = depths[transitions]
        #transitionsDepthFraction = transitionsDepth / maxDepth
        transitionsFromTipFraction = 1 - (transitionsDepth / maxDepth)

        if 0:
            for indt, depthFrac in zip(transitionsInds,transitionsFromTipFraction):
                oneStruct = structDict[indt]
                if oneStruct is None:
                    continue
                pcStr = f'{100*depthFrac:3.0f}%'
                depthStr = f'{maxDepth-depths[indt]:04.0f}um'
                nameStr = oneStruct["name"]
                structID = oneStruct["id"]
                print(f'[{pcStr}] {depthStr}: {nameStr} ({structID})')

        # -- Plot the results --
        axAtlas = plt.subplot(gsMain[indplot,0])
        cmap = plt.get_cmap('tab20b')(np.linspace(0, 1, 32))
        cmap = np.tile(cmap, (3,1)) / 1.5
        cmap[0] = [0.1, 0.1, 0.1, 1]  # Set background to black
        #cmap[0] = [1, 1, 1, 1]  # Set background to white
        cmap = mcolors.ListedColormap(cmap)
        plt.imshow(rspAnnotationVolumeRotated[:,:,atlasZ].T, clim=[0,1600], cmap=cmap)
        yLims = plt.ylim()
        plt.plot(*zip(brainSurfCoords, topElectrodeCoords), lw=4, color=[0.5,0,0], clip_on=False)
        plt.plot(*zip(brainSurfCoords, tipCoords), lw=4, color=[1,0,0])
        plt.ylabel(f'{experiment.date}')
        plt.axis('equal')
        plt.axis('off')
        plt.title(f'{subject}: {experiment.date} [{indpage+1}/{nPages}]', fontsize=16)
        plt.ylim(yLims)
        plt.show()
        
        axAreas = plt.subplot(gsMain[indplot,1])
        plt.cla()
        for indlabel, (indt, depthFrac) in enumerate(zip(transitionsInds,
                                                         transitionsFromTipFraction)):
            oneStruct = structDict[indt]
            if oneStruct is None:
                continue
            pcStr = f'{100*depthFrac:3.0f}%'
            #depthStr = f'{maxDepth-depths[indt]:04.0f}um'
            depthStr = f'{maxDepth-depths[indt]:0.0f}um'
            nameStr = oneStruct["name"]
            structID = oneStruct["id"]
            fontweight = 'bold' if 'auditory' in nameStr.lower() else 'normal'
            #plt.text(0, len(transitionsInds)-indlabel, f'[{pcStr}] {depthStr}: {nameStr}',
            #         fontweight=fontweight)
            plt.text(0, len(transitionsInds)-indlabel,
                     f'[{pcStr}] {depthStr}')
            plt.text(0.1, len(transitionsInds)-indlabel-0.5,
                     f'{nameStr}', fontweight=fontweight)
        plt.title(f'Distance from tip\n(assuming depth = {maxDepth}um)')
        
        plt.axis('off')
        plt.ylim([0, 1.05*len(transitionsInds)])
    # -- Save the figure --
    if SAVE:
        outputFilename = f'{subject}_penetrations_{indpage+1:02.0f}'
        extraplots.save_figure(outputFilename, 'png', pageSize,
                               outputDir=outputDir, facecolor='w')

    plt.pause(0.1)        
    #break
