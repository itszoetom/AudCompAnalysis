"""
Show the location of each recorded cell on a lateral view of the brain surface.
"""

import os
import sys
sys.path.append('..')
import numpy as np
from matplotlib import pyplot as plt
from jaratoolbox import settings
from jaratoolbox import celldatabase
from jaratoolbox import extraplots
import studyparams
import studyutils


SAVE = 0
figSize = [7, 10]  # Size of the figure in inches

if len(sys.argv)>1:
    subject = sys.argv[1]
else:
    subject = None
  
np.random.seed(0)

labelSize = 10
fontSizeLabels = 10
boundDataFile = 'brain_areas_boundaries_L4.npz'
boundDataFileL6 = 'brain_areas_boundaries_L6a.npz'
boundDataFileL1 = 'brain_areas_boundaries_L1.npz'

# -- Load the database of cells --    
dbPath = os.path.join(settings.DATABASE_PATH, studyparams.STUDY_NAME)
dbCoordsFilename = os.path.join(dbPath,f'celldb_{studyparams.STUDY_NAME}_coords.h5')
celldb = celldatabase.load_hdf(dbCoordsFilename)

dbCoordsFilename = os.path.join(dbPath,f'celldb_2024traje_responsive.h5')
celldbResp = celldatabase.load_hdf(dbCoordsFilename)

if subject is not None:
    celldb = celldb.query('subject==@subject')
    celldbResp = celldbResp.query('subject==@subject')

# -- Define brain areas of interest --
audCtxAreas = ['Primary auditory area','Dorsal auditory area',
               'Ventral auditory area', 'Temporal association areas']
recordingSiteName = celldb.recordingSiteName.copy()
recordingSiteName = recordingSiteName.str.replace('Posterior auditory area',
                                                  'Dorsal auditory area')
recordingSiteName = np.array(recordingSiteName)
isAudArea = celldb.recordingSiteName.str.contains('auditory area')
ctxAreas = ['Anterolateral visual area', 'Dorsal auditory area', 'Ectorhinal area',
            'Endopiriform nucleus', 'Lateral visual area', 'Laterointermediate area',
            'Perirhinal area', 'Posterior auditory area', 'Primary auditory area',
            'Supplemental somatosensory area', 'Temporal association areas',
            'Ventral auditory area', 'auditory radiation', 'optic radiation',
            'corpus callosum', 'external capsule']

isCortical = np.zeros(len(isAudArea), dtype = bool)
for indArea, thisArea in enumerate(ctxAreas):
    isCortical[celldb.recordingSiteName.str.contains(thisArea)] = True
    
layersDeep = celldb.recordingSiteName.str.contains('layer 5|layer 6') & isCortical
layer4 =  celldb.recordingSiteName.str.contains('layer 4') & isCortical
layersSuperficial =  celldb.recordingSiteName.str.contains('layer 1|layer 2|layer 3') & isCortical

y_coords = celldb.y_coord.copy()
z_coords = celldb.z_coord.copy()
x_coords = celldb.x_coord.copy()

jitter = 0.4 # in pixels
z_coords_jittered = z_coords + jitter*(2*np.random.randn(len(z_coords))-1)
x_coords_jittered = x_coords + jitter*(2*np.random.randn(len(x_coords))-1)

# -- Load main boundary file --
boundDataDir = os.path.join(settings.FIGURES_DATA_PATH, studyparams.STUDY_NAME)
boundDataFullPath = os.path.join(boundDataDir,boundDataFile)
boundData = np.load(boundDataFullPath, allow_pickle = True)
contours = boundData['contours']
extentAP = boundData['extentAP']
extentDV = boundData['extentDV']
# -- Load additional boundary files --
boundDataFullPath = os.path.join(boundDataDir,boundDataFileL6)
boundDataL6 = np.load(boundDataFullPath, allow_pickle = True)
contoursL6 = boundDataL6['contours']
boundDataFullPath = os.path.join(boundDataDir,boundDataFileL1)
boundDataL1 = np.load(boundDataFullPath, allow_pickle = True)
contoursL1 = boundDataL1['contours']

APtickLocs = np.array([ 156 ,176, 196, 216, 236])
APtickLabels = np.round(studyutils.pix2mmAP(APtickLocs),1)
DVtickLocs = np.array([210, 190, 170, 150, 130, 110, 90, 70, 50])
DVtickLabels = np.round(studyutils.pix2mmDV(DVtickLocs),1)

# -- Plot the locations of the recorded cells -- #
fig = plt.gcf()
plt.clf()
axCellLocs = plt.gca()
axCellLocs.invert_yaxis()
#studyutils.plot_quadrants(axCellLocs, extentAP, extentDV, color='0.8')
for indc, contour in enumerate(contours):
    contourL1 = contoursL1[indc]
    contourL6 = contoursL6[indc]
    L1, = plt.plot(contourL1[:, 1], contourL1[:, 0], lw=1.5, ls=':', color='g', clip_on=False)
    L4, = plt.plot(contour[:, 1], contour[:, 0], lw=1.5, color='k', clip_on=False)
    L6, = plt.plot(contourL6[:, 1], contourL6[:, 0], lw=1.5, ls=':', color='r', clip_on=False)
if 0:
    plt.text(234, 119, 'D', ha='center', fontsize=labelSize)
    plt.text(224, 127, 'P', ha='center', fontsize=labelSize)
    plt.text(233, 142, 'V', ha='center', fontsize=labelSize)
    plt.text(233, 165, 'TeA', ha='center', fontsize=labelSize)
else:      
    plt.text(240, 114, 'D', ha='center', fontsize=labelSize, fontweight='bold')
    plt.text(232, 122, 'P', ha='center', fontsize=labelSize, fontweight='bold')
    plt.text(240, 138, 'V', ha='center', fontsize=labelSize, fontweight='bold')
    plt.text(240, 165, 'TeA', ha='center', fontsize=labelSize, fontweight='bold')

selectedCells = isCortical #& isAudArea #& layersDeep
allCell = plt.plot(z_coords_jittered[selectedCells],
                    y_coords[selectedCells], '.', color='0.5', ms=1)
responsiveOnset = celldbResp.nsMinPvalOnset < 0.01
responsiveSustain = celldbResp.nsMinPvalSustain < 0.01
if 1:
    sustainCells, = plt.plot(z_coords_jittered[selectedCells & responsiveSustain],
                            y_coords[selectedCells & responsiveSustain], 'o',
                            color='C0', ms=2, label='Sustain')
if 0:    
    onsetCells, = plt.plot(z_coords_jittered[selectedCells & responsiveOnset],
                          y_coords[selectedCells & responsiveOnset], 'o',
                          color='C3', ms=1, label='Onset')

#plt.xlim(146, 246)
#plt.ylim(220,40)
SHOW_UNITS_IN_MM = 1
if SHOW_UNITS_IN_MM:
    plt.xticks(APtickLocs, APtickLabels)
    plt.yticks(DVtickLocs, DVtickLabels)
    units = 'mm'
else:
    units = 'atlas voxels'
plt.legend([L1, L4, L6, sustainCells],
           ['Projected from L1', 'Projected from L4', 'Projected from L6', 'Sustained response'],
           loc='upper left', fontsize=labelSize, frameon=False)
plt.xlabel(f'Posterior-Anterior ({units})', fontsize = fontSizeLabels)
plt.ylabel(f'Ventral-Dorsal ({units})', fontsize = fontSizeLabels)
axCellLocs.spines["right"].set_visible(False)
axCellLocs.spines["top"].set_visible(False)
axCellLocs.set_aspect('equal')

if subject is None:
    subjectStr = ', '.join(celldb.subject.unique())
else:
    subjectStr = subject
plt.title(subjectStr, fontsize=12, fontweight='bold', pad=20)

plt.show()

# Save the figure
outputDir = os.path.join(settings.FIGURES_DATA_PATH, studyparams.STUDY_NAME)
if SAVE:
    extraplots.save_figure('cell_locations_lateral_view', 'png', figSize,
                           outputDir=outputDir, facecolor='w')
