"""
Estimate the boundaries of brain areas from the Allen Brain Atlas.

This requires the AllenSDK to be installed (we use the virtual environment allensdk).

3D viewer with auditory and surrounding areas:
https://connectivity.brain-map.org/3d-viewer?v=1&types=PLY%2CIMAGEPLANE&PLY=541%2C1027%2C1002%2C1011%2C1018%2C402%2C378%2C895%2C312782574%2C312782628%2C677&IMAGEPLANE=imageplanes

Issues:
annot[60,110,150] -> 312782656
but that id is not in atlas.structureTable.
plt.clf(); plt.imshow(annot[60,:,:],vmax=1400)
"""

import os
import sys
import numpy as np
from skimage import measure
from skimage import filters
import matplotlib.pyplot as plt
from jaratoolbox import histologyanalysis as ha
from jaratoolbox import settings
import studyparams
import studyutils

SAVE_TO_FIGURESDATA = False  # Use False for testing (saved to temp folder)

JOIN_DORSAL_AREAS = True

#cortical_layer = '6a'  # Cortical layer
cortical_layer = '4'  # Cortical layer
figDataFile = f'brain_areas_boundaries_L{cortical_layer}.npz'
if SAVE_TO_FIGURESDATA:
    figDataDir = os.path.join(settings.FIGURES_DATA_PATH, studyparams.STUDY_NAME)
else:
    figDataDir = settings.TEMP_OUTPUT_PATH  # Save to tmp during testing
figDataFullPath = os.path.join(figDataDir,figDataFile)

brain_areas = ['AUDp','AUDv','AUDd','AUDpo','TEa']
nAreas = len(brain_areas)
areas_to_join = ['AUDd','AUDpo'] if JOIN_DORSAL_AREAS else []

# -- Load the atlas annotations --
atlas = ha.AllenAnnotation()
annot = atlas.annotationVol.astype('int')

print(f'Estimating brain silhouette (background)...', flush=True)
yz_background = np.any(annot,axis=0)

print(f'Estimating areas boundaries on cortical layer {cortical_layer}...', flush=True)
areasIDs = []
for inda, oneArea in enumerate(brain_areas):
    thisAreaID = atlas.get_id_from_acronym(oneArea+cortical_layer)
    areasIDs.append(thisAreaID)
annot[~np.isin(annot, areasIDs)] = 0

# -- Find the indices of the first non-zero value along the x-axis for each (y, z) position --
first_nonzero_indices = np.argmax(annot != 0, axis=0)

# -- Use the indices to extract the values from annot (and get a view of the yz plane) --
yinds, zinds = np.meshgrid(np.arange(annot.shape[1]), np.arange(annot.shape[2]), indexing='ij')
yz_plane_view = annot[first_nonzero_indices, yinds, zinds]  # View of brain surface from the right side

# -- Extract extent of regions combined (to define quadrants) --
indsAP = np.flatnonzero(yz_plane_view.sum(axis=0))
indsDV = np.flatnonzero(yz_plane_view.sum(axis=1))
extentAP = [indsAP[0], indsAP[-1]]
extentDV = [indsDV[0], indsDV[-1]]

# -- Estimate contours of each area --
print(f'Estimating contours...')
yz_subset_view = np.copy(yz_background)-2  # Useful for testing
eachAreaPixels = np.empty((nAreas, *yz_plane_view.shape), dtype=bool)
for inda, areaID in enumerate(areasIDs):
    areaPixels = np.isin(yz_plane_view, areaID)
    yz_subset_view[areaPixels] = inda
    eachAreaPixels[inda] = areaPixels

if JOIN_DORSAL_AREAS:
    inds_to_join = [brain_areas.index(oneArea) for oneArea in areas_to_join]
    eachAreaPixels[inds_to_join[0]] = np.logical_or(eachAreaPixels[inds_to_join[0]],
                                                    eachAreaPixels[inds_to_join[1]])
    eachAreaPixels = np.delete(eachAreaPixels, inds_to_join[1], axis=0)
   
contours = []
for inda, areaPixels in enumerate(eachAreaPixels):
    # -- Smooth out the area pixels image --
    areaPixelsSmooth = filters.gaussian(areaPixels.astype('float'), sigma=2)
    contours.append(measure.find_contours(areaPixelsSmooth, 0.5)[0])


# -- Extract extent of regions combined (to define quadrants) --
indsAP = np.flatnonzero((yz_subset_view>=0).sum(axis=0))
indsDV = np.flatnonzero((yz_subset_view>=0).sum(axis=1))
extentAP = [indsAP[0], indsAP[-1]]
extentDV = [indsDV[0], indsDV[-1]]


# -- Display the image and plot all contours found for a single level--
plt.clf()
#plt.imshow(yz_subset_view, cmap=plt.cm.gray)  # vmax=1400
plt.imshow(yz_background, cmap=plt.cm.gray)
for contour in contours:
    plt.plot(contour[:, 1], contour[:, 0], linewidth=1)
plt.axis('image')
if 0:
    pad = 10
    plt.xlim([extentAP[0]-pad, extentAP[1]+pad])
    plt.ylim([extentDV[0]-pad, extentDV[1]+pad])
plt.show()

# -- Save the data --
np.savez(figDataFullPath, yz_plane_view=yz_plane_view, yz_background=yz_background,
         contours=contours, brain_areas=brain_areas, cortical_layer=cortical_layer,
         areas_to_join=areas_to_join, extentAP=extentAP, extentDV=extentDV)
print(f'Saved boundaries data to {figDataFullPath}')


