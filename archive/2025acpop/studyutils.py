"""
Additional functions for 2024traje project.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def simplify_site_name(recordingSiteName):
    simplerName = recordingSiteName.split(',')[0]
    simplerName = simplerName.replace('Supplemental', 'Supp')
    return simplerName

def get_onset_offset_bins(spkData):
    """
    Args:
        spkData (dict): must contain keys 'binEdges', 'soundDuration'
    """
    onsetBin = np.flatnonzero(spkData['binEdges']<0)[-1]
    offsetBin = np.flatnonzero(spkData['binEdges']<spkData['soundDuration'])[-1]
    return onsetBin, offsetBin

# -- Methods to convert from pixels of Atlas to mm (from Bregma)--
def pix2mmAP(pixels):
    return -0.94 - (280-pixels)*0.025
def mm2pixAP(mm):
    return ((mm+0.94) / 0.025) + 280
def pix2mmDV(pixels):
    return (pixels-10)*0.025
def mm2pixDV(mm):
    return (mm/0.025) + 10

