from matplotlib import cm
import numpy as np
import os
from jaratoolbox import settings

subject_list = ['feat004', 'feat005', 'feat006', 'feat007', 'feat008', 'feat009', 'feat010']
# 'feat001', 'feat011','feat014', 'feat015', 'feat016', 'feat017', 'feat018', 'feat019'

recordingDate_list = {
    'feat001': ['2021-11-09', '2021-11-11', '2021-11-16', '2021-11-17', '2021-11-18', '2021-11-19'],
    'feat004': [('2022-01-11'), ('2022-01-19'), ('2022-01-21')],
    'feat005': [('2022-02-07'), ('2022-02-08'), ('2022-02-11'), ('2022-02-14'), ('2022-02-15'), ('2022-02-16')],
    'feat006': [('2022-02-21'), ('2022-02-22'), ('2022-02-24'), ('2022-02-25'), ('2022-02-26'),
                ('2022-02-28'), ('2022-03-01'), ('2022-03-02')],
    'feat007': [('2022-03-10'), ('2022-03-11'), ('2022-03-15'), ('2022-03-16'), ('2022-03-18'), ('2022-03-21')],
    'feat008': [('2022-03-23'), ('2022-03-24'), ('2022-03-25')],
    'feat009': [('2022-06-04'), ('2022-06-05'), ('2022-06-06'), ('2022-06-07'), ('2022-06-09'), ('2022-06-10')],
    'feat010': [('2022-06-21'), ('2022-06-22'), ('2022-06-27'), ('2022-06-28'), ('2022-06-30')],
    'feat011': ['2022-11-16', '2022-11-18', '2022-11-20', '2022-11-21', '2022-11-22'],
    'feat014': ['2024-02-22', '2024-02-28', '2024-02-29', '2024-03-04', '2024-03-06', '2024-03-08', '2024-03-09'],
    'feat015': ['2024-02-23', '2024-02-27', '2024-02-28', '2024-03-01', '2024-03-06', '2024-03-20', '2024-03-21',
                '2024-03-22'],
    'feat016': ['2024-03-21', '2024-03-22', '2024-03-23', '2024-03-24', '2024-04-04', '2024-04-08', '2024-04-09',
                '2024-04-10', '2024-04-11', '2024-04-12', '2024-04-17'],
    'feat018': ['2024-06-06', '2024-06-07', '2024-06-10', '2024-06-11', '2024-06-12', '2024-06-14', '2024-06-15',
                '2024-06-17', '2024-06-18', '2024-06-26', '2024-06-27'],
    'feat019': ['2024-06-12', '2024-06-13', '2024-06-14', '2024-06-17', '2024-06-18', '2024-06-19', '2024-06-27',
                '2024-06-28']
}

STUDY_NAME_2022 = '2022paspeech'
STUDY_NAME = '2024popanalysis'
figSavePath = "/Users/zoetomlinson/Desktop/MurrayLab/figures/"
dbSavePath = "/Users/zoetomlinson/Desktop/MurrayLab/AudPopAnalysis/data"
fullDbPath = 'celldb_2024popanalysis.h5'
databaseDir = os.path.join(settings.DATABASE_PATH, STUDY_NAME)
fullPath = os.path.join(databaseDir, fullDbPath)

targetSiteNames = ["Primary auditory area", "Dorsal auditory area", "Ventral auditory area"]
max_trials = {'PT': 640, 'AM': 220, 'speech': 381}
leastCellsArea = 10000
evoked_start = 0.015
evoked_end = 0.3
pt_evoked_end = 0.1
binWidth = 0.01
periodsNameSpeech = ['base200', 'respOnset', 'respSustained']
allPeriodsSpeech = [[-0.2, 0], [0, 0.12], [0.12, 0.24]]
timeRangeSpeech = [allPeriodsSpeech[0][0], allPeriodsSpeech[-1][-1]]
binSize = 0.005
data_dict = {}
binEdges = np.arange(evoked_start, evoked_end, binWidth)
binEdgesPT = np.arange(evoked_start, pt_evoked_end, binWidth)
binEdgesSpeech = np.arange(allPeriodsSpeech[1][0], allPeriodsSpeech[1][1], binSize)
sound_type_load = [("speech", "FTVOTBorders"), ("AM", "AM"), ("PT", "pureTones")]
min_speech_freq_dict = {(0, 0): 31, (0, 33): 29, (0, 67): 32, (0, 100): 24, (33, 100): 34, (67, 100): 35,
                        (100, 100): 33, (100, 67): 29, (100, 33): 35, (100, 0): 35, (67, 0): 31, (33, 0): 33}
unique_labels = [(0, 0), (0, 33), (0, 67), (0, 100), (33, 100), (67, 100),
                 (100, 100), (100, 67), (100, 33), (100, 0), (67, 0), (33, 0)]

# Initialize a dictionary to store counts for each frequency across mouse-date combos
freq_kept_counts = {tuple(freq): 0 for freq in unique_labels}
# Initialize an empty dictionary to store the neuron counts for each subject
neuron_counts = {subject: 0 for subject in subject_list}

min_neuron_dict = {"Primary auditory area": 9, "Ventral auditory area": 9}
previous_frequency_speech = None
previous_frequency_AM = None
previous_frequency_PT = None

spike_windows = {'pt - onset': [0.0, 0.03],  # [evoked_start, evoked_stop] in s
                 'pt - sustained': [0.03, 0.1],
                 'pt - offset': [0.1, 0.13],
                 'am - onset': [0.0, 0.2],      # [evoked_start, evoked_stop] in s
                 'am - sustained': [0.2, 0.5],
                 'am - offset': [0.5, 0.7],
                 'speech - onset': [0.0, 0.2],
                 'speech - sustained': [0.2, 0.5],
                 'speech - offset': [0.5, 0.7],
                 'naturalSound - onset': [0.0, 0.5],
                 'naturalSound - sustained': [1, 4],
                 'naturalSound - offset': [4, 4.5]}

# Mapping from old window names to simplified names
window_name_mapping = {
    'speech - onset': 'onset',
    'speech - sustained': 'sustained',
    'speech - offset': 'offset',
    'pt - onset': 'onset',
    'pt - sustained': 'sustained',
    'pt - offset': 'offset',
    'am - onset': 'onset',
    'am - sustained': 'sustained',
    'am - offset': 'offset'
}

short_names = {
    'Primary auditory area': 'Primary',
    'Dorsal auditory area':  'Dorsal',
    'Ventral auditory area': 'Ventral'
}

color_palette = {
    "Primary auditory area - PT": cm.winter(1),      # Dark blue (Winter bottom)
    "Primary auditory area - AM": cm.winter(0.66),      # Medium blue (Winter middle)
    "Primary auditory area - speech": cm.winter(0.33),  # Light blue (Winter top)

    "Dorsal auditory area - PT": cm.magma(1),       # Dark purple (Magma bottom)
    "Dorsal auditory area - AM": cm.magma(0.66),       # Purple-pink (Magma middle)
    "Dorsal auditory area - speech": cm.magma(0.33),   # Light pinkish (Magma top)

    "Ventral auditory area - PT": cm.summer(1),      # Dark green (Summer bottom)
    "Ventral auditory area - AM": cm.summer(0.66),      # Medium green (Summer middle)
    "Ventral auditory area - speech": cm.summer(0.33),  # Pale minty green (Summer top)
}