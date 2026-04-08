"""Shared paths, dataset metadata, spike windows, and plotting settings."""

from __future__ import annotations

# --- Shared dataset metadata ---
SOUND_ORDER = ("speech", "AM", "PT", "naturalSound")
SOUND_FILE_KEYS = {
    "speech": "speech",
    "AM": "AM",
    "PT": "pureTones",
    "naturalSound": "naturalSound",
}
WINDOW_ORDER = ("onset", "sustained", "offset")
WINDOW_TO_KEY = {
    "onset": "onsetfr",
    "sustained": "sustainedfr",
    "offset": "offsetfr",
}
SOUND_DISPLAY_NAMES = {
    "speech": "Speech",
    "AM": "AM",
    "PT": "Pure Tones",
    "naturalSound": "Natural Sounds",
}
NEURONS_PER_SESSION = {
    "speech": 10,
    "AM": 30,
    "PT": 30,
    "naturalSound": 30,
}
recordingDate_list = {
    "feat001": ["2021-11-09", "2021-11-11", "2021-11-16", "2021-11-17", "2021-11-18", "2021-11-19"],
    "feat004": ["2022-01-11", "2022-01-19", "2022-01-21"],
    "feat005": ["2022-02-07", "2022-02-08", "2022-02-11", "2022-02-14", "2022-02-15", "2022-02-16"],
    "feat006": [
        "2022-02-21",
        "2022-02-22",
        "2022-02-24",
        "2022-02-25",
        "2022-02-26",
        "2022-02-28",
        "2022-03-01",
        "2022-03-02",
    ],
    "feat007": ["2022-03-10", "2022-03-11", "2022-03-15", "2022-03-16", "2022-03-18", "2022-03-21"],
    "feat008": ["2022-03-23", "2022-03-24", "2022-03-25"],
    "feat009": ["2022-06-04", "2022-06-05", "2022-06-06", "2022-06-07", "2022-06-09", "2022-06-10"],
    "feat010": ["2022-06-21", "2022-06-22", "2022-06-27", "2022-06-28", "2022-06-30"],
    "feat011": ["2022-11-16", "2022-11-18", "2022-11-20", "2022-11-21", "2022-11-22"],
    "feat014": ["2024-02-22", "2024-02-28", "2024-02-29", "2024-03-04", "2024-03-06", "2024-03-08", "2024-03-09"],
    "feat015": ["2024-02-23", "2024-02-27", "2024-02-28", "2024-03-01", "2024-03-06", "2024-03-20", "2024-03-21", "2024-03-22"],
    "feat016": [
        "2024-03-21",
        "2024-03-22",
        "2024-03-23",
        "2024-03-24",
        "2024-04-04",
        "2024-04-08",
        "2024-04-09",
        "2024-04-10",
        "2024-04-11",
        "2024-04-12",
        "2024-04-17",
    ],
    "feat018": [
        "2024-06-06",
        "2024-06-07",
        "2024-06-10",
        "2024-06-11",
        "2024-06-12",
        "2024-06-14",
        "2024-06-15",
        "2024-06-17",
        "2024-06-18",
        "2024-06-26",
        "2024-06-27",
    ],
    "feat019": ["2024-06-12", "2024-06-13", "2024-06-14", "2024-06-17", "2024-06-18", "2024-06-19", "2024-06-27", "2024-06-28"],
}

WINDOW_NAMES = ("onset", "sustained", "offset")
targetSiteNames = [
    "Primary auditory area",
    "Dorsal auditory area",
    "Ventral auditory area",
    "Posterior auditory area",
]

# --- Paths ---
DATABASE_PATH = "/Volumes/NardociData/jarahubdata/figuresdata/"
figSavePath = "/Users/zoetomlinson/Desktop/MurrayLab/figures/"
dbSavePath = "/Users/zoetomlinson/Desktop/MurrayLab/AudPopAnalysis/data/"

# --- Speech dataset ---
SPEECH_STUDY_NAME = "2024popanalysis"
SPEECH_SUBJECTS = ["feat004", "feat005", "feat006", "feat007", "feat008", "feat009", "feat010"]
fullPath_Speech = f"{DATABASE_PATH}{SPEECH_STUDY_NAME}/celldb_2024popanalysis.h5"
speech_time_range = [0.0, 0.7]
speech_allPeriods = [[-0.5, 0], [0, 0.2], [0.2, 0.5], [0.5, 0.7]]
SPEECH_REPEATS_PER_TOKEN = 20

unique_labels = [
    (0, 0),
    (0, 33),
    (0, 67),
    (0, 100),
    (33, 100),
    (67, 100),
    (100, 100),
    (100, 67),
    (100, 33),
    (100, 0),
    (67, 0),
    (33, 0),
]
SPEECH_SYLLABLES = {
    (0, 0): "ba",
    (100, 0): "da",
    (0, 100): "pa",
    (100, 100): "ta",
}

# --- Non-speech dataset ---
fullPath = f"{DATABASE_PATH}2025acpop/celldb_2025acpop_coords.h5"

NAT_SOUND_CATEGORIES = ["Frogs", "Crickets", "Streamside", "Bubbling", "Bees"]
NAT_SOUND_LABELS = [f"{category} {index}" for category in NAT_SOUND_CATEGORIES for index in range(1, 5)]
NAT_SOUND_LABEL_MAP = dict(enumerate(NAT_SOUND_LABELS))
naturalStimVar = "soundID"
NS_timeRange = [-2, 6]
natSounds_allPeriods = [[-1, 0], [0, 0.5], [1, 4], [4, 4.5]]

simpleStimVar = "currentFreq"
AM_timeRange = [-0.5, 1.5]
AM_allPeriods = [[-0.5, 0], [0, 0.2], [0.2, 0.5], [0.5, 0.7]]
PT_timeRange = [-0.1, 0.3]
PT_allPeriods = [[-0.1, 0], [0, 0.05], [0.05, 0.1], [0.1, 0.15]]

STIMULUS_BUILD_CONFIGS = {
    "speech": {
        "time_range": speech_time_range,
        "periods": speech_allPeriods,
        "window_names": WINDOW_NAMES,
    },
    "naturalSound": {
        "stim_var": naturalStimVar,
        "time_range": NS_timeRange,
        "periods": natSounds_allPeriods,
        "n_trials": 200,
    },
    "AM": {
        "stim_var": simpleStimVar,
        "time_range": AM_timeRange,
        "periods": AM_allPeriods,
        "n_trials": 220,
    },
    "pureTones": {
        "stim_var": simpleStimVar,
        "time_range": PT_timeRange,
        "periods": PT_allPeriods,
        "n_trials": 320,
    },
}

short_names = {
    "Primary auditory area": "Primary",
    "Dorsal auditory area": "Dorsal",
    "Ventral auditory area": "Ventral",
    "Posterior auditory area": "Posterior",
}
