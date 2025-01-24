from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.conftest import subjects_dir

######## Preprocessing fNIRS data ########
# How to convert fNIRS data from raw measurements to relative oxyhaemoglobin (HbO) and
# deoxyhaemoglobin (HbR) concentration, view the average waveform, and topographic
# representation of the data

# single subject recorded at Macquarie University. It has optodes placed over the motor cortex.
# There are three conditions:
# tapping the left thumb to fingers
# tapping the right thumb to fingers
# a control where nothing happens
# The tapping lasts 5 seconds, and there are 30 trials of each condition.

fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_cw_amplitude_dir = fnirs_data_folder / "Participant-1"
raw_intensity = mne.io.read_raw_nirx(fnirs_cw_amplitude_dir, verbose=True)
raw_intensity.load_data()

## Providing more meaningful annotation information:
# 1. meaningful names to trigger codes stored in annotation
# stored in description
raw_intensity.annotations.rename(
    {"1.0": "Control", "2.0": "Tapping/Left", "3.0": "Tapping/Right"}
)
# 2. include information about duration of each stimulus (5s for all conditions in experiment)
raw_intensity.annotations.set_durations(5)
# 3. remove trigger code 15, which signaled the start and end of the experiment - not relevant
unwanted = np.nonzero(raw_intensity.annotations.description == "15.0")
raw_intensity.annotations.delete(unwanted)

## Viewing location of sensors over brain surface
# validate location of sources-detector pairs and channels are in expected locations
# source-detector pairs are shown as lines between optodes
# channels are mid-point of source detector pairs optionally shown as orange dots
# sources optionally shown as red dots and detectors as black
subjects_dir = mne.datasets.sample.data_path() / "subjects"

brain = mne.viz.Brain(
    "fsaverage", subjects_dir=subjects_dir, background="w", cortex="0.5"
)
brain.add_sensors(
    raw_intensity.info,
    trans = "fsaverage",
    fnirs=["channels","pairs","sources","detectors"]
)
brain.show_view(azimuth=20, elevation=60, distance=400)







