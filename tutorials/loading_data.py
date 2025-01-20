import os
import numpy as np
import mne
import matplotlib

#%matplotlib qt

### TUTORIAL ###
# basic EEG/MEG pipeline for event-related analysis: loading data, epoching, averaging, plotting, and estimating cortical
# activity from sensor data. It introduces the core MNE-Python data structures Raw, Epochs, Evoked, and SourceEstimate

## Loading data
# MNE-Python data structures -> FIF file format
# EEG and MEG data from one subject performing audiovisual experiment + structural MRI scans

sample_data_folder = mne.datasets.sample.data_path(download=False) # Creating C:\Users\beatr\mne_data
sample_data_raw_file = (
    sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
)
raw = mne.io.read_raw_fif(sample_data_raw_file)

## 'Raw' object
print(raw)
# MNE-Python detects different sensor types and handles each appropriately
print(raw.info) # dictionary-like object preserved for Raw, Epochs, Evoked objects


# Plotting methods of 'Raw' objects
raw.compute_psd(fmax=50).plot(picks="data", exclude="bads", amplitude=False)
# Setting block=True halts after plotting
raw.plot(duration=5, n_channels=30,block=True)

#print(mne.sys_info())

## Preprocessing
# mne.preprocessing
# Clean data by performing independent component analysis (ICA)
# https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html#tut-artifact-ica


