import os
import shutil
from collections import Counter
from glob import glob

import mne
from mne_nirs.io.snirf import write_raw_snirf
from mne_nirs.io import fold
import mne_nirs
from mne_bids import write_raw_bids, BIDSPath, read_raw_bids
import matplotlib.pyplot as plt
from mne_nirs.experimental_design import make_first_level_design_matrix
from mne.preprocessing.nirs import (optical_density,
                                    temporal_derivative_distribution_repair)
from nilearn.plotting import plot_design_matrix
import numpy as np
import pandas as pd

root = r'C:\Datasets\Test-retest study\bids_dataset'  # Replace with the path to your data
task = "auditory"        # Set to whatever the name of your experiment is
stimulus_duration = {'Control': 5, 'Noise': 5, 'Speech': 5.25}
trigger_info = {'1.0': 'Control',
                '2.0': 'Noise',
                '3.0': 'Speech',
                '4.0': 'Xstart', # start of break?
                '5.0': 'Xend'}   # end of break?

subject_dirs = glob(os.path.join(root, "sub-*"))
subjects = [subject_dir.split("-")[-1] for subject_dir in subject_dirs] # ["01","02",...]

ses_list = np.array([])
for folder in subject_dirs:
    ses_list = np.append(ses_list,np.array([ses for ses in os.listdir(folder)]))
ses_dict = dict(Counter(ses_list))
# Check if all participants came the same number of times
sessions = list(ses_dict.keys())
sessions =[ses.split("-")[-1] for ses in sessions] # ["01","02",...]
complete_sub_ses = all([value == len(subjects) for value in ses_dict.values()])

print(f"Found subjects: {subjects}")
print(f"Found sessions: {sessions}")
if complete_sub_ses:
    print(f"All {len(subjects)} subject came for {len(sessions)} sessions.")

def load_data(bids_path):

    raw_intensity = read_raw_bids(bids_path=bids_path, verbose=False)
    raw_intensity.load_data()
    return raw_intensity

raw_intensities = []

for sub in subjects:
    for ses in sessions:
        # Create path to file based on experiment info
        bids_path = BIDSPath(subject=sub,
                             session=ses,
                             task="auditory",
                             root=r"C:\Datasets\Test-retest study\bids_dataset",
                             datatype="nirs",
                             suffix="nirs",
                             extension=".snirf")

        raw_int_sub_ses = load_data(bids_path)
        raw_intensities.append(raw_int_sub_ses)

# Include information about duration of each stimulus
raw_intensity = raw_intensities[0]
raw_intensity.annotations.set_durations({'Control': 5, 'Noise': 5, 'Speech': 5.25})

# Retrieve all events
AllEvents, event_dict = mne.events_from_annotations(raw_intensity)
# Break events - between end and start of a block
### TODO: check trigger code start and end
Breaks, event_dict_breaks = mne.events_from_annotations(raw_intensity, {'Xend': 5, 'Xstart': 4})

# Converting from index to time -> T = N/Fs
# We resample the data to make indexing exact times more convenient.
sampling_freq = raw_intensity.info["sfreq"]
Breaks = Breaks[:, 0] /sampling_freq
LastEvent = AllEvents[-1, 0] / sampling_freq

# Constructing block for each block in experiment
# TODO: crops and append blocks without breaks?
# TODO: cropping only for sci analysis?
cropped_intensity = raw_intensity.copy().crop(Breaks[0],Breaks[1])
block2 = raw_intensity.copy().crop(Breaks[2],Breaks[3])
block3 = raw_intensity.copy().crop(Breaks[4],Breaks[5])
block4 = raw_intensity.copy().crop(Breaks[6], LastEvent+15.25)

# Combining all blocks
# append time courses, not subjects
cropped_intensity.append([block2, block3, block4])
cropped_od = optical_density(cropped_intensity)
cropped_corrected_od = temporal_derivative_distribution_repair(cropped_od)

# Selecting channels appropriate for detecting neural responses
picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True) # 66 channels, 33 HbO 33 HbR
# Average distances larger than 3cm?
dists = mne.preprocessing.nirs.source_detector_distances(
    raw_intensity.info, picks=picks
)

# Alternative method for short channel extraction
# Separating channels that are too close together (short channels) to detect a neural response
# (less than 1 cm distance between optodes)
# raw_intensity.pick(picks[dists > 0.01]) # go from 66 to 50 channels

# Preprocessing the data to get od and haemo information
raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=6)


# Extracting short and long channels
short_chs = mne_nirs.channels.get_short_channels(raw_haemo) # 8 short channels 13 -> 20
raw_haemo = mne_nirs.channels.get_long_channels(raw_haemo) # 25 long channels 33-8 = 25

print(f"Long channels: {raw_haemo.ch_names}")
print(f"Short channels: {short_chs.ch_names}")

sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)


### TODO: PLOTS NOT WORKING - ADJUST TO EXPERIMENT
#event_plot = mne.viz.plot_events(AllEvents, event_id=event_dict, sfreq=raw_haemo.info['sfreq'])

#s = mne_nirs.experimental_design.create_boxcar(raw_haemo,stim_dur =500)
#fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
#plt.plot(raw_haemo.times, s, axes=axes)
#plt.xlabel("Time (s)")

# this launches MNE plotter
#raw_haemo.plot(
#    n_channels=len(raw_haemo.ch_names), duration=500, show_scrollbars=False
#)

#----------------

## Start General Linear Model (GLM)
# Create design matrix without regressors
# TODO: try with and without short channel regressors
# TODO: which HRF model to chose? Investigate FIR?
# TODO: verify durations?
# Returns a dataframe
design_matrix = make_first_level_design_matrix(raw_haemo,
                                               drift_model=None,
                                               high_pass=0.01,
                                               hrf_model='spm',
                                               stim_dur=raw_haemo.annotations.duration)

# Add short channel regressors to dataframe
design_matrix["ShortHbO"] = np.mean(short_chs.copy().pick(picks="hbo").get_data(), axis=0)
design_matrix["ShortHbR"] = np.mean(short_chs.copy().pick(picks="hbr").get_data(), axis=0)

fig, ax1 = plt.subplots(figsize=(10, 6), nrows=1, ncols=1)
fig = plot_design_matrix(design_matrix, ax=ax1)
