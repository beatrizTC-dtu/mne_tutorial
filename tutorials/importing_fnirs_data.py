from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
import mne
import os.path
#from mne.conftest import subjects_dir
import os.path as op

## Supported measurement types include amplitude, optical density, oxyhaemoglobin concentration, and deoxyhemoglobin
# concentration (for continuous wave fNIRS), and additionally AC amplitude and phase (for frequency domain fNIRS).

# Standardized data: SNIRF (.snirf) -> Shared NIR Format
# Facilitate sharing and analysis of fNIRS data (NIRx manufacturer) mne.io.read_raw_nirx()
fnirs_data_folder = (r'C:/Users/beatr/Desktop/PhD/Data/Subject01-visit01/Subject01-visit01')
fnirs_data_header_file = os.path.join(fnirs_data_folder, 'NIRS-2022-10-19_001.hdr')
raw_intensity = mne.io.read_raw_nirx(fnirs_data_header_file)
raw_intensity.load_data()

# Meaningful names to trigger codes stored in annotation
# stored in description
raw_intensity.annotations.rename({'1.0': 'Control',
                                  '2.0': 'Noise',
                                  '3.0': 'Speech',
                                  '4.0': 'Xstart',
                                  '5.0': 'Xend'})

# Include information about duration of each stimulus
raw_intensity.annotations.set_durations({'Control': 5, 'Noise': 5, 'Speech': 5.25})

# Break events
Breaks, event_dict = mne.events_from_annotations(raw_intensity, {'Xend': 4, 'Xstart': 5})

# All events
AllEvents, event_dict = mne.events_from_annotations(raw_intensity)

# Converting from index to time -> T = N/Fs
sampling_freq = raw_intensity.info["sfreq"]
Breaks = Breaks[:, 0] /sampling_freq
LastEvent = AllEvents[-1, 0] / sampling_freq

# constructing block for each block in experiment
cropped_intensity = raw_intensity.copy().crop(Breaks[0],Breaks[1])
block2 = raw_intensity.copy().crop(Breaks[2],Breaks[3])
block3 = raw_intensity.copy().crop(Breaks[4],Breaks[5])
block4 = raw_intensity.copy().crop(Breaks[6], LastEvent+15.25)

# Viewing location of sensors over brain surface
#ubjects_dir = op.join(mne.datasets.sample.data_path(), "subjects")

brain = mne.viz.Brain(
    "fsaverage", subjects_dir=fnirs_data_folder, background="w", cortex="0.5"
)
brain.add_sensors(
    raw_intensity.info,
    trans="fsaverage",
    fnirs=["channels", "pairs", "sources", "detectors"],
)
brain.show_view(azimuth=20, elevation=60, distance=400)