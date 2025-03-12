from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
import mne
import os.path
mne.set_config('MNE_BROWSER_BACKEND', 'qt')
#from mne.conftest import subjects_dir
import os.path as op

## Supported measurement types include amplitude, optical density, oxyhaemoglobin concentration, and deoxyhemoglobin
# concentration (for continuous wave fNIRS), and additionally AC amplitude and phase (for frequency domain fNIRS).

# Standardized data: SNIRF (.snirf) -> Shared NIR Format
# Facilitate sharing and analysis of fNIRS data (NIRx manufacturer) mne.io.read_raw_nirx()
#fnirs_data_folder = (r'C:\Users\bede\OneDrive - Danmarks Tekniske Universitet\Desktop\Data\Test-retest study\Subject01-visit01')
fnirs_data_folder = r"C:\Datasets\Test-retest refactor\bids_dataset"
# Loop over subjectID and visitID
fnirs_data_file = os.path.join(fnirs_data_folder, 'NIRS-2022-10-19_001.hdr')
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
# We resample the data to make indexing exact times more convenient.
sampling_freq = raw_intensity.info["sfreq"]
Breaks = Breaks[:, 0] /sampling_freq
LastEvent = AllEvents[-1, 0] / sampling_freq

# constructing block for each block in experiment
cropped_intensity = raw_intensity.copy().crop(Breaks[0],Breaks[1])
block2 = raw_intensity.copy().crop(Breaks[2],Breaks[3])
block3 = raw_intensity.copy().crop(Breaks[4],Breaks[5])
block4 = raw_intensity.copy().crop(Breaks[6], LastEvent+15.25)

# Viewing location of sensors over brain surface
#subjects_dir = op.join(mne.datasets.sample.data_path(), "subjects")


#brain = mne.viz.Brain(
#    "fsaverage", subjects_dir=fnirs_data_folder, background="w", cortex="0.5"
#)
#brain.add_sensors(
#    raw_intensity.info,
#    trans="fsaverage",
#    fnirs=["channels", "pairs", "sources", "detectors"],
#)
#brain.show_view(azimuth=20, elevation=60, distance=400

## Selecting channels appropriate for detecting neural responses
picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True)
dists = mne.preprocessing.nirs.source_detector_distances(
    raw_intensity.info, picks=picks
)
# Removing channels that are too close together (short channels) to detect a neural response
# (less than 1 cm distance between optodes)
raw_intensity.pick(picks[dists > 0.01]) # go from 66 to 50 channels
# this launches MNE plotter
#raw_intensity.plot(
#    n_channels=len(raw_intensity.ch_names), duration=500, show_scrollbars=False
#)

# Converting from raw intensity to optical density
raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
# raw_od.plot(n_channels=len(raw_od.ch_names), duration=500, show_scrollbars=False)

# Evaluating the quality of the data
# quantify the quality of the coupling between the scalp and the optodes using the scalp coupling index (SCI)
# this method looks for the presence of a prominent synchronous signal in the frequency range of cardiac signals across
# both photodetector signals
sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
#fig, ax = plt.subplots(layout="constrained")
#ax.hist(sci)
#ax.set(xlabel="Scalp Coupling Index", ylabel="Count", xlim=[0, 1])

# all channels with a SCI less than 0.5 as bad -> 6 channels
raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < 0.5)) # does it matter to compute SCI in raw intensity or optical density?

# Motion artifact correction methods
# Temporal derivative distribution repair (tddr) - robust regression, which
# effectively removes baseline shift and spike artifacts without the need for any user-supplied parameters.
corrected_tddr = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)
# plot corrected
# Converting from optical density to haemoglobin
raw_haemo = mne.preprocessing.nirs.beer_lambert_law(corrected_tddr, ppf=6.1)
raw_haemo.plot(n_channels=len(raw_haemo.ch_names),
                duration=500, show_scrollbars=False,show=False)