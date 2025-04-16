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

def signal_quality_single_subject(data_path, save_path, task, subject, session):
    print(f"Processing subject {subject}, session {session}...")
    os.makedirs(save_path, exist_ok=True)

    # Create path to file based on experiment info
    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=task,
                         root=data_path,
                         datatype="nirs",
                         suffix="nirs",
                         extension=".snirf")

    # Load data
    print("Loading raw NIRS data from BIDS dataset format")
    raw_intensity = read_raw_bids(bids_path=bids_path, verbose=False)
    raw_intensity.load_data()

    # Rename annotations
    print("Renaming annotations")
    raw_intensity.annotations.rename(
        {"1.0": "Control", "2.0": "Noise", "3.0": "Speech", '4.0': 'XStop', '5.0': 'XStart'})

    # Set durations
    raw_intensity.annotations.set_durations({'Control': 5, 'Noise': 5, 'Speech': 5.25})

    # Get event timings
    print("Extracting event timings...")
    Breaks, _ = mne.events_from_annotations(raw_intensity, {'XStop': 4, 'XStart': 5})
    AllEvents, _ = mne.events_from_annotations(raw_intensity)
    Breaks = Breaks[:, 0] / raw_intensity.info['sfreq']
    LastEvent = AllEvents[-1, 0] / raw_intensity.info['sfreq']


