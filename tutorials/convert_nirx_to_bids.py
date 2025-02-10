import os
import shutil
from collections import Counter
from glob import glob
import mne
from mne_nirs.io.snirf import write_raw_snirf
from mne_bids import write_raw_bids, BIDSPath, read_raw_bids
import numpy as np


### REFACTOR SOURCE DATA DIRECTORY ###
# Source data should be organized like such to be able to convert to BIDS
#sub-01/
#    ses-01/
#        nirs/
#    ses-02/
#        nirs/

def refactor_directory_structure_to_bids(base_directory):
    # List all folders in the base directory
    for folder in os.listdir(base_directory):
        # Check if the folder matches the "SubjectXX-visitXX" pattern
        if 'Subject' in folder:
            subject_name, visit_name = folder.split('-')

            # Create subject directory (sub-XX)
            subject_folder = f"sub-{subject_name[7:]}"  # Extract '01' from 'Subject01'
            subject_path = os.path.join(base_directory, subject_folder)
            if not os.path.exists(subject_path):
                os.makedirs(subject_path)

            # Create ses-XX folder based on the visit name
            visit_number = visit_name[-2:]  # Extract '01' or '02' from 'visit01' or 'visit02'
            session_folder = f"ses-{visit_number}"
            session_path = os.path.join(subject_path, session_folder)
            if not os.path.exists(session_path):
                os.makedirs(session_path)

            # Create nirs folder inside the session folder
            nirs_path = os.path.join(session_path, "nirs")
            if not os.path.exists(nirs_path):
                os.makedirs(nirs_path)

            # Move the data from the original folder to the new session folder
            original_folder = os.path.join(base_directory, folder)
            for item in os.listdir(original_folder):
                item_path = os.path.join(original_folder, item)
                if os.path.exists(item_path):  # Move directories (or files) from the original folder
                    shutil.move(item_path, nirs_path)

            # Optional: Remove the empty original folder
            #os.rmdir(original_folder)

base_directory = r'C:\Datasets\Test-retest refactor\sourcedata'  # Replace with the path to your data
#refactor_directory_structure_to_bids(base_directory)



# These are the values you are likely to need to change
root = r"C:\Datasets\Test-retest refactor\sourcedata"  # Where is the raw data directly from the machine stored?
task = "auditory"        # Set to whatever the name of your experiment is
stimulus_duration = {'Control': 5, 'Noise': 5, 'Speech': 5.25}
trigger_info = {'1.0': 'Control',
                '2.0': 'Noise',
                '3.0': 'Speech',
                '4.0': 'Xstart',
                '5.0': 'Xend'}

# Extract information from organized folders containing raw source data
subject_dirs = glob(os.path.join(root, "sub-*"))
subjects = [subject_dir.split("-")[-1] for subject_dir in subject_dirs] # ["01","02",...]
# Remove duplicates (each subject may come for more than 1 visit)
#subjects = list(dict.fromkeys(subjects_list))
#sessions = [None]  # Only a single session per subject, so this optional field is not used
ses_list = np.array([])
for folder in subject_dirs:
    ses_list = np.append(ses_list,np.array([ses for ses in os.listdir(folder)]))
ses_dict = dict(Counter(ses_list))
# Check if all participants came the same number of times
sessions = list(ses_dict.keys())
sessions =[ses.split("-")[-1] for ses in sessions] # ["01","02",...]
complete_sub_ses = all([value == len(subjects) for value in ses_dict.values()])


# Create a BIDSPath object for handling paths for us
dataset = BIDSPath(root=root,
                   task=task,
                   datatype='nirs',
                   suffix="nirs",
                   extension=".snirf")

print("\n\nConverting raw NIRx data to BIDS format.\n")
print(f"Processing data stored at {root} = {os.path.abspath(root)}")
print(f"Found subjects: {subjects}")
print(f"Found sessions: {sessions}")
if complete_sub_ses:
    print(f"All {len(subjects)} subject came for {len(sessions)} sessions.")


def find_subfolder_with_data(sub_dir):
    """This function returns the first directory in a folder"""
    d_name = str(sub_dir.directory)
    return d_name + "/" + next(os.walk(d_name))[1][0]

### CONVERT NIRX TO BIDS ###

# Loop over each participant "XX"
for sub in subjects:
    # Loop over each session "XX"
    for ses in sessions:
        # Find source data
        b_path = dataset.update(subject=sub, session=ses)
        #f_name = find_subfolder_with_data(b_path)
        #f_name = os.path.join(root,f"sub{sub}/ses{ses}/nirs")
        f_name = b_path.directory

        # Read source data
        raw = mne.io.read_raw_nirx(f_name, preload=False)
        raw.annotations.rename(trigger_info)
        raw.annotations.set_durations(stimulus_duration)

        # Create BIDS path and write to file
        snirf_path = dataset.fpath
        print(f"  Recoding NIRx data as snirf in temporary location: {snirf_path}")
        write_raw_snirf(raw, snirf_path)

        # Read source data
        raw = mne.io.read_raw_snirf(snirf_path, preload=False)

        #raw.annotations.set_durations(stimulus_duration)
        #raw.annotations.rename(trigger_info)

        #raw.info['line_freq'] = 50  # Hangover from EEG

        trigger_code = {'Control':1,
                        'Noise'  :2,
                        'Speech' :3,
                        'Xstart' :4,
                        'Xend'   :5}

        # Now we write the correctly formatted files to the base directory (not back in to source data)
        bids_path = dataset.copy().update(root=r"C:\Datasets\Test-retest study\bids_dataset") # where you will store the bids dataset
        write_raw_bids(raw, bids_path, event_id=trigger_code, overwrite=True)

        print("  Cleaning up temporary files\n")
        os.remove(snirf_path)


### VALIDATE BIDS CONVERSION ###

def load_data(bids_path):

    raw_intensity = read_raw_bids(bids_path=bids_path, verbose=False)
    raw_intensity.load_data()
    return 1


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

        assert load_data(bids_path)