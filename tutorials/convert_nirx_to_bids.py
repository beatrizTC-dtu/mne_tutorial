import os
import shutil
from collections import Counter
from glob import glob
import mne
from mne_nirs.io.snirf import write_raw_snirf
from mne_bids import write_raw_bids, BIDSPath


### REFACTOR SOURCE DATA DIRECTORY
# Source data should be organized like such
#sub-01/
#    ses-01/
#        nirs/
#    ses-02/
#        nirs/

# WRONG

def refactor_directory_structure(base_directory):
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

            # Create nirs folder inside the subject folder
            nirs_path = os.path.join(subject_path, "nirs")
            if not os.path.exists(nirs_path):
                os.makedirs(nirs_path)

            # Create ses-XX folder based on the visit name
            visit_number = visit_name[-2:]  # Extract '01' or '02' from 'visit01' or 'visit02'
            session_folder = f"ses-{visit_number}"
            session_path = os.path.join(nirs_path, session_folder)
            if not os.path.exists(session_path):
                os.makedirs(session_path)

            # Move the data from the original folder to the new session folder
            original_folder = os.path.join(base_directory, folder)
            for item in os.listdir(original_folder):
                item_path = os.path.join(original_folder, item)
                if os.path.isdir(item_path):  # Move directories (or files) from the original folder
                    shutil.move(item_path, session_path)

            # Optional: Remove the empty original folder
            #os.rmdir(original_folder)


# Usage example
base_directory = r"C:\Datasets\Test-retest study\sourcedata"  # Replace with the path to your data
refactor_directory_structure(base_directory)

# There are the values you are likely to need to change
root = r"C:\Datasets\Test-retest study\sourcedata"  # Where is the raw data directly from the machine stored?
task = "auditory"        # Set to whatever the name of your experiment is
stimulus_duration = {'Control': 5, 'Noise': 5, 'Speech': 5.25}
trigger_info = {'1.0': 'Control',
                '2.0': 'Noise',
                '3.0': 'Speech',
                '4.0': 'Xstart',
                '5.0': 'Xend'}

# Extract information from organized folders containing raw source data - REFACTOR
subject_dirs = os.listdir(os.path.join(root))
subjects_list = [subject_dir.split("-")[0] for subject_dir in subject_dirs if "sub" in subject_dir]
visits_list =[subject_dir.split("-")[-1] for subject_dir in subject_dirs if "ses" in subject_dir]
# Remove duplicates (each subject may come for more than 1 visit)
subjects = list(dict.fromkeys(subjects_list))
#sessions = [None]  # Only a single session per subject, so this optional field is not used
# Create dictionary to store number of visits for each subject
visits_dict = dict(Counter(visits_list))
subj_visits_dict = dict(Counter(subjects_list))
# Check if all participants came the same number of times
n_visits = len(visits_dict.keys())
sessions = [f"{i:02}" for i in range(1,n_visits+1)]
complete_subj_visit = all([value == len(subjects) for value in visits_dict.values()])






# Create a BIDSPath object for handling paths for us
#dataset = BIDSPath(root=root,
#                   task=task,
#                   datatype='nirs',
#                   suffix="nirs")
#                   ##extension=".snirf")

print("\n\nConverting raw NIRx data to BIDS format.\n")
print(f"Processing data stored at {root} = {os.path.abspath(root)}")
print(f"Found subjects: {subjects}")
print(f"Found sessions: {sessions}")
if complete_subj_visit:
    print(f"All {len(subjects)} subject came for {n_visits} sessions.")


def find_subfolder_with_data(sub_dir):
    """This function returns the first directory in a folder"""
    d_name = str(sub_dir.directory)
    return d_name + "/" + next(os.walk(d_name))[1][0]


# Loop over each participant
# Loop over each session (range?)
for sub in subjects:
    for ses in sessions:
        # Find source data
        #b_path = dataset.update(subject=sub, session=ses)
        #f_name = find_subfolder_with_data(b_path)
        f_name = os.path.join(root,f"sub{sub}/ses{ses}/nirs")

        # Read source data
        raw = mne.io.read_raw_nirx(f_name, preload=False)

        # Create BIDS path and write to file
        snirf_path = dataset.fpath
        print(f"  Recoding NIRx data as snirf in temporary location: {snirf_path}")
        write_raw_snirf(raw, snirf_path)

        # Read source data
        raw = mne.io.read_raw_snirf(snirf_path, preload=False)

        raw.annotations.set_durations(stimulus_duration)
        raw.annotations.rename(trigger_info)

        raw.info['line_freq'] = 50  # Hangover from EEG

        # Now we write the correctly formatted files to the base directory (not back in to source data)
        bids_path = dataset.copy().update(root='../')
        write_raw_bids(raw, bids_path, overwrite=True)

        print("  Cleaning up temporary files\n")
        os.remove(snirf_path)