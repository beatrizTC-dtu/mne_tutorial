from IPython import get_ipython
get_ipython().magic('reset -sf')

# Loading packages
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import mne
import mne_nirs
import pandas as pd

from scipy import stats
from scipy.stats import sem
from mne.preprocessing.nirs import (optical_density,
                                    temporal_derivative_distribution_repair)
from itertools import compress
from mne_nirs.channels import (get_long_channels,
                               get_short_channels,
                               picks_pair_to_idx)
from mne.viz import plot_compare_evokeds
from pprint import pprint


#from fooof import FOOOF
from mne_nirs.preprocessing import quantify_mayer_fooof

# Import StatsModels
import statsmodels.formula.api as smf
from mne_nirs.statistics import statsmodels_to_results
# Import Plotting Library
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from itertools import compress
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

from mne_nirs.experimental_design import make_first_level_design_matrix, create_boxcar
from mne_nirs.statistics import run_glm

from nilearn.plotting import plot_design_matrix
from mne_nirs.utils._io import glm_to_tidy

from mne_nirs.statistics import run_GLM, glm_region_of_interest

import pickle
import statsmodels

import sys
sys.path.append(r'C:\GitHub\AudTacLocalizer\Analysis scripts\final pipelines\MNEchanreject')
from preprocessing_and_first_level_MNEbadchanreject import individual_analysis # first level function for both glm and waveforms

sys.path.append(r'C:\GitHub\audiohaptics_exp1\selfwrittenFunctions\functions complementing mne')
from plot_nirs_source_detector_wAxixLims import plot_nirs_source_detector_wAxixLims # first level function for both glm and waveforms
#from evoked2_badchanremoval import plot_compare_evokeds_nobads # too complicated to have parallel universium of these stacked functions -> now mean changed to nanmean in original function

from bad_channels_stats import bad_channels_stats#



"""
Created on Mon Mar 11 15:10:31 2024
@author: aicu

Group analysis script for audio-tactile localizer data:
- Same preprocessing for glm and waveforms with short channels being scaled by separate glm and then subtracted 
- bad channels are cleaned as defined in matlab

- for paper: analyse silent conditions only (dicsussion about T_noise)
ROI analysis
- Waveform and GLM for 
    - single conditions
    - Control subtracted from the rest


"""
# function def
def format_number(val):
    val = pd.to_numeric(val, errors='coerce')
     
    # Apply your formatting conditionally
    if abs(val) < 0.001:
        return '<0.001'
    else:
        return f'{val:.3f}'
    
    
def determine_roi(ch_name, rois_pairs):
    # Extract source and detector from the channel name (e.g., 'S15_D9 hbo')
    parts = ch_name.split('_')
    source = int(parts[0][1:])  # Extract and convert source number
    detector = int(parts[1][1:].split(' ')[0])  # Extract and convert detector number

    # Check each ROI to see if the channel belongs to it
    for roi_name, pairs in rois_pairs.items():
        if any([source == pair[0] and detector == pair[1] for pair in pairs]):
            return roi_name
    return 'Other'


#%% Preprocessing and first level analysis

# stimulus grouping 
stimrenaming = {'1.0': 'Speech_Sil', '2.0':'Noise_Sil', '3.0':'Control_Sil', '4.0':'A_SPIN_Sil', '5.0': 'T_Sil','6.0': 'AT_SPIN_Sil', '7.0': 'Speech_Noi', '8.0': 'Control_Noi'}
T_Noi_rename = {'9.0': 'T_Noi'}

plots = True #False#
glm = True
waveform = True
normalization = False # only for NAHA breathing task is available
# could use that for signal quality check on roi or single channel level: response for breathing in all rois ?
# normalization across channels, also across participants

df_channelstats = pd.DataFrame() 
df_roiglm = pd.DataFrame() 
df_shortchanglm = pd.DataFrame() 
all_evokeds_allCond = defaultdict(list)
all_evokeds_ShortChans = defaultdict(list)

epochrejectCutOffs =[]
bad_channels =[]
percent_bad_epochs = []
SCIs = []
PeakPowers = []
CoefficientsVariaion = []

filelist = [r'01\2023-04-18_002',r'02\2023-04-19_001',r'03\2023-04-21_001',r'04\2023-04-24_001',r'07\2023-05-04_001',r'08\2023-12-18_002',r'09\2023-12-21_001',r'10\2024-01-04_001', r'11\2024-01-09_001',r'12\2024-01-16_003',r'13\2024-01-19_001',r'14\2024-01-19_002',r'15\2024-01-23_001',r'16\2024-01-23_002',r'17\2024-01-24_003',  r'19\2024-01-26_002',  r'20\2024-01-29_001', r'21\2024-01-30_001', r'22\2024-02-01_001', r'23\2024-02-08_001', r'25\2024-02-15_001'] # subject 5 excluded, subject 6 not loadable, 5\2023-04-27_005',,'6\2023-05-02_001', subj 18 trigger problem., subject 24 bad
#filelist = [ r'10\2024-01-04_001'] # subject 5 excluded, subject 6 not loadable, 5\2023-04-27_005',,'6\2023-05-02_001', subj 18 trigger problem., subject 24 bad
filelist = [r'21\2024-01-30_001']


for sub in range(0,len(filelist)): 
    raw_haemo,short_chs, raw_haemo_longandshort, epochs_allCond, channelstats, roiglm, epochs_shortchans, glm_shortchans, epochrejectCutOff, bads_SCI_PP, bad_epochs, sci, PP_scores, ChanStd  = individual_analysis(filelist[sub], plots, stimrenaming, T_Noi_rename, glm, waveform)
        # Save individual-evoked participant data along with others in all_evokeds
    df_channelstats = pd.concat([df_channelstats, channelstats], ignore_index=True)
    df_roiglm = pd.concat([df_roiglm, roiglm], ignore_index=True)
    df_shortchanglm = pd.concat([ df_shortchanglm, glm_shortchans], ignore_index=True)
    
    # Save individual-evoked participant data along with others in all_evokeds
    for cidx, condition in enumerate(epochs_allCond.event_id):
        all_evokeds_allCond[condition].append(epochs_allCond[condition].average())
        
        
    for cidx, condition in enumerate(epochs_shortchans.event_id):
        all_evokeds_ShortChans[condition].append(epochs_shortchans[condition].average())

    epochrejectCutOffs.append(epochrejectCutOff)
    bad_channels.append(bads_SCI_PP)
    percent_bad_epochs.append(bad_epochs)
    SCIs.append(sci) # scalp coulpling index for every channel from every subject
    PeakPowers.append(PP_scores)
    CoefficientsVariaion.append(ChanStd)     
    
# #%% save results

# savepath = r"C:\Users\AICU\OneDrive - Demant\Documents - Eriksholm Research Centre\Research Projects\AudTacLoc\fNIRS results - reanalysis with more subjects\Final files_21_03_24\newMNEbadchanreject\separateConditions_MAINANALYSIS"


#   #    waveforms
# with open(savepath + r'\waveforms\all_evokeds.pkl', 'wb') as file:
#     pickle.dump(all_evokeds_allCond, file)

# with open(savepath + r'\waveforms\all_evokedsShortchanss.pkl', 'wb') as file:
#       pickle.dump(all_evokeds_ShortChans, file)

# raw_haemo.save(savepath + r'\raw_haemo.fif', overwrite= True)

    
#     # glm data frames as pkl
# df_shortchanglm.to_pickle(savepath + r'\glm\df_shortchanglm.pkl')  
# df_roiglm.to_pickle(savepath + r'\glm\df_roiglm.pkl')
# df_channelstats.to_pickle(savepath + r'\glm\df_channelstats.pkl')
# df_shortchanglm.to_pickle

# with open(savepath + r'\waveforms\epochrejectioncutoffs.pkl', 'wb') as file:
#     pickle.dump(epochrejectCutOffs, file)
# with open(savepath + r'\waveforms\percentbadepochs.pkl', 'wb') as file:
#     pickle.dump(percent_bad_epochs, file)
# with open(savepath +  r'\badchannels.pkl', 'wb') as file:
#     pickle.dump(bad_channels, file)
# with open(savepath +  r'\SCIs.pkl', 'wb') as file:
#     pickle.dump(SCIs, file)
# with open(savepath +  r'\PeakPowers.pkl', 'wb') as file:
#     pickle.dump(PeakPowers, file)
# with open(savepath +  r'\CVs.pkl', 'wb') as file:
#     pickle.dump(CoefficientsVariaion, file)
        
            
    
# # as csv
# df_roiglm.to_csv(savepath + r'\glm\df_roiGLM_allcond.csv') 
# df_channelstats.to_csv(savepath + r'\glm\df_channelstats_allcond.csv') 
# df_shortchanglm.to_csv(savepath + r'\glm\df_shortchanglm.csv') 

#%% load
savepath = r"C:\Users\AICU\OneDrive - Demant\Documents - Eriksholm Research Centre\Research Projects\AudTacLoc\fNIRS results - reanalysis with more subjects\Final files_21_03_24\newMNEbadchanreject\separateConditions_MAINANALYSIS"

# load results
with open(savepath + r'\waveforms\all_evokeds.pkl', 'rb') as file:
    all_evokeds_allCond =pickle.load( file)

with open(savepath + r'\glm\df_roiglm.pkl', 'rb') as file:
    df_roiglm =pickle.load( file)
with open(savepath + r'\glm\df_channelstats.pkl', 'rb') as file:
    df_channelstats =pickle.load( file)
    
raw_haemo = mne.io.read_raw_fif(savepath + r'\raw_haemo.fif')

with open(savepath + r'\waveforms\all_evokedsShortchanss.pkl', 'rb') as file:
    all_evokeds_ShortChans =pickle.load(file)

with open(savepath + r'\glm\df_shortchanglm.pkl', 'rb') as file:
    df_shortchanglm =pickle.load( file)


# load also epoch /bad chan stats
with open(savepath + r'\waveforms\epochrejectioncutoffs.pkl', 'rb') as file:
    epochrejectCutOffs = pickle.load(file)
with open(savepath + r'\waveforms\percentbadepochs.pkl', 'rb') as file:
    percent_bad_epochs = pickle.load(file)
with open(savepath +  r'\badchannels.pkl', 'rb') as file:
    bad_channels=  pickle.load(file)


with open(savepath +  r'\SCIs.pkl', 'rb') as file:
    SCIs=  pickle.load(file)
with open(savepath +  r'\PeakPowers.pkl', 'rb') as file:
    PeakPowes =  pickle.load(file)
with open(savepath +  r'\CVs.pkl', 'rb') as file:
    CoefficientsVariation =  pickle.load(file)


# too long path -< save first all results on desktop and copy to other folder later
savepath2 = r"C:\Users\AICU\OneDrive - Demant\Desktop\audtacloc_results_desktop_tempfolder"
#savepath2 = r"C:\Users\AICU\OneDrive - Demant\Desktop\audtacloc_results_desktop_tempfolder/contrastfocus"

#%% reject bads in epoched data
for evoked in all_evokeds_allCond.values():
    for evo in evoked:
        for bad_chan in evo.info['bads']:
            if bad_chan in evo.ch_names:
                idx = evo.ch_names.index(bad_chan)
                evo.data[idx, :] = np.nan # Set all time points for this channel to NaN  


for evoked in all_evokeds_ShortChans.values():
    for evo in evoked:
        for bad_chan in evo.info['bads']:
            if bad_chan in evo.ch_names:
                idx = evo.ch_names.index(bad_chan)
                evo.data[idx, :] = np.nan # Set all time points for this channel to NaN  

#%% calculate bad channel stats
bad_chan_stats = bad_channels_stats(bad_channels, raw_haemo)
# entries mean:    return (total_counts_per_subject, long_counts_per_subject, short_counts_per_subject, mean_percentage_total, mean_percentage_long, mean_percentage_short)

# statistics epoch reject:
meancutoff_mol = np.mean(epochrejectCutOffs) #53 micromol ?!
meanpercent_bad_epochs = np.mean(percent_bad_epochs)
medianpercent_bad_epochs = np.median(percent_bad_epochs)
stdpercent_bad_epochs = np.std(percent_bad_epochs)


# Calculating the number of bad channels for each subject
num_bad_channels_per_subject = [len(sublist) for sublist in bad_channels]
meanbadchans = np.mean(num_bad_channels_per_subject)
stdbadchans = np.std(num_bad_channels_per_subject)
medianbadchans = np.median(num_bad_channels_per_subject)


# Creating the scatter plots
channel_names_reduced = raw_haemo_longandshort.ch_names[::2] #save one raw_haemo /or its channellist form early preprocessing before separating in long and short for these plots

# Prepare the data
flat_SCIs = np.stack(SCIs, axis=1)[::2, :]
flat_PeakPowers = np.stack(PeakPowers, axis=1)[::2, :]
CoefficientsVariation = CoefficientsVariaion
flat_CoefficientsVariation = np.stack(CoefficientsVariation, axis=1)[::2, :]

fig, axes = plt.subplots(3, 1, figsize=(10, 18))

# Plot SCI
for i in range(flat_SCIs.shape[1]):
    axes[0].scatter(channel_names_reduced, flat_SCIs[:, i])
axes[0].axhline(y=0.7, color='r', linestyle='--')  # Rejection threshold
axes[0].set_title('Scalp Coupling Index (SCI) Values')
axes[0].set_xlabel('Channel')
axes[0].set_ylabel('SCI Value')
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend()  # Add a legend to the first subplot


# Plot Peak Powers %redo 
for i in range(flat_PeakPowers.shape[1]):
    axes[1].scatter(channel_names_reduced, flat_PeakPowers[:, i])
axes[1].axhline(y=0.1, color='r', linestyle='--')  # Rejection threshold
axes[1].set_title('Peak Powers')
axes[1].set_xlabel('Channel')
axes[1].set_ylabel('percent of segements with Bad Peak Power (<0.1)')
axes[1].set_ylim([0, 100])
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend()  # Add a legend to the first subplot
# Plot Coefficients of Variation
for i in range(flat_CoefficientsVariation.shape[1]):
    axes[2].scatter(channel_names_reduced, flat_CoefficientsVariation[:, i])
axes[2].axhline(y=0.2, color='r', linestyle='--')  # Rejection threshold
axes[2].set_title('Coefficients of Variation')
axes[2].set_xlabel('Channel')
axes[2].set_ylabel('Coefficient of Variation')
axes[2].set_ylim([0, 0.4])
axes[2].tick_params(axis='x', rotation=45)
axes[2].legend()  # Add a legend to the first subplot

# change to logarithmic scale to see all points but those around cutoff better


from kneed import KneeLocator

data = flat_SCIs #flat_CoefficientsVariation  #flat_PeakPowers #flat_SCIs #
cutoffs = np.linspace(data.min(), data.max(), 100)
percent_below_cutoff = [np.mean(data < cutoff) for cutoff in cutoffs]

# Using KneeLocator to find the knee point
knee_locator = KneeLocator(cutoffs, percent_below_cutoff, curve='convex', direction='increasing')
knee_point = knee_locator.knee

# Finding the 5th and 95th percentile values
lower_5_percentile = np.percentile(data, 5)
upper_5_percentile = np.percentile(data, 95)

# Plotting the results with the knee point and percentiles
plt.figure(figsize=(10, 6))
plt.plot(cutoffs, percent_below_cutoff, label='Percentage of Channels Below Cutoff')
plt.scatter(knee_point, knee_locator.knee_y, color='red', label=f'Knee Point: {knee_point:.2f}', zorder=5)
plt.axvline(x=lower_5_percentile, color='green', linestyle='--', label=f'Lower 5%: {lower_5_percentile:.2f}')
plt.axvline(x=upper_5_percentile, color='blue', linestyle='--', label=f'Upper 5%: {upper_5_percentile:.2f}')
plt.title('Finding Optimal Cutoff using KneeLocator')
plt.xlabel('Cutoff Value')
plt.ylabel('Percentage of Channels Below Cutoff')
plt.legend()
plt.grid(True)


# histograms
flattened_data = data.flatten()

# Create a histogram to show the distribution of SCI values
plt.figure(figsize=(10, 6))
plt.hist(flattened_data, bins=50, color='blue', alpha=0.7)
plt.title('Distribution of SCI Values')
plt.xlabel('SCI Value')
plt.ylabel('Number of Occurrences')
plt.grid(True)
plt.show()


#%% define ROIs for waveforms (make sure to use the same as inside first level function is used for glm)

lIFG = [[8,6],[9,6],[9,7],[10,6],[10,7]]; # 8-6 and 9-6 are the same as in luke
#lSTG = [[13,9],[13,13],[14,9],[14,13],[14,14]] # exactly as in robs paper
lSTG = [[13,9],[13,13],[14,9],[14,13],[14,14],[15,10],[15,9],[15,14],[14,15],[14,10]] # exactly as in robs paper + extra ones inbetween
#rSTG = [[7,5],[5,5],[3,5],[5,2],[3,2]] # as in luke except one chan that we don't have
rSTG = [[7,5],[5,5],[3,5],[5,2],[3,2],[4,4],[4,5],[3,4]]# 6_4 and 7_4 additonal channels in our montage (posterior)

bilatSTG = lSTG + rSTG
lSC =[[9,9],[11,8],[11,9],[11,11],[11,12],[12,8],[12,12],[12,11]]
rSC =[[2,3],[4,3],[6,3]]
bilatSC= lSC + rSC
lpostTemp = [[15,12], [15,14],[15,15],[14,14],[14,15],[14,13],[16,13],[16,14],[14,11]] # maybe not 14,11


rois = dict(lSTG=picks_pair_to_idx(raw_haemo, lSTG, on_missing ='ignore'),
        rSTG =picks_pair_to_idx(raw_haemo, rSTG, on_missing ='ignore'),
        bilateralSTG =picks_pair_to_idx(raw_haemo, bilatSTG, on_missing ='ignore'),
        Left_SC=picks_pair_to_idx(raw_haemo, lSC, on_missing ='ignore'),
        Right_SC=picks_pair_to_idx(raw_haemo, rSC, on_missing ='ignore'),
        #Bilateral_SC=picks_pair_to_idx(raw_haemo, bilatSC, on_missing ='ignore'),
        leftIFG=picks_pair_to_idx(raw_haemo, lIFG, on_missing ='ignore'),
        leftPostTemp=picks_pair_to_idx(raw_haemo, lpostTemp, on_missing ='ignore'))



#%% short chan analysis-> in the other script
#%% Long channels analysis starts here
#%% 0. all channels together, plots, overview condition activations, plots, but no stats
#first look at data, plot conditions for all channels together

selected_conditions =  [ 'Noise_Sil','Speech_Sil','A_SPIN_Sil','T_Sil','AT_SPIN_Sil','Control_Sil'] 

fig, axes = plt.subplots(nrows=1, ncols=len(selected_conditions), figsize=(20, 4))
#fig, axes = plt.subplots(ncols=1, nrows=len(selected_conditions), figsize=(3, 8))
lims = dict(hbo=[-1 ,3], hbr=[-1, 3])

for (pick, color) in zip(['hbo', 'hbr'], ['r', 'b']):
    for idx, evoked in enumerate(selected_conditions ):

        plot_compare_evokeds({evoked: all_evokeds_allCond[evoked]}, combine='mean',
                             picks=pick, axes=axes[idx], show=False,
                             colors=[color], legend=False, ylim=lims, ci=0.95,
                             show_sensors=idx == 1)
        axes[idx].set_title('{}'.format(evoked))
#axes[0].legend(["Oxyhaemoglobin","95% CI", "Stimulus onset","Deoxyhaemoglobin"])
        axes[idx].set_xlim([-5, 20])  # Setting x-axis range
        axes[idx].set_xticks(np.arange(-5, 20, 5))  # Setting y-ticks to full numbers
        axes[idx].set_ylim([-1, 3])  # Adjusting y-axis limits
        axes[idx].set_yticks(np.arange(-1, 3, 1))  # Setting y-ticks to full numbers only
        axes[idx].grid(False)  # Turning off the grid
    

fig.savefig(savepath2 + r"\waveforms\longchans_waveform_allSilconditions.svg", format='svg')       


## waveform ROI plots

selected_conditions =  [ 'Speech_Sil','Noise_Sil','T_Sil'] 
#selected_conditions =  ['Noise_Sil','T_Sil'] 
selected_rois = dict(bilateralSTG =picks_pair_to_idx(raw_haemo, bilatSTG, on_missing ='ignore'),
        Bilateral_SC=picks_pair_to_idx(raw_haemo, bilatSC, on_missing ='ignore'))

selected_rois = dict(lSTG=picks_pair_to_idx(raw_haemo, lSTG, on_missing ='ignore'),
        rSTG =picks_pair_to_idx(raw_haemo, rSTG, on_missing ='ignore'),
        Left_SC=picks_pair_to_idx(raw_haemo, lSC, on_missing ='ignore'),
        Right_SC=picks_pair_to_idx(raw_haemo, rSC, on_missing ='ignore'))



# Specify the figure size and limits per chromophore.
fig, axes = plt.subplots(nrows=len(selected_rois), ncols=len(selected_conditions),
                         figsize=(10, 10))
lims = dict(hbo=[-2.5, 5.5], hbr=[-2.5, 5.5])

for (pick, color) in zip(['hbo', 'hbr'], ['r', 'b']):
    for ridx, roi in enumerate(selected_rois):
        for cidx, evoked in enumerate(selected_conditions):
            if pick == 'hbr':
                picks = selected_rois[roi][1::2]  # Select only the hbr channels
            else:
                picks = selected_rois[roi][0::2]  # Select only the hbo channels

            plot_compare_evokeds({evoked: all_evokeds_allCond[evoked]}, combine='mean',
                                 picks=picks, axes=axes[ridx, cidx],
                                 show=False, colors=[color], legend=False,
                                 ylim=lims, ci=0.95, show_sensors=cidx == 2)
            axes[ridx, cidx].set_title("")
        axes[0, cidx].set_title(f"{evoked}")
        axes[ridx, 0].set_ylabel(f"{roi}\nChromophore (ΔμMol)")
axes[0, 0].legend(["Oxyhaemoglobin", "Deoxyhaemoglobin"]) 
#fig.savefig(savepath2 + r"\waveforms\longchans_waveform_allSilconditionsROIS.svg", format='svg')       



## glm equivlaent to waveform plot: bars for whole montage single condition activations

  
grp_results = df_roiglm.query("Condition in ['Noise_Sil','Speech_Sil','A_SPIN_Sil','T_Sil','AT_SPIN_Sil','Control_Sil']")
grp_results = grp_results.query("Chroma in ['hbo','hbr']")
roi_glmmodel = smf.mixedlm("theta ~ -1 + Condition:Chroma", 
                        grp_results, groups=grp_results["ID"]).fit() # method='nm')
roi_glmmodel.summary()
df_lme_glmlong = statsmodels_to_results(roi_glmmodel)



condition_order = ['Noise_Sil', 'Speech_Sil', 'A_SPIN_Sil', 'T_Sil', 'AT_SPIN_Sil', 'Control_Sil']
df_lme_glmlong['Condition'] = pd.Categorical(df_lme_glmlong['Condition'], categories=condition_order, ordered=True)

# Sort the DataFrame by the 'condition' column to ensure the plot follows the specified order
df_lme_glmlong = df_lme_glmlong.sort_values('Condition')

df_lme_glmlong['error_lower'] = df_lme_glmlong['Coef.'] - df_lme_glmlong['[0.025']
df_lme_glmlong['error_upper'] = df_lme_glmlong['0.975]'] - df_lme_glmlong['Coef.']
errors = df_lme_glmlong[['error_lower', 'error_upper']].T.values

# Creating the bar plot
plt.figure(figsize=(10, 6))
plt.bar(df_lme_glmlong['Condition'], df_lme_glmlong['Coef.'], color='lightcoral', yerr=errors, capsize=5) # color='darkred'
plt.xlabel('Condition')
plt.ylabel('Coefficient')
plt.title('HbO Beta estimates for the whole montage')
plt.xticks(rotation=45)


df_lme_glmlong['Condition'] = pd.Categorical(df_lme_glmlong['Condition'], categories=condition_order, ordered=True)


# Split the DataFrame into HbO and HbR
df_hbo = df_lme_glmlong[df_lme_glmlong['Chroma'] == 'hbo']
df_hbr = df_lme_glmlong[df_lme_glmlong['Chroma'] == 'hbr']


# Creating the bar plots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)

# Plot for HbO
axes[0].bar(df_hbo['Condition'], df_hbo['Coef.'], color='darkred', capsize=5)
axes[0].set_title('HbO Beta estimates per Condition')
axes[0].set_ylabel('Coefficient')
axes[0].set_ylim([-0.5,2.5])
axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)

# Plot for HbR
axes[1].bar(df_hbr['Condition'], df_hbr['Coef.'], color='darkblue', capsize=5)
axes[1].set_title('HbR Beta estimates per Condition')
axes[1].set_xlabel('Condition')
axes[1].set_ylabel('Coefficient')
axes[1].set_ylim([-0.5,2.5])
axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)

plt.xticks(rotation=45)


plt.figure(figsize=(12, 6))
ax = sns.barplot(x='Condition', y='Coef.', hue='Chroma', data=df_lme_glmlong,
                 palette={'hbo': 'firebrick', 'hbr': 'mediumblue'}, ci=None, capsize=.1) #'indianred'

# Customize the plot with labels and title
plt.xlabel('Condition')
plt.ylabel('Coefficient')
plt.title('Beta estimates per Condition for HbO and HbR')
plt.xticks(rotation=45)
plt.legend(title='Chroma')



#%% data frame waveforms longchans

        # stats: generate data frame from waveform responses
df_waveformamplitudes = pd.DataFrame(columns=['ID', 'ROI', 'Chroma', 'Condition', 'Value'])

for idx, evoked in enumerate(all_evokeds_allCond):
    for subj_data in all_evokeds_allCond[evoked]:
        subj_id = subj_data.info['subject_info']['his_id'] if 'subject_info' in subj_data.info and 'his_id' in subj_data.info['subject_info'] else 'unknown'
        bads = subj_data.info['bads']
        for roi, picks in rois.items():
            for chroma in ["hbo", "hbr"]:
                   good_picks = [pick for pick in picks if subj_data.ch_names[pick] not in bads]
                   if good_picks:  # Proceed only if there are good channels
                       data = deepcopy(subj_data).pick(picks=good_picks).pick(chroma)
                       if len(data.ch_names) > 0:
                           value = data.crop(tmin=5.0, tmax=8.0).data.mean() * 1.0e6
                           this_df = pd.DataFrame({'ID': subj_id, 'ROI': roi, 'Chroma': chroma, 'Condition': evoked, 'Value': value}, index=[0])
                           df_waveformamplitudes = pd.concat([df_waveformamplitudes, this_df], ignore_index=True)

df_waveformamplitudes.reset_index(inplace=True, drop=True)
df_waveformamplitudes['Value'] = pd.to_numeric( df_waveformamplitudes['Value'])  # some Pandas have this as object


# Extract digits and convert to int
df_waveformamplitudes['ID'] = df_waveformamplitudes['ID'].str.extract('(\d+)').astype(int)
df_waveformamplitudes['ID'] = df_waveformamplitudes['ID'].replace(0, 25)

##

# # see all subjects (--> plot from joerg with waveforms or this): instead of this scatter plots, violing + scatter also with compared to betas
# grp_results = df_waveformamplitudes.query("Condition in ['Speech_Sil','Noise_Sil', 'Control_Sil', 'A_SPIN_Sil', 'T_Sil','AT_SPIN_Sil', 'Speech_Noi', 'Control_Noi', 'T_Noi']")
# grp_results = grp_results.query("Chroma in ['hbo']")

# fig = sns.catplot(x="ID", y="Value", col="Condition",  hue="ROI", data=grp_results, col_wrap=2, errorbar=None, palette="muted", height=9, s=60)
# fig.savefig(savepath + r"\waveforms\longchans_scatter.svg", format='svg')


##### 0.0 whole montage analysis


 ###################0.1c: A_SPIN vs ASPIQ
 
 ################## 0.2  AT vs. A, T for all channels/ROIs together 

df_waveformamplitudes_Rename = df_waveformamplitudes.copy() # it changes the original df otherwise..!!
df_waveformamplitudes_Rename['Condition'] = df_waveformamplitudes_Rename['Condition'].replace('AT_SPIN_Sil', 'AAT_SPIN_Sil')
df_waveformamplitudes_Rename['Condition'] = df_waveformamplitudes_Rename['Condition'].replace('T_Sil', 'B_T_Sil')


 ###################0.1a: A NOise, A_SPIQ - TSPIQ 
input_data = df_waveformamplitudes_Rename.query("Condition in ['Speech_Sil',  'B_T_Sil']")  
## and : A NOise - TSPIQ 
input_data = df_waveformamplitudes_Rename.query("Condition in ['Noise_Sil',  'B_T_Sil']")  

 ###################0.1b: A_Noise vs ASPIQ
input_data = df_waveformamplitudes_Rename.query("Condition in ['Speech_Sil', 'Noise_Sil']")  #'T'

 ###################0.1c: A_SPIN vs ASPIQ
input_data = df_waveformamplitudes_Rename.query("Condition in ['Speech_Sil', 'A_SPIN_Sil']")  #'T'


 ###################0.2: AT_SPIN vs A-SPIn,
input_data = df_waveformamplitudes_Rename.query("Condition in ['AAT_SPIN_Sil', 'A_SPIN_Sil']")  #'T'
 #                  0.2: AT_SPIN vs T-SPIQ
input_data = df_waveformamplitudes_Rename.query("Condition in ['AAT_SPIN_Sil', 'B_T_Sil']")  #'T'

#input_data = df_waveformamplitudes.query("Condition in ['AT_SPIN_Sil', 'A_SPIN_Sil']")  #'T
input_data = input_data.query("Chroma in ['hbo']")
#input_data = input_data.query("ROI in ['leftPostTemp']") # ['lSTG',  'rSTG' , 'Left_SC',   'Right_SC',  'leftIFG'] ['leftPostTemp']
roi_model = smf.mixedlm("Value ~ Condition", input_data,
                        groups=input_data["ID"]).fit(method='nm')
roi_model.summary()
df_lme_waveslongchans= statsmodels_to_results(roi_model)
# df_lme_waveslongchans.reset_index(drop=True, inplace=True)
# df_lme_waveslongchans = df_lme_waveslongchans.set_index(['ROI', 'Condition'])

# Filter rows where the index contains ':'
#df_lme_waveslongchans= df_lme_waveslongchans[df_lme_waveslongchans.index.str.contains(':')]


# FDR correction
pvals = df_lme_waveslongchans['P>|z|'].values
pvals_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)[1]

# Add the corrected p-values back to the dataframe
df_lme_waveslongchans['p_FDRcorrected'] = pvals_corrected
df_lme_waveslongchans = df_lme_waveslongchans.rename(columns={'Coef.': 'Amplitude', 'P>|z|': 'p-value'})
df_lme_waveslongchans = df_lme_waveslongchans.drop(columns=['Std.Err.', 'z', '[0.025', '0.975]', 'Significant'])
df_lme_waveslongchans['significant?'] = df_lme_waveslongchans['p_FDRcorrected'] <= 0.05


# Apply this function to the specific columns you want to format
df_lme_waveslongchans['Amplitude'] = df_lme_waveslongchans['Amplitude'].apply(lambda x: format_number(x))
df_lme_waveslongchans['p-value'] = df_lme_waveslongchans['p-value'].apply(lambda x: format_number(x))
df_lme_waveslongchans['p_FDRcorrected'] = df_lme_waveslongchans['p_FDRcorrected'].apply(lambda x: format_number(x))

df_lme_waveslongchans['Amplitude'] = pd.to_numeric(df_lme_waveslongchans['Amplitude'], errors='coerce')

#df_lme_waveslongchans.to_csv(savepath2 + r'\waveforms\df_waveforms_ATvsAvsT_wholecap.csv')

########################################## same for beta


df_roiglm_Rename = df_roiglm.copy()
df_roiglm_Rename['Condition'] = df_roiglm_Rename['Condition'].replace('AT_SPIN_Sil', 'AAT_SPIN_Sil')
df_roiglm_Rename['Condition'] = df_roiglm_Rename['Condition'].replace('T_Sil', 'B_T_Sil')

 ###################0.1a: A_SPIQ - TSPIQ 
input_data = df_roiglm_Rename.query("Condition in ['Speech_Sil', 'B_T_Sil']")  #'T'
#                  Noise - TSPIQ 
input_data = df_roiglm_Rename.query("Condition in ['Noise_Sil', 'B_T_Sil']")  #'T'

 ###################0.1b: A_Noise vs ASPIQ
input_data = df_roiglm_Rename.query("Condition in ['Speech_Sil', 'Noise_Sil']")  #'T'

 ###################0.1c: A_SPIN vs ASPIQ
input_data = df_roiglm_Rename.query("Condition in ['Speech_Sil', 'A_SPIN_Sil']")  #'T'
 ###################0.2: AT_SPIN vs A-SPIn, T-SPIQ
input_data = df_roiglm_Rename.query("Condition in ['AAT_SPIN_Sil', 'A_SPIN_Sil','B_T_Sil']") # 'T'




input_data = input_data.query("Chroma in ['hbo']")
#input_data = input_data.query("ROI in ['lSTG',  'rSTG' , 'Left_SC',   'Right_SC',  'leftIFG', 'leftPostTemp']") # 'lSTG',  'rSTG' , 'Left_SC',   'Right_SC',  'leftIFG', 'leftPostTemp'
roi_model = smf.mixedlm("theta ~ Condition", input_data,
                        groups=input_data["ID"]).fit(method='nm')
roi_model.summary()
df_lme_GLMROIlongchans= statsmodels_to_results(roi_model)

# FDR correction  Multiple comparison correction of LME outputs
pvals = df_lme_GLMROIlongchans['P>|z|'].values
pvals_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)[1]

# Add the corrected p-values back to the dataframe
df_lme_GLMROIlongchans['p_FDRcorrected'] = pvals_corrected
df_lme_GLMROIlongchans = df_lme_GLMROIlongchans.rename(columns={'Coef.': 'Beta', 'P>|z|': 'p-value'})
df_lme_GLMROIlongchans = df_lme_GLMROIlongchans.drop(columns=['Std.Err.', 'z', '[0.025', '0.975]', 'Significant'])
df_lme_GLMROIlongchans['significant?'] = df_lme_GLMROIlongchans['p_FDRcorrected'] <= 0.05


# Apply this function to the specific columns you want to format
df_lme_GLMROIlongchans['Beta'] = df_lme_GLMROIlongchans['Beta'].apply(lambda x: format_number(x))
df_lme_GLMROIlongchans['p-value'] = df_lme_GLMROIlongchans['p-value'].apply(lambda x: format_number(x))
df_lme_GLMROIlongchans['p_FDRcorrected'] = df_lme_GLMROIlongchans['p_FDRcorrected'].apply(lambda x: format_number(x))


df_lme_GLMROIlongchans['Beta'] = pd.to_numeric(df_lme_GLMROIlongchans['Beta'], errors='coerce')
#df_lme_GLMROIlongchans.to_csv(savepath2 + r'\glm\ATvsAvsT_wholeCAP.csv')


#%%  (1)	 To identify brain areas respond to auditory, tactile and audio-tactile speech stimuli.
###--> 1. Research aim: find ROIs based on ROI glm and ROI waveform amplitudes, relative to control
        
grp_results = df_roiglm.query("Condition in ['Speech_Sil','Control_Sil', 'A_SPIN_Sil', 'T_Sil','AT_SPIN_Sil', 'Speech_Noi', 'Control_Noi', 'T_Noi']")
grp_results = grp_results.query("ROI in ['bilateralSTG']")
#grp_results = df_roiglm.query("Condition in ['Speech_Sil','Noise_Sil', 'Control_Sil', 'A_SPIN_Sil', 'T_Sil','AT_SPIN_Sil', 'Speech_Noi', 'Control_Noi', 'T_Noi']")
grp_results = grp_results.query("Chroma in ['hbo']")
roi_glmmodel = smf.mixedlm("theta ~ -1 + Condition:ROI", 
                        grp_results, groups=grp_results["ID"]).fit(method='nm')
roi_glmmodel.summary()
df_lme_glmlong = statsmodels_to_results(roi_glmmodel)


# ## relative to control
# df_roiglm_ControlRename = df_roiglm
# df_roiglm_ControlRename['Condition'] = df_roiglm_ControlRename['Condition'].replace('Control_Sil', 'AA_Control_Sil')
# #grp_results = df_roiglm.query("Condition in ['Speech_Sil','Noise_Sil', 'AA_Control_Sil', 'A_SPIN_Sil', 'T_Sil','AT_SPIN_Sil']")
# grp_results = df_roiglm.query("Condition in ['Speech_Sil', 'AA_Control_Sil', 'A_SPIN_Sil', 'T_Sil','AT_SPIN_Sil']")
# grp_results = grp_results.query("Chroma in ['hbo']")
# grp_results = grp_results.query("ROI in ['lSTG',  'rSTG' , 'Left_SC',   'Right_SC',  'leftIFG']") #grp_results = grp_results.query("ROI in ['lSTG']")
# roi_glmmodel = smf.mixedlm("theta ~ Condition:ROI", 
#                         grp_results, groups=grp_results["ID"]).fit(method='nm')
# roi_glmmodel.summary()
# df_lme_glmlong = statsmodels_to_results(roi_glmmodel)


# FDR correction
pvals = df_lme_glmlong['P>|z|'].values
pvals_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)[1]

# Add the corrected p-values back to the dataframe
df_lme_glmlong['p_FDRcorrected'] = pvals_corrected
df_lme_glmlong = df_lme_glmlong.rename(columns={'Coef.': 'Beta', 'P>|z|': 'p-value'})
df_lme_glmlong = df_lme_glmlong.drop(columns=['Std.Err.', 'z', '[0.025', '0.975]', 'Significant'])
df_lme_glmlong['significant?'] = df_lme_glmlong['p_FDRcorrected'] <= 0.05

# Apply this function to the specific columns you want to format
df_lme_glmlong['Beta'] = df_lme_glmlong['Beta'].apply(lambda x: format_number(x))
df_lme_glmlong['p-value'] = df_lme_glmlong['p-value'].apply(lambda x: format_number(x))
df_lme_glmlong['p_FDRcorrected'] = df_lme_glmlong['p_FDRcorrected'].apply(lambda x: format_number(x))

df_lme_glmlong['Beta'] = pd.to_numeric(df_lme_glmlong['Beta'], errors='coerce').round(1)


#df_lme_glmlong.to_csv(savepath2 + r'\glm\df_roiglm_lme_longchans_relative control.csv')


# reshape to better reportable form (by hand in word together with waveform results..)


# samw scatter plot as befor but now for thetas instead of mean or peak vals
grp_results = df_roiglm.query("Condition in ['Speech_Sil','Noise_Sil', 'AA_Control_Sil', 'A_SPIN_Sil', 'T_Sil','AT_SPIN_Sil']") # 'Speech_Noi', 'Control_Noi', 'T_Noi'
grp_results = grp_results.query("Chroma in ['hbo']")

fig = sns.catplot(x="ID", y="theta", col="Condition",  hue="ROI", data=grp_results, col_wrap=2, errorbar=None, palette="muted", height=9, s=60)



# same for waveforms:
    
    # single conditions: use for plots only
df_waveformamplitudes_ControlRename=  df_waveformamplitudes.copy()
df_waveformamplitudes_ControlRename['Condition'] = df_waveformamplitudes_ControlRename['Condition'].replace('Control_Sil', 'AA_Control_Sil')
input_data = df_waveformamplitudes_ControlRename.query("Condition in ['Speech_Sil','AA_Control_Sil', 'A_SPIN_Sil', 'T_Sil','AT_SPIN_Sil']") #'Noise_Sil', 
input_data = input_data.query("Chroma in ['hbo']")
input_data = input_data.query("ROI in ['bilateralSTG' ]") #, 'Left_SC',   'Right_SC',  'leftIFG'
#roi_model = smf.mixedlm("Value ~ Condition:ROI", 
roi_model = smf.mixedlm("Value ~-1+  Condition:ROI", input_data, #roi_model = smf.mixedlm("Value ~ -1 +Condition:ROI", input_data,
                        groups=input_data["ID"]).fit(method='nm')
roi_model.summary()
df_lme_waveslongchans= statsmodels_to_results(roi_model)
#     
    
# # relative to control
# df_waveformamplitudes_ControlRename=  df_waveformamplitudes.copy()
# df_waveformamplitudes_ControlRename['Condition'] = df_waveformamplitudes_ControlRename['Condition'].replace('Control_Sil', 'AA_Control_Sil')
# input_data = df_waveformamplitudes_ControlRename.query("Condition in ['Speech_Sil','Noise_Sil', 'AA_Control_Sil', 'A_SPIN_Sil', 'T_Sil','AT_SPIN_Sil']")
# input_data = input_data.query("Chroma in ['hbo']")
# input_data = input_data.query("ROI in ['lSTG',  'rSTG' , 'Left_SC',   'Right_SC',  'leftIFG']")
# #roi_model = smf.mixedlm("Value ~ Condition:ROI", 
# roi_model = smf.mixedlm("Value ~ Condition:ROI", input_data, #roi_model = smf.mixedlm("Value ~ -1 +Condition:ROI", input_data,
#                         groups=input_data["ID"]).fit(method='nm')
# roi_model.summary()
# df_lme_waveslongchans= statsmodels_to_results(roi_model)
# # df_lme_waveslongchans.reset_index(drop=True, inplace=True)
# # df_lme_waveslongchans = df_lme_waveslongchans.set_index(['ROI', 'Condition'])


# FDR correction
pvals = df_lme_waveslongchans['P>|z|'].values
pvals_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)[1]

# Add the corrected p-values back to the dataframe
df_lme_waveslongchans['p_FDRcorrected'] = pvals_corrected
df_lme_waveslongchans = df_lme_waveslongchans.rename(columns={'Coef.': 'Amplitude', 'P>|z|': 'p-value'})
df_lme_waveslongchans = df_lme_waveslongchans.drop(columns=['Std.Err.', 'z', '[0.025', '0.975]', 'Significant'])
df_lme_waveslongchans['significant?'] = df_lme_waveslongchans['p_FDRcorrected'] <= 0.05


# Apply this function to the specific columns you want to format
df_lme_waveslongchans['Amplitude'] = df_lme_waveslongchans['Amplitude'].apply(lambda x: format_number(x))
df_lme_waveslongchans['p-value'] = df_lme_waveslongchans['p-value'].apply(lambda x: format_number(x))
df_lme_waveslongchans['p_FDRcorrected'] = df_lme_waveslongchans['p_FDRcorrected'].apply(lambda x: format_number(x))

df_lme_waveslongchans['Amplitude'] = pd.to_numeric(df_lme_waveslongchans['Amplitude'], errors='coerce').round(1)
#df_lme_waveslongchans.to_csv(savepath + r'\waveforms\df_waveforms_RQ1SpeechNoiseControl.csv')
#df_lme_waveslongchans.to_csv(savepath2 +  r'\waveforms\df_waveforms_lme_longchans_realtiveControl.csv')


# # see all subjects (--> plot from joerg or this)
# grp_results = df_waveformamplitudes.query("Condition in ['Speech_Sil','Noise_Sil', 'AA_Control_Sil', 'A_SPIN_Sil', 'T_Sil','AT_SPIN_Sil']")
# grp_results = grp_results.query("Chroma in ['hbo']")

# fig = sns.catplot(x="ID", y="Value", col="Condition",  hue="ROI", data=grp_results, col_wrap=2, errorbar=None, palette="muted", height=9, s=60)


## scatter plot



# Filter out the data for 'hbo' only
df_hbo = df_waveformamplitudes[df_waveformamplitudes['Chroma'] == 'hbo']

# Define the order of Conditions to be displayed on the x-axis
condition_order =  ['Speech_Sil','A_SPIN_Sil','T_Sil','AT_SPIN_Sil','Control_Sil'] 

# Plotting one subplot for each condition
fig, axes = plt.subplots(nrows=1, ncols=len(condition_order), figsize=(20, 5), sharey=True)

for idx, condition in enumerate(condition_order):
    condition_data = df_hbo[df_hbo['Condition'] == condition]
    
    # Combine violin plot (density) and strip plot (scatter) for the raincloud effect
   # sns.violinplot(x='ROI', y='Value', data=condition_data, ax=axes[idx], inner=None, color=".8")
    sns.stripplot(x='ROI', y='Value', data=condition_data, ax=axes[idx], jitter=True, size=4, palette="Set2", linewidth=0)

    axes[idx].set_title(f'Condition: {condition}')
    axes[idx].set_xlabel('ROI')
    axes[idx].set_ylabel('Amplitude' if idx == 0 else '')  # Set y-label only on the first subplot




# including also glm results




# Define the ROIs to include in the plot
included_rois = ['bilateralSTG'] #, 'Left_SC', 'Right_SC', 'leftIFG'


# Filter out the data for 'hbo' only and for the included ROIs
df_hbo = df_waveformamplitudes[(df_waveformamplitudes['Chroma'] == 'hbo') & (df_waveformamplitudes['ROI'].isin(included_rois))]
df_roiglm_hbo = df_roiglm[(df_roiglm['Chroma'] == 'hbo') & (df_roiglm['ROI'].isin(included_rois))]

# Define the order of Conditions to be displayed on the x-axis
condition_order = ['Speech_Sil', 'A_SPIN_Sil', 'T_Sil', 'AT_SPIN_Sil', 'Control_Sil']


# Plotting one subplot for each condition
fig, axes = plt.subplots(nrows=1, ncols=len(condition_order), figsize=(20, 5), sharey='row')

for idx, condition in enumerate(condition_order):
    condition_data_amp = df_hbo[df_hbo['Condition'] == condition]
    condition_data_theta = df_roiglm_hbo[df_roiglm_hbo['Condition'] == condition]
    
    ax1 = axes[idx]
    # Offset for separation
    amplitude_offset = np.full(condition_data_amp.shape[0], -0.1)  # Slight left shift
    theta_offset = np.full(condition_data_theta.shape[0], 0.1)  # Slight right shift
    
    # Jitter and shift positions for amplitude and theta for clarity
    sns.stripplot(x='ROI', y='Value', data=condition_data_amp, ax=ax1, jitter=True, size=4, color="darkcyan", linewidth=0)
    ax2 = ax1.twinx()
    sns.stripplot(x='ROI', y='theta', data=condition_data_theta, ax=ax2, jitter=True, size=4, color="orchid", linewidth=0)


    ax1.set_title(f'Condition: {condition}')
    ax1.set_xlabel('ROI')
    ax1.set_ylabel('Amplitude')
    ax2.set_ylabel('Theta')

    # Rotate x-axis labels
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")

    # Set y-axis limits uniformly across all axes
    ax1.set_ylim(-5, 12)
    ax2.set_ylim(-5, 12)

# Create legend with custom handles
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color="darkcyan", lw=4),
                Line2D([0], [0], color="orchid", lw=4)]
axes[0].legend(custom_lines, ['Amplitude', 'Theta'], loc='upper left')




# Filter out the data for 'hbo' only and for the included ROIs
df_hbo = df_waveformamplitudes[(df_waveformamplitudes['Chroma'] == 'hbo') & (df_waveformamplitudes['ROI'].isin(included_rois))]
df_roiglm_hbo = df_roiglm[(df_roiglm['Chroma'] == 'hbo') & (df_roiglm['ROI'].isin(included_rois))]

# Define the order of Conditions to be displayed on the x-axis
condition_order = ['Noise_Sil', 'Speech_Sil', 'A_SPIN_Sil', 'T_Sil', 'AT_SPIN_Sil', 'Control_Sil']

# Plotting one subplot for each condition
fig, axes = plt.subplots(nrows=1, ncols=len(condition_order), figsize=(20, 5), sharey='row')

for idx, condition in enumerate(condition_order):
    ax = axes[idx]
    condition_data_amp = df_hbo[df_hbo['Condition'] == condition]
    condition_data_theta = df_roiglm_hbo[df_roiglm_hbo['Condition'] == condition]

    # Create a data frame that can be used for violin plots
    condition_data_amp['Type'] = 'Amplitude'
    condition_data_theta['Type'] = 'Theta'
    combined_data = pd.concat([condition_data_amp, condition_data_theta])
    combined_data.rename(columns={'Value': 'Amplitude', 'theta': 'Theta'}, inplace=True)

    # Melt the data for seaborn plotting
    melt_data = combined_data.melt(id_vars=['ROI', 'Type'], value_vars=['Amplitude', 'Theta'],
                                   var_name='Measurement', value_name='Value')

    # Plot the violin plot
    sns.violinplot(x='ROI', y='Value', hue='Type', data=melt_data, split=True, inner=None, linewidth=1.5,
                   palette={'Amplitude': 'cadetblue', 'Theta': 'navy'}, ax=ax) #inner='quart'

    # Customize plot
    ax.set_title(f'Condition: {condition}')
    ax.set_xlabel('ROI')
    ax.set_ylabel('Measurement Value')
    ax.legend(title='Type')

    # Rotate x-axis labels for better visibility
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)


### in one plot: scatter
    

# Define the ROIs to include in the plot
included_rois = ['bilateralSTG'] #, 'Left_SC', 'Right_SC', 'leftIFG'

# Filter out the data for 'hbo' only and for the included ROIs
df_hbo = df_waveformamplitudes[(df_waveformamplitudes['Chroma'] == 'hbo') & (df_waveformamplitudes['ROI'].isin(included_rois))]
df_roiglm_hbo = df_roiglm[(df_roiglm['Chroma'] == 'hbo') & (df_roiglm['ROI'].isin(included_rois))]

# Define the order of Conditions to be displayed on the x-axis
condition_order = ['Speech_Sil', 'A_SPIN_Sil', 'T_Sil', 'AT_SPIN_Sil', 'Control_Sil']

# Create a combined dataframe with 'Amplitude' and 'Theta' values
df_hbo['Type'] = 'Amplitude'
df_roiglm_hbo['Type'] = 'Theta'
df_roiglm_hbo.rename(columns={'theta': 'Value'}, inplace=True)

combined_data = pd.concat([df_hbo, df_roiglm_hbo], ignore_index=True)

# Ensure there are no duplicate labels by resetting index
combined_data.reset_index(drop=True, inplace=True)

# Plotting all conditions in one plot
plt.figure(figsize=(15, 8))

sns.stripplot(x='Condition', y='Value', hue='Type', data=combined_data, order=condition_order, jitter=True, size=4, palette={'Amplitude': 'darkcyan', 'Theta': 'orchid'})

# Customize plot
plt.title('Amplitude and Theta Values across Different Conditions')
plt.xlabel('Condition')
plt.ylabel('Measurement Value')
plt.xticks(rotation=45)
plt.ylim(-5, 12)
plt.legend(title='Type')
    

# violins:
    # Define the ROIs to include in the plot
included_rois = ['bilateralSTG'] #, 'Left_SC', 'Right_SC', 'leftIFG'

# Filter out the data for 'hbo' only and for the included ROIs
df_hbo = df_waveformamplitudes[(df_waveformamplitudes['Chroma'] == 'hbo') & (df_waveformamplitudes['ROI'].isin(included_rois))]
df_roiglm_hbo = df_roiglm[(df_roiglm['Chroma'] == 'hbo') & (df_roiglm['ROI'].isin(included_rois))]

# Define the order of Conditions to be displayed on the x-axis
condition_order = ['Speech_Sil', 'A_SPIN_Sil', 'T_Sil', 'AT_SPIN_Sil', 'Control_Sil']

# Create a combined dataframe with 'Amplitude' and 'Theta' values
df_hbo['Type'] = 'Amplitude'
df_roiglm_hbo['Type'] = 'Theta'
df_roiglm_hbo.rename(columns={'theta': 'Value'}, inplace=True)

combined_data = pd.concat([df_hbo, df_roiglm_hbo], ignore_index=True)

# Ensure there are no duplicate labels by resetting index
combined_data.reset_index(drop=True, inplace=True)

# Plotting all conditions in one plot
plt.figure(figsize=(15, 8))

sns.violinplot(x='Condition', y='Value', hue='Type', data=combined_data, order=condition_order, split=True, inner=None, linewidth=1.5,
               palette={'Amplitude': 'cadetblue', 'Theta': 'navy'})

# Customize plot
plt.title('peak averages and beta-values in bilateral STG')
plt.xlabel('Condition')
plt.ylabel('HbO response(µM)')
plt.xticks(rotation=45)
plt.ylim(-5, 12)
plt.legend(title='Type')

plt.savefig(savepath2 + r"\violinplotsROIsinglecond.svg", format='svg')       


#%% 1 a. area.specific responses expected:   or  test A vs T in somatos sensory and  STG
 ## betas ##############
 
     
input_data = df_roiglm.query("Condition in  ['T_Sil','Noise_Sil']") # ['T_Sil','Speech_Sil']          'T_Sil','Noise_Sil'
input_data = input_data.query("Chroma in ['hbo']")
input_data = input_data.query("ROI in ['lSTG',  'rSTG' , 'Left_SC',  'Right_SC']") # 'lSTG']
roi_model = smf.mixedlm("theta ~ Condition:ROI", input_data,
                        groups=input_data["ID"]).fit(method='nm')
roi_model.summary()
df_lme_GLMROIlongchans= statsmodels_to_results(roi_model)

 # only keep those starting with condition for FDR correct:
# Filter to include only rows where the 'Index' column indicates an interaction term
df_lme_GLMROIlongchans = df_lme_GLMROIlongchans[df_lme_GLMROIlongchans.index.str.contains(':')]



# FDR correction  Multiple comparison correction of LME outputs
pvals = df_lme_GLMROIlongchans['P>|z|'].values
pvals_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)[1]

# Add the corrected p-values back to the dataframe
df_lme_GLMROIlongchans['p_FDRcorrected'] = pvals_corrected
df_lme_GLMROIlongchans = df_lme_GLMROIlongchans.rename(columns={'Coef.': 'Beta', 'P>|z|': 'p-value'})
df_lme_GLMROIlongchans = df_lme_GLMROIlongchans.drop(columns=['Std.Err.', 'z', '[0.025', '0.975]', 'Significant'])
df_lme_GLMROIlongchans['significant?'] = df_lme_GLMROIlongchans['p_FDRcorrected'] <= 0.05


# Apply this function to the specific columns you want to format
df_lme_GLMROIlongchans['Beta'] = df_lme_GLMROIlongchans['Beta'].apply(lambda x: format_number(x))
df_lme_GLMROIlongchans['p-value'] = df_lme_GLMROIlongchans['p-value'].apply(lambda x: format_number(x))
df_lme_GLMROIlongchans['p_FDRcorrected'] = df_lme_GLMROIlongchans['p_FDRcorrected'].apply(lambda x: format_number(x))
df_lme_GLMROIlongchans['Beta'] = pd.to_numeric(df_lme_GLMROIlongchans['Beta'], errors='coerce')
#df_lme_GLMROIlongchans.to_csv(savepath2+ r'\glm\df_RQ1.speechversustactile.csv')
df_lme_GLMROIlongchans.to_csv(savepath2+ r'\glm\df_RQ1.noiseversustactile.csv')




 ## waves #################
input_data = df_waveformamplitudes.query("Condition in ['T_Sil','Speech_Sil']") # ['T_Sil','Noise_Sil']  'T_Sil','Speech_Sil'
input_data = input_data.query("Chroma in ['hbo']")
input_data = input_data.query("ROI in ['lSTG',  'rSTG' , 'Left_SC',  'Right_SC']") # 'lSTG']
roi_model = smf.mixedlm("Value ~ Condition:ROI", input_data,
                        groups=input_data["ID"]).fit(method='nm')
roi_model.summary()
df_lme_waveslongchans= statsmodels_to_results(roi_model)

df_lme_waveslongchans = df_lme_waveslongchans[df_lme_waveslongchans.index.str.contains(':')]


# FDR correction
pvals = df_lme_waveslongchans['P>|z|'].values
pvals_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)[1]

# Add the corrected p-values back to the dataframe
df_lme_waveslongchans['p_FDRcorrected'] = pvals_corrected
df_lme_waveslongchans = df_lme_waveslongchans.rename(columns={'Coef.': 'Amplitude', 'P>|z|': 'p-value'})
df_lme_waveslongchans = df_lme_waveslongchans.drop(columns=['Std.Err.', 'z', '[0.025', '0.975]', 'Significant'])
df_lme_waveslongchans['significant?'] = df_lme_waveslongchans['p_FDRcorrected'] <= 0.05


df_lme_waveslongchans['Amplitude'] = df_lme_waveslongchans['Amplitude'].apply(lambda x: format_number(x))
df_lme_waveslongchans['p-value'] = df_lme_waveslongchans['p-value'].apply(lambda x: format_number(x))
df_lme_waveslongchans['p_FDRcorrected'] = df_lme_waveslongchans['p_FDRcorrected'].apply(lambda x: format_number(x))

df_lme_waveslongchans['Amplitude'] = pd.to_numeric(df_lme_waveslongchans['Amplitude'], errors='coerce').round(1)

df_lme_waveslongchans.to_csv(savepath2 + r'\waveforms\RQ1aspeechvstactile.csv')
#df_lme_waveslongchans.to_csv(savepath2 + r'\waveforms\RQ1anoisevstactile.csv')


#%%  1b. speech larger than noise

### betas

input_data = df_roiglm.query("Condition in ['Speech_Sil','Noise_Sil']")
input_data = input_data.query("Chroma in ['hbo']")
input_data = input_data.query("ROI in ['lSTG','rSTG']")  # "ROI in ['lSTG']"
roi_model = smf.mixedlm("theta ~ Condition:ROI", input_data,
                        groups=input_data["ID"]).fit(method='nm')
roi_model.summary()
df_lme_GLMROIlongchans= statsmodels_to_results(roi_model)

df_lme_GLMROIlongchans = df_lme_GLMROIlongchans[df_lme_GLMROIlongchans.index.str.contains(':')] # normally : should exclude main effects, while * includes, but main effect of ROI is returned here either way



# FDR correction  Multiple comparison correction of LME outputs
pvals = df_lme_GLMROIlongchans['P>|z|'].values
pvals_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)[1]

# Add the corrected p-values back to the dataframe
df_lme_GLMROIlongchans['p_FDRcorrected'] = pvals_corrected
df_lme_GLMROIlongchans = df_lme_GLMROIlongchans.rename(columns={'Coef.': 'Beta', 'P>|z|': 'p-value'})
df_lme_GLMROIlongchans = df_lme_GLMROIlongchans.drop(columns=['Std.Err.', 'z', '[0.025', '0.975]', 'Significant'])
df_lme_GLMROIlongchans['significant?'] = df_lme_GLMROIlongchans['p_FDRcorrected'] <= 0.05


# Apply this function to the specific columns you want to format
df_lme_GLMROIlongchans['Beta'] = df_lme_GLMROIlongchans['Beta'].apply(lambda x: format_number(x))
df_lme_GLMROIlongchans['p-value'] = df_lme_GLMROIlongchans['p-value'].apply(lambda x: format_number(x))
df_lme_GLMROIlongchans['p_FDRcorrected'] = df_lme_GLMROIlongchans['p_FDRcorrected'].apply(lambda x: format_number(x))
df_lme_GLMROIlongchans['Beta'] = pd.to_numeric(df_lme_GLMROIlongchans['Beta'], errors='coerce')
df_lme_GLMROIlongchans.to_csv(savepath2+ r'\glm\df_RQ1.2SpeechlargerNoise_STGs.csv')


### waves
# compare statsitically for all chan (hbo only)
input_data = df_waveformamplitudes.query("Condition in ['Speech_Sil','Noise_Sil']")
input_data = input_data.query("Chroma in ['hbo']")
input_data = input_data.query("ROI in ['lSTG', 'rSTG']") # ]
roi_model = smf.mixedlm("Value ~ Condition:ROI", input_data,
                        groups=input_data["ID"]).fit(method='nm') # same results as with default fitting methods
roi_model.summary()
df_lme_waveslongchans= statsmodels_to_results(roi_model)

# interested in interaction effects only
df_lme_waveslongchans = df_lme_waveslongchans[df_lme_waveslongchans.index.str.contains(':')]

# FDR correction
pvals = df_lme_waveslongchans['P>|z|'].values
pvals_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)[1]

# Add the corrected p-values back to the dataframe
df_lme_waveslongchans['p_FDRcorrected'] = pvals_corrected
df_lme_waveslongchans = df_lme_waveslongchans.rename(columns={'Coef.': 'Amplitude', 'P>|z|': 'p-value'})
df_lme_waveslongchans = df_lme_waveslongchans.drop(columns=['Std.Err.', 'z', '[0.025', '0.975]', 'Significant'])
df_lme_waveslongchans['significant?'] = df_lme_waveslongchans['p_FDRcorrected'] <= 0.05


df_lme_waveslongchans['Amplitude'] = df_lme_waveslongchans['Amplitude'].apply(lambda x: format_number(x))
df_lme_waveslongchans['p-value'] = df_lme_waveslongchans['p-value'].apply(lambda x: format_number(x))
df_lme_waveslongchans['p_FDRcorrected'] = df_lme_waveslongchans['p_FDRcorrected'].apply(lambda x: format_number(x))

df_lme_waveslongchans['Amplitude'] = pd.to_numeric(df_lme_waveslongchans['Amplitude'], errors='coerce').round(1)
df_lme_waveslongchans.to_csv(savepath2+ r'\waveforms\df_RQ1.2SpeechlargerNoise_STGs.csv')

#%% [c. SIN vs SIQ]

### betas
input_data = df_roiglm.query("Condition in ['Speech_Sil','A_SPIN_Sil']")
input_data = input_data.query("Chroma in ['hbo']")
input_data = input_data.query("ROI in ['lSTG',  'rSTG' , 'leftIFG']") # 'lSTG']
roi_model = smf.mixedlm("theta ~ Condition:ROI", input_data,
                        groups=input_data["ID"]).fit(method='nm')
roi_model.summary()
df_lme_GLMROIlongchans= statsmodels_to_results(roi_model)

df_lme_GLMROIlongchans = df_lme_GLMROIlongchans[df_lme_GLMROIlongchans.index.str.contains(':')] # normally : should exclude main effects, while * includes, but main effect of ROI is returned here either way



# FDR correction  Multiple comparison correction of LME outputs
pvals = df_lme_GLMROIlongchans['P>|z|'].values
pvals_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)[1]

# Add the corrected p-values back to the dataframe
df_lme_GLMROIlongchans['p_FDRcorrected'] = pvals_corrected
df_lme_GLMROIlongchans = df_lme_GLMROIlongchans.rename(columns={'Coef.': 'Beta', 'P>|z|': 'p-value'})
df_lme_GLMROIlongchans = df_lme_GLMROIlongchans.drop(columns=['Std.Err.', 'z', '[0.025', '0.975]', 'Significant'])
df_lme_GLMROIlongchans['significant?'] = df_lme_GLMROIlongchans['p_FDRcorrected'] <= 0.05


# Apply this function to the specific columns you want to format
df_lme_GLMROIlongchans['Beta'] = df_lme_GLMROIlongchans['Beta'].apply(lambda x: format_number(x))
df_lme_GLMROIlongchans['p-value'] = df_lme_GLMROIlongchans['p-value'].apply(lambda x: format_number(x))
df_lme_GLMROIlongchans['p_FDRcorrected'] = df_lme_GLMROIlongchans['p_FDRcorrected'].apply(lambda x: format_number(x))
df_lme_GLMROIlongchans['Beta'] = pd.to_numeric(df_lme_GLMROIlongchans['Beta'], errors='coerce')
df_lme_GLMROIlongchans.to_csv(savepath2+ r'\glm\df_RQ1.cSpeechlargerNoisySpeech.csv')


### waves
# compare statsitically for all chan (hbo only)
input_data = df_waveformamplitudes.query("Condition in ['Speech_Sil','A_SPIN_Sil']")
input_data = input_data.query("Chroma in ['hbo']")
input_data = input_data.query("ROI in ['lSTG',  'rSTG' , 'leftIFG']") # 'lSTG']
roi_model = smf.mixedlm("Value ~ Condition:ROI", input_data,
                        groups=input_data["ID"]).fit(method='nm')
roi_model.summary()
df_lme_waveslongchans= statsmodels_to_results(roi_model)

# interested in interaction effects only
df_lme_waveslongchans = df_lme_waveslongchans[df_lme_waveslongchans.index.str.contains(':')]

# FDR correction
pvals = df_lme_waveslongchans['P>|z|'].values
pvals_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)[1]

# Add the corrected p-values back to the dataframe
df_lme_waveslongchans['p_FDRcorrected'] = pvals_corrected
df_lme_waveslongchans = df_lme_waveslongchans.rename(columns={'Coef.': 'Amplitude', 'P>|z|': 'p-value'})
df_lme_waveslongchans = df_lme_waveslongchans.drop(columns=['Std.Err.', 'z', '[0.025', '0.975]', 'Significant'])
df_lme_waveslongchans['significant?'] = df_lme_waveslongchans['p_FDRcorrected'] <= 0.05


df_lme_waveslongchans['Amplitude'] = df_lme_waveslongchans['Amplitude'].apply(lambda x: format_number(x))
df_lme_waveslongchans['p-value'] = df_lme_waveslongchans['p-value'].apply(lambda x: format_number(x))
df_lme_waveslongchans['p_FDRcorrected'] = df_lme_waveslongchans['p_FDRcorrected'].apply(lambda x: format_number(x))

df_lme_waveslongchans['Amplitude'] = pd.to_numeric(df_lme_waveslongchans['Amplitude'], errors='coerce')
df_lme_waveslongchans.to_csv(savepath2 + r'\waveforms\df_RQ1.cSpeechlargerNoisySpeech.csv')

  

#%% Tnoi different than Tsil? compare for last seven subj only (based on 6 good subjects)

#%% ROI resulsts

#betas
df_roiglm['ID'] = df_roiglm['ID'].astype(int)  # Ensure ID is an integer
grp_results = df_roiglm[df_roiglm['ID'] >= 18]


# Apply the specified conditions and chromophore filters
condition = "Condition in ['T_Sil','T_Noi']"
chroma = "Chroma in ['hbo']"

grp_results = grp_results.query(condition)
grp_results = grp_results.query(chroma)
grp_results = grp_results.query("ROI in ['lSTG',  'rSTG' , 'Left_SC',   'Right_SC',  'leftIFG']")
# Ensure there are no NaN values in the 'theta' column that will be used in the model
grp_results = grp_results.dropna(subset=['theta'])

# its not a chan model, but don't wanna rename everything
# Fit the mixed linear model
chan_glmmodel = smf.mixedlm("theta ~ Condition:ROI", 
                            grp_results, groups=grp_results["ID"], missing="ignore").fit(method='nm')

# Display the summary
chan_glmmodel.summary()

# Convert the model results to a DataFrame (assuming statsmodels_to_results does this)
df_lme_glmchanlong = statsmodels_to_results(chan_glmmodel)


# FDR correction
pvals = df_lme_glmchanlong['P>|z|'].values
pvals_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)[1]

# Add the corrected p-values back to the dataframe
df_lme_glmchanlong['p_FDRcorrected'] = pvals_corrected
df_lme_glmchanlong = df_lme_glmchanlong.rename(columns={'Coef.': 'Beta', 'P>|z|': 'p-value'})
df_lme_glmchanlong = df_lme_glmchanlong.drop(columns=['Std.Err.', 'z', '[0.025', '0.975]', 'Significant'])
df_lme_glmchanlong['significant?'] = df_lme_glmchanlong['p_FDRcorrected'] <= 0.05


# Apply this function to the specific columns you want to format
#df_lme_glmchanlong['Beta'] = df_lme_glmchanlong['Beta'].apply(lambda x: format_number(x))
    #small betas are replaced by string <0.001 which creates NaNs in plot)
df_lme_glmchanlong['p-value'] = df_lme_glmchanlong['p-value'].apply(lambda x: format_number(x))
df_lme_glmchanlong['p_FDRcorrected'] = df_lme_glmchanlong['p_FDRcorrected'].apply(lambda x: format_number(x))
df_lme_GLMROIlongchans.to_csv(savepath2+ r'\glm\extracheck_TSilvsTNoi.csv')


### waves

#rois
df_waveformamplitudes['ID'] = df_waveformamplitudes['ID'].astype(int)  # Ensure ID is an integer
input_data = df_waveformamplitudes[df_waveformamplitudes['ID'] >= 18]

# compare statsitically for all chan (hbo only)
input_data = input_data.query("Condition in ['T_Sil','T_Noi']")  #'T'
input_data = input_data.query("Chroma in ['hbo']")
input_data = input_data.query("ROI in ['lSTG',  'rSTG' , 'Left_SC',   'Right_SC',  'leftIFG']")
roi_model = smf.mixedlm("Value ~ Condition:ROI", input_data,
                        groups=input_data["ID"]).fit(method='nm')
roi_model.summary()
df_lme_waveslongchans= statsmodels_to_results(roi_model)


# FDR correction
pvals = df_lme_waveslongchans['P>|z|'].values
pvals_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)[1]

# Add the corrected p-values back to the dataframe
df_lme_waveslongchans['p_FDRcorrected'] = pvals_corrected
df_lme_waveslongchans = df_lme_waveslongchans.rename(columns={'Coef.': 'Amplitude', 'P>|z|': 'p-value'})
df_lme_waveslongchans = df_lme_waveslongchans.drop(columns=['Std.Err.', 'z', '[0.025', '0.975]', 'Significant'])
df_lme_waveslongchans['significant?'] = df_lme_waveslongchans['p_FDRcorrected'] <= 0.05


# Apply this function to the specific columns you want to format
df_lme_waveslongchans['Amplitude'] = df_lme_waveslongchans['Amplitude'].apply(lambda x: format_number(x))
df_lme_waveslongchans['p-value'] = df_lme_waveslongchans['p-value'].apply(lambda x: format_number(x))
df_lme_waveslongchans['p_FDRcorrected'] = df_lme_waveslongchans['p_FDRcorrected'].apply(lambda x: format_number(x))

df_lme_waveslongchans['Amplitude'] = pd.to_numeric(df_lme_waveslongchans['Amplitude'], errors='coerce')
df_lme_waveslongchans.to_csv(savepath2 + r'\waveforms\extracheck_TSilvsTNoi.csv')

#%% chans
    # glm
    
df_channelstats['ID'] = df_channelstats['ID'].astype(int)  # Ensure ID is an integer
grp_results = df_channelstats[df_channelstats['ID'] >= 18]


# Apply the specified conditions and chromophore filters
condition = "Condition in ['T_Sil','T_Noi']"
chroma = "Chroma in ['hbo']"

grp_results = grp_results.query(condition)
grp_results = grp_results.query(chroma)

# Ensure there are no NaN values in the 'theta' column that will be used in the model
grp_results = grp_results.dropna(subset=['theta'])

# Fit the mixed linear model
chan_glmmodel = smf.mixedlm("theta ~ Condition:ch_name", 
                            grp_results, groups=grp_results["ID"], missing="ignore").fit(method='nm')

# Display the summary
chan_glmmodel.summary()

# Convert the model results to a DataFrame (assuming statsmodels_to_results does this)
df_lme_glmchanlong = statsmodels_to_results(chan_glmmodel)

df_lme_glmchanlong = df_lme_glmchanlong[df_lme_glmchanlong.index.str.contains(':')]

# FDR correction
pvals = df_lme_glmchanlong['P>|z|'].values
pvals_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)[1]

# Add the corrected p-values back to the dataframe
df_lme_glmchanlong['p_FDRcorrected'] = pvals_corrected
df_lme_glmchanlong = df_lme_glmchanlong.rename(columns={'Coef.': 'Beta', 'P>|z|': 'p-value'})
df_lme_glmchanlong = df_lme_glmchanlong.drop(columns=['Std.Err.', 'z', '[0.025', '0.975]', 'Significant'])
df_lme_glmchanlong['significant?'] = df_lme_glmchanlong['p_FDRcorrected'] <= 0.05


# Apply this function to the specific columns you want to format
#df_lme_glmchanlong['Beta'] = df_lme_glmchanlong['Beta'].apply(lambda x: format_number(x))
    #small betas are replaced by string <0.001 which creates NaNs in plot)
df_lme_glmchanlong['p-value'] = df_lme_glmchanlong['p-value'].apply(lambda x: format_number(x))
df_lme_glmchanlong['p_FDRcorrected'] = df_lme_glmchanlong['p_FDRcorrected'].apply(lambda x: format_number(x))



# brain plots possible for "contrasts" interactions?


# theta different order than in info!
info = raw_haemo.copy().pick('hbo').info
channelorder = info.ch_names


import re
def extract_channel_name(text):
    match = re.search(r'ch_name\[(.*?)\]', text)
    return match.group(1) if match else None

df_lme_glmchanlong['ch_name'] = df_lme_glmchanlong.index.map(extract_channel_name)

# Reorder the DataFrame based on the extracted channel names
df_lme_glmchanlong_ordered = df_lme_glmchanlong.set_index('ch_name').reindex(channelorder).reset_index()

# Optional: set the original index back if needed
df_lme_glmchanlong_ordered.index = df_lme_glmchanlong.set_index('ch_name').reindex(channelorder).index
df_lme_glmchanlong_ordered.drop('ch_name', axis=1, inplace=True)


radius =df_lme_glmchanlong_ordered['significant?'].apply(lambda x: 0.0017 if x else 0.001) # result turn wrong (wrong colors) when choosing thinner values!!!
#does not seem to do the right thing..


fig= plot_nirs_source_detector_wAxixLims(
    df_lme_glmchanlong_ordered['Beta'], 
    raw_haemo.copy().pick('hbo').info , fnirs = True,
    radius= radius.values,
subject= 'fsaverage', coord_frame= 'head', trans = 'fsaverage', surfaces=['brain'],vmin = -4, vmax = 4)

plotter = fig.plotter  # Get the PyVista plotter from the figure

# Right hemisphere view (looking from the right to the left)
right_camera_position = [0.5, 0, 0]  # Adjust the x-coordinate as needed
focal_point = [0, 0, 0]
view_up = [0, 0, 1]

# Set and update for the right hemisphere view
plotter.camera_position = [right_camera_position, focal_point, view_up]
plotter.render()
plotter.screenshot(savepath2 + r'\glm\brainplots_lat7subs_Tsil_minus_TNoi_rh.png')
# Left hemisphere view (looking from the left to the right)
left_camera_position = [-0.5, 0, 0.07]  # Mirror the x-coordinate

# Set and update for the left hemisphere view
plotter.camera_position = [left_camera_position, focal_point, view_up]
plotter.render()
plotter.screenshot(savepath2 + r'\glm\brainplots_last7subs_Tsil_minus_TNoi_lh.png')

# save plots: difference Tnoi vs. T sil (in 6 subj ) that have both conditions

### also do single activations. next to difference: pick only one condition and do the reordering of channels as previously:
    #..

#############################################glm single conditions t sil or tnoi

df_channelstats['ID'] = df_channelstats['ID'].astype(int)  # Ensure ID is an integer
grp_results = df_channelstats[df_channelstats['ID'] >= 18]

# Apply the specified conditions and chromophore filters
condition = "Condition in ['T_Noi']" #'T_Sil']
chroma = "Chroma in ['hbo']"

grp_results = grp_results.query(condition)
grp_results = grp_results.query(chroma)
grp_results = grp_results.dropna(subset=['theta'])
chan_glmmodel = smf.mixedlm("theta ~ -1 + Condition:ch_name", 
                            grp_results, groups=grp_results["ID"], missing="ignore").fit(method='nm')
chan_glmmodel.summary()
df_lme_glmchanlong = statsmodels_to_results(chan_glmmodel)
df_lme_glmchanlong = df_lme_glmchanlong[df_lme_glmchanlong.index.str.contains(':')]

# FDR correction
pvals = df_lme_glmchanlong['P>|z|'].values
pvals_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)[1]
df_lme_glmchanlong['p_FDRcorrected'] = pvals_corrected
df_lme_glmchanlong = df_lme_glmchanlong.rename(columns={'Coef.': 'Beta', 'P>|z|': 'p-value'})
df_lme_glmchanlong = df_lme_glmchanlong.drop(columns=['Std.Err.', 'z', '[0.025', '0.975]', 'Significant'])
df_lme_glmchanlong['significant?'] = df_lme_glmchanlong['p_FDRcorrected'] <= 0.05


# Apply this function to the specific columns you want to format
#df_lme_glmchanlong['Beta'] = df_lme_glmchanlong['Beta'].apply(lambda x: format_number(x))
    #small betas are replaced by string <0.001 which creates NaNs in plot)
df_lme_glmchanlong['p-value'] = df_lme_glmchanlong['p-value'].apply(lambda x: format_number(x))
df_lme_glmchanlong['p_FDRcorrected'] = df_lme_glmchanlong['p_FDRcorrected'].apply(lambda x: format_number(x))

# theta different order than in info!
info = raw_haemo.copy().pick('hbo').info
channelorder = info.ch_names

df_lme_glmchanlong['ch_name'] = df_lme_glmchanlong.index.map(extract_channel_name)
df_lme_glmchanlong_ordered = df_lme_glmchanlong.set_index('ch_name').reindex(channelorder).reset_index()
df_lme_glmchanlong_ordered.index = df_lme_glmchanlong.set_index('ch_name').reindex(channelorder).index
df_lme_glmchanlong_ordered.drop('ch_name', axis=1, inplace=True)

radius =df_lme_glmchanlong_ordered['significant?'].apply(lambda x: 0.0017 if x else 0.001) # result turn wrong (wrong colors) when choosing thinner values!!!

fig= plot_nirs_source_detector_wAxixLims(
    df_lme_glmchanlong_ordered['Beta'], 
    raw_haemo.copy().pick('hbo').info , fnirs = True,
    radius= radius.values,
subject= 'fsaverage', coord_frame= 'head', trans = 'fsaverage', surfaces=['brain'],vmin = -4, vmax = 4)

plotter = fig.plotter  # Get the PyVista plotter from the figure

# Right hemisphere view (looking from the right to the left)
right_camera_position = [0.5, 0, 0]  # Adjust the x-coordinate as needed
focal_point = [0, 0, 0]
view_up = [0, 0, 1]

# Set and update for the right hemisphere view
plotter.camera_position = [right_camera_position, focal_point, view_up]
plotter.render()
#plotter.screenshot(savepath2 + r'\glm\brainplots_last7Subs_TSil_rh.png')
plotter.screenshot(savepath2 + r'\glm\brainplots_last7Subs_TNoi_rh.png')
# Left hemisphere view (looking from the left to the right)
left_camera_position = [-0.5, 0, 0.07]  # Mirror the x-coordinate

# Set and update for the left hemisphere view
plotter.camera_position = [left_camera_position, focal_point, view_up]
plotter.render()
#plotter.screenshot(savepath2 + r'\glm\brainplots_last7subs_TSil_lh.png')
plotter.screenshot(savepath2 + r'\glm\brainplots_last7subs_TNoi_lh.png')
#%% tsil vs tnoi waves #################################################

        # stats: generate data frame from waveform responses
df_waveformamplitudes_chans = pd.DataFrame(columns=['ID', 'ch_name', 'Chroma', 'Condition', 'Value'])

for idx, evoked in enumerate(all_evokeds_allCond):
    for subj_data in all_evokeds_allCond[evoked]:
        subj_id = subj_data.info['subject_info']['his_id'] if 'subject_info' in subj_data.info and 'his_id' in subj_data.info['subject_info'] else 'unknown'
        bads = subj_data.info['bads']
        for ch_name in subj_data.ch_names:
            if ch_name not in bads and ('hbo' in ch_name or 'hbr' in ch_name):
                chroma = 'hbo' if 'hbo' in ch_name else 'hbr'
                data = deepcopy(subj_data).pick(picks=[ch_name])
                value = data.crop(tmin=5.0, tmax=8.0).data.mean() * 1.0e6  
                
                # Append metadata and extracted feature to dataframe
                this_df = pd.DataFrame(
                    {'ID': subj_id, 'ch_name': ch_name.split(' ')[0], 'Chroma': chroma,
                     'Condition': evoked, 'Value': value}, index=[0])
                df_waveformamplitudes_chans = pd.concat([df_waveformamplitudes_chans, this_df], ignore_index=True)

df_waveformamplitudes_chans.reset_index(inplace=True, drop=True)
df_waveformamplitudes_chans['Value'] = pd.to_numeric(df_waveformamplitudes_chans['Value'])  # Convert Value column to numeric

df_waveformamplitudes_chans['ID'] = df_waveformamplitudes_chans['ID'].str.extract('(\d+)').astype(int)
# change 0 to 25
df_waveformamplitudes_chans['ID'] = df_waveformamplitudes_chans['ID'].replace(0, 25)



# now extract the last 7(6) with both T conditions:
df_waveformamplitudes_chans['ID'] = df_waveformamplitudes_chans['ID'].astype(int)  # Ensure ID is an integer
input_data = df_waveformamplitudes_chans[df_waveformamplitudes_chans['ID'] >= 18]


condition = "Condition in ['T_Noi','T_Sil']" #"Condition in ['Auditory Speech']","Condition in ['Noise']" 
chroma = "Chroma in ['hbo']"
chroma2 = 'hbo'

grp_results = df_waveformamplitudes_chans.query(condition)
grp_results = grp_results.query(chroma)

grp_results = grp_results.dropna(subset=['Value'])

chan_wavemodel = smf.mixedlm("Value ~ Condition:ch_name", 
                        grp_results, groups=grp_results["ID"], missing="ignore").fit(method='nm')
chan_wavemodel.summary()
df_lme_chan_wavemodel = statsmodels_to_results(chan_wavemodel)

df_lme_chan_wavemodel = df_lme_chan_wavemodel[df_lme_chan_wavemodel.index.str.contains(':')]

# FDR correction
pvals = df_lme_chan_wavemodel['P>|z|'].values
pvals_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)[1]

# Add the corrected p-values back to the dataframe
df_lme_chan_wavemodel['p_FDRcorrected'] = pvals_corrected
df_lme_chan_wavemodel = df_lme_chan_wavemodel .rename(columns={'Coef.': 'Amplitude', 'P>|z|': 'p-value'})
df_lme_chan_wavemodel = df_lme_chan_wavemodel .drop(columns=['Std.Err.', 'z', '[0.025', '0.975]', 'Significant'])
df_lme_chan_wavemodel['significant?'] = df_lme_chan_wavemodel['p_FDRcorrected'] <= 0.05


# Apply this function to the specific columns you want to format
#df_lme_chan_wavemodel ['Beta'] = df_lme_chan_wavemodel ['Beta'].apply(lambda x: format_number(x))
    #small betas are replaced by string <0.001 which creates NaNs in plot)
df_lme_chan_wavemodel['p-value'] = df_lme_chan_wavemodel['p-value'].apply(lambda x: format_number(x))
df_lme_chan_wavemodel['p_FDRcorrected'] = df_lme_chan_wavemodel['p_FDRcorrected'].apply(lambda x: format_number(x))


# channel order different than in info! -> align so plot is correct
info = raw_haemo.copy().pick(chroma2).info
channelorder = info.ch_names

df_lme_chan_wavemodel['ch_name'] = df_lme_chan_wavemodel.index.map(extract_channel_name)
df_lme_chan_wavemodel['ch_name'] = df_lme_chan_wavemodel['ch_name'] + ' hbo'

# Reorder the DataFrame based on the extracted channel names
df_lme_chan_wavemodel_ordered = df_lme_chan_wavemodel.set_index('ch_name').reindex(channelorder).reset_index()

# Optional: set the original index back if needed
df_lme_chan_wavemodel_ordered.index = df_lme_chan_wavemodel.set_index('ch_name').reindex(channelorder).index
df_lme_chan_wavemodel_ordered.drop('ch_name', axis=1, inplace=True)


#df_lme_chan_wavemodel_ordered['radius'] = df_lme_chan_wavemodel_ordered['significant?'].apply(lambda x: 0.0017 if x else 0.001)
radius = df_lme_chan_wavemodel_ordered['significant?'].apply(lambda x: 0.0017 if x else 0.001)

fig= plot_nirs_source_detector_wAxixLims(
    df_lme_chan_wavemodel_ordered['Amplitude'], 
    raw_haemo.copy().pick('hbo').info , fnirs = True,
    radius= radius.values,
subject= 'fsaverage', coord_frame= 'head', trans = 'fsaverage', surfaces=['brain'],vmin = -4, vmax = 4)

plotter = fig.plotter  # Get the PyVista plotter from the figure

# Right hemisphere view (looking from the right to the left)
right_camera_position = [0.5, 0, 0]  # Adjust the x-coordinate as needed
focal_point = [0, 0, 0]
view_up = [0, 0, 1]

# Set and update for the right hemisphere view
plotter.camera_position = [right_camera_position, focal_point, view_up]
plotter.render()
#plotter.screenshot(savepath + r'\waveforms\brainplots_AudSpeech_Sil_rh.png')
plotter.screenshot(savepath2 + r'\waveforms\brainplots_last7subs_Tsil_minus_TNoi_rh.png.png')

# Left hemisphere view (looking from the left to the right)
left_camera_position = [-0.5, 0, 0.07]  # Mirror the x-coordinate

# Set and update for the left hemisphere view
plotter.camera_position = [left_camera_position, focal_point, view_up]
plotter.render()
#plotter.screenshot(savepath + r'\waveforms\brainplots_AudSpeech_Sil_lh.png')
plotter.screenshot(savepath2 + r'\waveforms\brainplots_last7subs_Tsil_minus_TNoi_lh.png')


############################### waves singel condition tsil or tnoi

# show one condition only
condition = "Condition in ['T_Sil']" #'T_Sil' 'T_Noi
chroma = "Chroma in ['hbo']"
chroma2 = 'hbo'
grp_results = df_waveformamplitudes_chans.query(condition)
grp_results = grp_results.query(chroma)
grp_results = grp_results.dropna(subset=['Value'])
chan_wavemodel = smf.mixedlm("Value ~ -1 +Condition:ch_name", 
                        grp_results, groups=grp_results["ID"], missing="ignore").fit(method='nm')
chan_wavemodel.summary()
df_lme_chan_wavemodel = statsmodels_to_results(chan_wavemodel)
df_lme_chan_wavemodel = df_lme_chan_wavemodel[df_lme_chan_wavemodel.index.str.contains(':')]
# FDR correction
pvals = df_lme_chan_wavemodel['P>|z|'].values
pvals_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)[1]
# Add the corrected p-values back to the dataframe
df_lme_chan_wavemodel['p_FDRcorrected'] = pvals_corrected
df_lme_chan_wavemodel = df_lme_chan_wavemodel .rename(columns={'Coef.': 'Amplitude', 'P>|z|': 'p-value'})
df_lme_chan_wavemodel = df_lme_chan_wavemodel .drop(columns=['Std.Err.', 'z', '[0.025', '0.975]', 'Significant'])
df_lme_chan_wavemodel['significant?'] = df_lme_chan_wavemodel['p_FDRcorrected'] <= 0.05

df_lme_chan_wavemodel['p-value'] = df_lme_chan_wavemodel['p-value'].apply(lambda x: format_number(x))
df_lme_chan_wavemodel['p_FDRcorrected'] = df_lme_chan_wavemodel['p_FDRcorrected'].apply(lambda x: format_number(x))

# channel order different than in info! -> align so plot is correct
info = raw_haemo.copy().pick(chroma2).info
channelorder = info.ch_names

df_lme_chan_wavemodel['ch_name'] = df_lme_chan_wavemodel.index.map(extract_channel_name)
df_lme_chan_wavemodel['ch_name'] = df_lme_chan_wavemodel['ch_name'] + ' hbo'

# Reorder the DataFrame based on the extracted channel names
df_lme_chan_wavemodel_ordered = df_lme_chan_wavemodel.set_index('ch_name').reindex(channelorder).reset_index()
# Optional: set the original index back if needed
df_lme_chan_wavemodel_ordered.index = df_lme_chan_wavemodel.set_index('ch_name').reindex(channelorder).index
df_lme_chan_wavemodel_ordered.drop('ch_name', axis=1, inplace=True)

#df_lme_chan_wavemodel_ordered['radius'] = df_lme_chan_wavemodel_ordered['significant?'].apply(lambda x: 0.0017 if x else 0.001)
radius = df_lme_chan_wavemodel_ordered['significant?'].apply(lambda x: 0.0017 if x else 0.001)

fig= plot_nirs_source_detector_wAxixLims(
    df_lme_chan_wavemodel_ordered['Amplitude'], 
    raw_haemo.copy().pick('hbo').info , fnirs = True,
    radius= radius.values,
subject= 'fsaverage', coord_frame= 'head', trans = 'fsaverage', surfaces=['brain'],vmin = -4, vmax = 4)

plotter = fig.plotter  # Get the PyVista plotter from the figure

# Right hemisphere view (looking from the right to the left)
right_camera_position = [0.5, 0, 0]  # Adjust the x-coordinate as needed
focal_point = [0, 0, 0]
view_up = [0, 0, 1]

# Set and update for the right hemisphere view
plotter.camera_position = [right_camera_position, focal_point, view_up]
plotter.render()
#plotter.screenshot(savepath + r'\waveforms\brainplots_AudSpeech_Sil_rh.png')
#plotter.screenshot(savepath2 + r'\waveforms\brainplots_last7subs_TNoi_rh.png.png')
plotter.screenshot(savepath2 + r'\waveforms\brainplots_last7subs_TSil_rh.png.png')
# Left hemisphere view (looking from the left to the right)
left_camera_position = [-0.5, 0, 0.07]  # Mirror the x-coordinate

# Set and update for the left hemisphere view
plotter.camera_position = [left_camera_position, focal_point, view_up]
plotter.render()
#plotter.screenshot(savepath + r'\waveforms\brainplots_AudSpeech_Sil_lh.png')
#plotter.screenshot(savepath2 + r'\waveforms\brainplots_last7subs_TNoi_lh.png')
plotter.screenshot(savepath2 + r'\waveforms\brainplots_last7subs_TSil_lh.png')


#%%%  RQ3: AT speech
# compare statsitically for all chan (hbo only)


#%% in hypothesized ROI

df_waveformamplitudes_Rename = df_waveformamplitudes
df_waveformamplitudes_Rename['Condition'] = df_waveformamplitudes_Rename['Condition'].replace('AT_SPIN_Sil', 'AAT_SPIN_Sil')
input_data = df_waveformamplitudes_Rename.query("Condition in ['AAT_SPIN_Sil', 'A_SPIN_Sil']")  #'T'
#input_data = df_waveformamplitudes_Rename.query("Condition in ['AAT_SPIN_Sil', 'T_Sil']")  #'T'


#input_data = df_waveformamplitudes.query("Condition in ['AT_SPIN_Sil', 'A_SPIN_Sil']")  #'T
input_data = input_data.query("Chroma in ['hbo']")
#check also lSTG rSTG
#input_data = input_data.query("ROI in ['leftPostTemp']") 
#input_data = input_data.query("ROI in ['lSTG','rSTG' ]") # ['lSTG',  'rSTG' , 'Left_SC',   'Right_SC',  'leftIFG'] ['leftPostTemp']
input_data = input_data.query("ROI in ['bilateralSTG']") # ['lSTG',  'rSTG' , 'Left_SC',   'Right_SC',  'leftIFG'] ['leftPostTemp']
roi_model = smf.mixedlm("Value ~ Condition:ROI", input_data,
                        groups=input_data["ID"]).fit(method='nm')
roi_model.summary()
df_lme_waveslongchans= statsmodels_to_results(roi_model)
# df_lme_waveslongchans.reset_index(drop=True, inplace=True)
# df_lme_waveslongchans = df_lme_waveslongchans.set_index(['ROI', 'Condition'])

# # Filter rows where the index contains ':'
# df_lme_waveslongchans= df_lme_waveslongchans[df_lme_waveslongchans.index.str.contains(':')]

# # FDR correction
# pvals = df_lme_waveslongchans['P>|z|'].values
# pvals_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)[1]

# # Add the corrected p-values back to the dataframe
# df_lme_waveslongchans['p_FDRcorrected'] = pvals_corrected
# df_lme_waveslongchans = df_lme_waveslongchans.rename(columns={'Coef.': 'Amplitude', 'P>|z|': 'p-value'})
# df_lme_waveslongchans = df_lme_waveslongchans.drop(columns=['Std.Err.', 'z', '[0.025', '0.975]', 'Significant'])
# df_lme_waveslongchans['significant?'] = df_lme_waveslongchans['p_FDRcorrected'] <= 0.05


# # Apply this function to the specific columns you want to format
# df_lme_waveslongchans['Amplitude'] = df_lme_waveslongchans['Amplitude'].apply(lambda x: format_number(x))
# df_lme_waveslongchans['p-value'] = df_lme_waveslongchans['p-value'].apply(lambda x: format_number(x))
# df_lme_waveslongchans['p_FDRcorrected'] = df_lme_waveslongchans['p_FDRcorrected'].apply(lambda x: format_number(x))

# df_lme_waveslongchans['Amplitude'] = pd.to_numeric(df_lme_waveslongchans['Amplitude'], errors='coerce')

# df_lme_waveslongchans.to_csv(savepath2 + r'\waveforms\df_waveforms_RQ2_ATvsTslposttemp.csv')

# # see all subjects (--> plot from joerg or this)
# grp_results = df_waveformamplitudes.query("Condition in ['AT_SPIN_Sil', 'A_SPIN_Sil','T_Sil']")
# grp_results = grp_results.query("Chroma in ['hbo']")

# fig = sns.catplot(x="ID", y="Value", col="Condition",  hue="ROI", data=grp_results, col_wrap=2, errorbar=None, palette="muted", height=9, s=60)




#%% glm exact the same but for beta estimates


df_roiglm_Rename = df_roiglm
df_roiglm_Rename['Condition'] = df_roiglm_Rename['Condition'].replace('AT_SPIN_Sil', 'AAT_SPIN_Sil')
input_data = df_roiglm_Rename.query("Condition in ['AAT_SPIN_Sil', 'A_SPIN_Sil']") #
input_data = df_roiglm_Rename.query("Condition in ['AAT_SPIN_Sil', 'T_Sil']") #
input_data = input_data.query("Chroma in ['hbo']")
input_data = input_data.query("ROI in ['leftPostTemp']") #
roi_model = smf.mixedlm("theta ~ Condition:ROI", input_data,
                        groups=input_data["ID"]).fit(method='nm')
roi_model.summary()
df_lme_GLMROIlongchans= statsmodels_to_results(roi_model)

# # FDR correction  Multiple comparison correction of LME outputs
# pvals = df_lme_GLMROIlongchans['P>|z|'].values
# pvals_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)[1]

# # Add the corrected p-values back to the dataframe
# df_lme_GLMROIlongchans['p_FDRcorrected'] = pvals_corrected
# df_lme_GLMROIlongchans = df_lme_GLMROIlongchans.rename(columns={'Coef.': 'Beta', 'P>|z|': 'p-value'})
# df_lme_GLMROIlongchans = df_lme_GLMROIlongchans.drop(columns=['Std.Err.', 'z', '[0.025', '0.975]', 'Significant'])
# df_lme_GLMROIlongchans['significant?'] = df_lme_GLMROIlongchans['p_FDRcorrected'] <= 0.05


# # Apply this function to the specific columns you want to format
# df_lme_GLMROIlongchans['Beta'] = df_lme_GLMROIlongchans['Beta'].apply(lambda x: format_number(x))
# df_lme_GLMROIlongchans['p-value'] = df_lme_GLMROIlongchans['p-value'].apply(lambda x: format_number(x))
# df_lme_GLMROIlongchans['p_FDRcorrected'] = df_lme_GLMROIlongchans['p_FDRcorrected'].apply(lambda x: format_number(x))


# df_lme_GLMROIlongchans['Beta'] = pd.to_numeric(df_lme_GLMROIlongchans['Beta'], errors='coerce')
# df_lme_GLMROIlongchans.to_csv(savepath2 + r'\glm\RQ2_ATallROisexploratory.csv')

# FDR correct not needed in single roi single contrast results
# results are just 1 liners, put in text of manuscript, no tables needed
####### --> continue ROI analysis for AT later on data-defined ROI based on Speech_Sil


  
#%%% Channel results and brain plots
#%% brain plots single condition activations %

condition = "Condition in ['Control_Sil']" #"C("Condition in ['Speech_Sil','Noise_Sil', 'Control_Sil', 'A_SPIN_Sil', 'T_Sil','AT_SPIN_Sil', 'Speech_Noi', 'Control_Noi', 'T_Noi']")
chroma = "Chroma in ['hbo']"
chroma2 = 'hbo'

grp_results = df_channelstats.query(condition)
grp_results = grp_results.query(chroma)

grp_results = grp_results.dropna(subset=['theta'])

chan_glmmodel = smf.mixedlm("theta ~ -1 + Condition:ch_name", 
                        grp_results, groups=grp_results["ID"], missing="ignore").fit(method='nm')
chan_glmmodel.summary()
df_lme_glmchanlong = statsmodels_to_results(chan_glmmodel, order=raw_haemo.copy().pick("hbo").ch_names)


# try surface projection plot
# same error as for NH dataset and montage
clim = dict(kind='value', pos_lims=(0, 8, 11))
brain = mne_nirs.visualisation.plot_glm_surface_projection(raw_haemo.copy().pick("hbo"),
                                    df_lme_glmchanlong, clim=clim, view='dorsal',
                                    colorbar=True, size=(800, 700))
brain.add_text(0.05, 0.95, "Left-Right", 'title', font_size=16, color='k')




# ###############################alternatively hypothesis tests on single channel level
df_channelstats_Rename = df_channelstats.copy()
df_channelstats_Rename['Condition'] = df_channelstats_Rename['Condition'].replace('AT_SPIN_Sil', 'AAT_SPIN_Sil')
df_channelstats_Rename['Condition'] = df_channelstats_Rename['Condition'].replace('T_Sil', 'B_T_Sil')

 ###################0.1a: A NOise, A_SPIQ - TSPIQ 
input_data = df_channelstats_Rename.query("Condition in ['Speech_Sil', 'Noise_Sil', 'B_T_Sil']")  #'T'

# for plot split up in two
input_data = df_channelstats_Rename.query("Condition in ['Noise_Sil', 'B_T_Sil']")  #'T'


 ###################0.1b: A_Noise vs ASPIQ
input_data = df_channelstats_Rename.query("Condition in ['Speech_Sil', 'Noise_Sil']")  #'T'

 ###################0.1c: A_SPIN vs ASPIQ
input_data = df_channelstats_Rename.query("Condition in ['Speech_Sil', 'A_SPIN_Sil']")  #'T'
 ###################0.2: AT_SPIN vs A-SPIn, T-SPIQ
input_data = df_channelstats_Rename.query("Condition in ['AAT_SPIN_Sil', 'A_SPIN_Sil','B_T_Sil']") # 'T'



chroma = "Chroma in ['hbo']"
chroma2 = 'hbo'
grp_results = input_data.query(chroma)
grp_results = grp_results.dropna(subset=['theta'])

chan_glmmodel = smf.mixedlm("theta ~ Condition:ch_name", 
                        grp_results, groups=grp_results["ID:Condition"], missing="ignore").fit(method='nm')
chan_glmmodel.summary()
df_lme_glmchanlong = statsmodels_to_results(chan_glmmodel)

df_lme_glmchanlong= df_lme_glmchanlong[df_lme_glmchanlong.index.str.contains(':')]





chroma = "Chroma in ['hbo']"
chroma2 = 'hbo'
grp_results = input_data.query(chroma)
grp_results = grp_results.dropna(subset=['theta'])

chan_glmmodel = smf.mixedlm("theta ~ -1 + Condition:ch_name", 
                        grp_results, groups=grp_results["ID"], missing="ignore").fit(method='nm')
chan_glmmodel.summary()
df_lme_glmchanlong = statsmodels_to_results(chan_glmmodel)

df_lme_glmchanlong_nointercept= df_lme_glmchanlong[df_lme_glmchanlong.index.str.contains(':')]






# FDR correction
pvals = df_lme_glmchanlong['P>|z|'].values
pvals_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)[1]

# Add the corrected p-values back to the dataframe
df_lme_glmchanlong['p_FDRcorrected'] = pvals_corrected
df_lme_glmchanlong = df_lme_glmchanlong.rename(columns={'Coef.': 'Beta', 'P>|z|': 'p-value'})
df_lme_glmchanlong = df_lme_glmchanlong.drop(columns=['Std.Err.', 'z', '[0.025', '0.975]', 'Significant'])
df_lme_glmchanlong['significant?'] = df_lme_glmchanlong['p_FDRcorrected'] <= 0.05


# Apply this function to the specific columns you want to format
#df_lme_glmchanlong['Beta'] = df_lme_glmchanlong['Beta'].apply(lambda x: format_number(x))
    #small betas are replaced by string <0.001 which creates NaNs in plot)
df_lme_glmchanlong['p-value'] = df_lme_glmchanlong['p-value'].apply(lambda x: format_number(x))
df_lme_glmchanlong['p_FDRcorrected'] = df_lme_glmchanlong['p_FDRcorrected'].apply(lambda x: format_number(x))
df_lme_glmchanlong.to_csv(savepath2 + r'\glm\single channel analyses\channelbetaresults_TactilevsAud.csv')




# theta different order than in info!
info = raw_haemo.copy().pick(chroma2).info
channelorder = info.ch_names

# for normal condition activations
df_lme_glmchanlong_ordered = df_lme_glmchanlong.set_index('ch_name').reindex(channelorder).reset_index()
df_lme_glmchanlong_ordered['Beta'] = pd.to_numeric(df_lme_glmchanlong_ordered['Beta'], errors='coerce')
radius =df_lme_glmchanlong_ordered['significant?'].apply(lambda x: 0.0017 if x else 0.001) # result turn wrong (wrong colors) when choosing thinner values!!!

# reordering for contrast data frames
# Extract channel names from the index
df_lme_glmchanlong['ch_name'] = df_lme_glmchanlong.index.to_series().str.extract(r'ch_name\[(.*?)\]')[0]
df_lme_glmchanlong.set_index('ch_name', inplace=True)
df_lme_glmchanlong_ordered = df_lme_glmchanlong.reindex(channelorder)
df_lme_glmchanlong_ordered.reset_index(inplace=True)
radius =df_lme_glmchanlong_ordered['significant?'].apply(lambda x: 0.0017 if x else 0.001) # result turn wrong (wrong colors) when choosing thinner values!!!




fig= plot_nirs_source_detector_wAxixLims(
    df_lme_glmchanlong_ordered['Beta'], 
    raw_haemo.copy().pick('hbo').info , fnirs = True,
    radius= radius.values,
subject= 'fsaverage', coord_frame= 'head', trans = 'fsaverage', surfaces=['brain'],vmin = -4, vmax = 4)

plotter = fig.plotter  # Get the PyVista plotter from the figure

# Right hemisphere view (looking from the right to the left)
right_camera_position = [0.5, 0, 0]  # Adjust the x-coordinate as needed
focal_point = [0, 0, 0]
view_up = [0, 0, 1]

# Set and update for the right hemisphere view
plotter.camera_position = [right_camera_position, focal_point, view_up]
plotter.render()
plotter.screenshot(savepath2 + r'\glm\brainchanplots_Control_Sil_GLM_rh.png')

# Left hemisphere view (looking from the left to the right)
left_camera_position = [-0.5, 0, 0.07]  # Mirror the x-coordinate

# Set and update for the left hemisphere view
plotter.camera_position = [left_camera_position, focal_point, view_up]
plotter.render()
#plotter.screenshot(savepath2 + r'\glm\single channel analyses\brainchanplots_Control_Sil_GLM_lh.png')



# Define the ROI source-detector pair mappings based on the provided definitions
rois_pairs = {
    'lIFG': lIFG,
    'lSTG': lSTG,
    'rSTG': rSTG,
    'Left_SC': lSC,
    'Right_SC': rSC,
    'leftPostTemp': lpostTemp,
}

# Apply the function to each row in the DataFrame
df_lme_glmchanlong_ordered['ROI'] = df_lme_glmchanlong_ordered['ch_name'].apply(lambda x: determine_roi(x, rois_pairs))

ten_largest_betas = df_lme_glmchanlong_ordered.nlargest(10, 'Beta').round(1)

# brain plot
# extract significant ones and show significant ones only
df_lme_glmchanlong_ordered.loc[~df_lme_glmchanlong_ordered['significant?'], 'Beta'] = 0
non_zero_count = df_lme_glmchanlong_ordered[df_lme_glmchanlong_ordered['Beta'] != 0].shape[0]
positive_count = df_lme_glmchanlong_ordered[df_lme_glmchanlong_ordered['Beta'] > 0].shape[0]
negative_count = df_lme_glmchanlong_ordered[df_lme_glmchanlong_ordered['Beta'] < 0].shape[0]

mean_betas = df_lme_glmchanlong_ordered['Beta'].mean()
std_betas = df_lme_glmchanlong_ordered['Beta'].std()


#ten_largest_betas.to_csv(savepath2 + r'\glm\tenlargestbetas_T_Sil_Sil.csv')



#combine waveforma nd glm ten largest dataframes

ten_largest_betas['ch_name'] = ten_largest_betas['ch_name'].str.replace(' hbo', '')


#%% same brain channel plot for waveforms

# extract mean amps for each chan 

        # stats: generate data frame from waveform responses
df_waveformamplitudes_chans = pd.DataFrame(columns=['ID', 'ch_name', 'Chroma', 'Condition', 'Value'])

for idx, evoked in enumerate(all_evokeds_allCond):
    for subj_data in all_evokeds_allCond[evoked]:
        subj_id = subj_data.info['subject_info']['his_id'] if 'subject_info' in subj_data.info and 'his_id' in subj_data.info['subject_info'] else 'unknown'
        bads = subj_data.info['bads']
        for ch_name in subj_data.ch_names:
            if ch_name not in bads and ('hbo' in ch_name or 'hbr' in ch_name):
                chroma = 'hbo' if 'hbo' in ch_name else 'hbr'
                data = deepcopy(subj_data).pick(picks=[ch_name])
                value = data.crop(tmin=5.0, tmax=8.0).data.mean() * 1.0e6  
                
                # Append metadata and extracted feature to dataframe
                this_df = pd.DataFrame(
                    {'ID': subj_id, 'ch_name': ch_name.split(' ')[0], 'Chroma': chroma,
                     'Condition': evoked, 'Value': value}, index=[0])
                df_waveformamplitudes_chans = pd.concat([df_waveformamplitudes_chans, this_df], ignore_index=True)

df_waveformamplitudes_chans.reset_index(inplace=True, drop=True)
df_waveformamplitudes_chans['Value'] = pd.to_numeric(df_waveformamplitudes_chans['Value'])  # Convert Value column to numeric



# show one condition only
condition = "Condition in ['Control_Sil']" #"C("Condition in ['Speech_Sil','Noise_Sil', 'Control_Sil', 'A_SPIN_Sil', 'T_Sil','AT_SPIN_Sil', 'Speech_Noi', 'Control_Noi', 'T_Noi']")
chroma = "Chroma in ['hbo']"
chroma2 = 'hbo'

grp_results = df_waveformamplitudes_chans.query(condition)
grp_results = grp_results.query(chroma)

grp_results = grp_results.dropna(subset=['Value'])

chan_wavemodel = smf.mixedlm("Value ~ -1 + ch_name", 
                        grp_results, groups=grp_results["ID"], missing="ignore").fit(method='nm')
chan_wavemodel.summary()
df_lme_chan_wavemodel = statsmodels_to_results(chan_wavemodel)

# FDR correction
pvals = df_lme_chan_wavemodel['P>|z|'].values
pvals_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)[1]

# Add the corrected p-values back to the dataframe
df_lme_chan_wavemodel['p_FDRcorrected'] = pvals_corrected
df_lme_chan_wavemodel = df_lme_chan_wavemodel .rename(columns={'Coef.': 'Amplitude', 'P>|z|': 'p-value'})
df_lme_chan_wavemodel = df_lme_chan_wavemodel .drop(columns=['Std.Err.', 'z', '[0.025', '0.975]', 'Significant'])
df_lme_chan_wavemodel['significant?'] = df_lme_chan_wavemodel['p_FDRcorrected'] <= 0.05


# Apply this function to the specific columns you want to format
#df_lme_chan_wavemodel ['Beta'] = df_lme_chan_wavemodel ['Beta'].apply(lambda x: format_number(x))
    #small betas are replaced by string <0.001 which creates NaNs in plot)
df_lme_chan_wavemodel['p-value'] = df_lme_chan_wavemodel['p-value'].apply(lambda x: format_number(x))
df_lme_chan_wavemodel['p_FDRcorrected'] = df_lme_chan_wavemodel['p_FDRcorrected'].apply(lambda x: format_number(x))
#df_lme_chan_wavemodel.to_csv(savepath2 + r'\waveforms\channelresults_Control_Sil.csv')




# channel order different than in info! -> align so plot is correct
info = raw_haemo.copy().pick(chroma2).info
channelorder = info.ch_names

# Remove the chroma part from channelorder
channelorder = [ch.split(' ')[0] for ch in channelorder]

df_lme_chan_wavemodel_ordered = df_lme_chan_wavemodel.set_index('ch_name').reindex(channelorder).reset_index()


#df_lme_chan_wavemodel_ordered['radius'] = df_lme_chan_wavemodel_ordered['significant?'].apply(lambda x: 0.0017 if x else 0.001)
radius = df_lme_chan_wavemodel_ordered['significant?'].apply(lambda x: 0.0017 if x else 0.001)

fig= plot_nirs_source_detector_wAxixLims(
    df_lme_chan_wavemodel_ordered['Amplitude'], 
    raw_haemo.copy().pick('hbo').info , fnirs = True,
    radius= radius.values,
subject= 'fsaverage', coord_frame= 'head', trans = 'fsaverage', surfaces=['brain'],vmin = -4, vmax = 4)

plotter = fig.plotter  # Get the PyVista plotter from the figure

# Right hemisphere view (looking from the right to the left)
right_camera_position = [0.5, 0, 0]  # Adjust the x-coordinate as needed
focal_point = [0, 0, 0]
view_up = [0, 0, 1]

# Set and update for the right hemisphere view
plotter.camera_position = [right_camera_position, focal_point, view_up]
plotter.render()
plotter.screenshot(savepath2 + r'\waveforms\brainplots_Control_Sill_rh.png')
#plotter.screenshot(savepath2 + r'\waveforms\brainplots_TNoi_rh.png')

# Left hemisphere view (looking from the left to the right)
left_camera_position = [-0.5, 0, 0.07]  # Mirror the x-coordinate

# Set and update for the left hemisphere view
plotter.camera_position = [left_camera_position, focal_point, view_up]
plotter.render()
plotter.screenshot(savepath2 + r'\waveforms\brainplots_Control_Sil_lh.png')
#plotter.screenshot(savepath2 + r'\waveforms\brainplots_AT_SPIN_Sil_lh.png')





# Define the ROI source-detector pair mappings based on the provided definitions
rois_pairs = {
    'lIFG': lIFG,
    'lSTG': lSTG,
    'rSTG': rSTG,
    'Left_SC': lSC,
    'Right_SC': rSC,
    'leftPostTemp': lpostTemp,
}

# Apply the function to each row in the DataFrame
df_lme_chan_wavemodel_ordered['ROI'] = df_lme_chan_wavemodel_ordered['ch_name'].apply(lambda x: determine_roi(x, rois_pairs))
ten_largest_amps= df_lme_chan_wavemodel_ordered.nlargest(10, 'Amplitude').round(1)


#ten_largest_amps.to_csv(savepath2 + r'\waveforms\tenlargestwaves_AT_SPIN_Sil.csv')

# extract significant ones and show significant ones only
df_lme_chan_wavemodel_ordered.loc[~df_lme_chan_wavemodel_ordered['significant?'], 'Amplitude'] = 0
non_zero_count = df_lme_chan_wavemodel_ordered[df_lme_chan_wavemodel_ordered[ 'Amplitude'] != 0].shape[0]
positive_count = df_lme_chan_wavemodel_ordered[df_lme_chan_wavemodel_ordered[ 'Amplitude'] > 0].shape[0]
negative_count = df_lme_chan_wavemodel_ordered[df_lme_chan_wavemodel_ordered[ 'Amplitude'] < 0].shape[0]

mean_amps = df_lme_chan_wavemodel_ordered[ 'Amplitude'].mean()
std_amps = df_lme_chan_wavemodel_ordered[ 'Amplitude'].std()


Combined_tenlargestactivation = pd.merge(ten_largest_amps, ten_largest_betas, on='ch_name', suffixes=('_waveform', '_glm'))

# # Calculating the mean for 'Amplitude' and 'Beta'
# mean_values_largestacti = pd.DataFrame({
#     'ch_name': ['Mean'],
#     'Amplitude': [Combined_tenlargestactivation['Amplitude'].mean().round(1)],
#     'Beta': [Combined_tenlargestactivation['Beta'].mean().round(1)],
#     'Condition': ['Average'],
#     'p_FDR_waveform': [None],
#     'ROI_waveform': [None],
#     'p_FDR_glm': [None],
#     'ROI_glm': [None]
# })


# Combined_tenlargestactivation = pd.concat([Combined_tenlargestactivation, mean_values_largestacti], ignore_index=True)

#Combined_tenlargestactivation.to_csv(savepath2 + r'\tenlargestactivations_AT_SPIN_Sil.csv')

#%%% try surface proejction for waveforms
# try surface projection plot
# same error as for NH dataset and montage
df_lme_chan_wavemodel_ordered["ch_name"] = df_lme_chan_wavemodel_ordered["ch_name"]+' hbo'


clim = dict(kind='value', pos_lims=(0, 8, 11))
brain = mne_nirs.visualisation.plot_glm_surface_projection(raw_haemo.copy().pick("hbo"),
                                   df_lme_chan_wavemodel_ordered, clim=clim, view='dorsal',
                                    colorbar=True, size=(800, 700))
brain.add_text(0.05, 0.95, "Left-Right", 'title', font_size=16, color='k')

# SAME ERROR

#   File ~\AppData\Local\anaconda3\envs\mne\Lib\site-packages\mne\source_estimate.py:3901 in stc_near_sensors
#     nz_data = w @ evoked.data

# ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 49 is different from 44)

#%%% Repeat for contrasts of conditions on channel level
#%%% Channel results and brain plots
#%% brain plots and anlysis based on contrast models
# for 1.  with  and A-SPIQ - T-SPIQ modelled as intercept,
    #  A-Noise- T-SPIQ m
# for 2. with A-Noise and A-SPIQ, of which A-Noise served as intercept.
# for 3. with A-SPIN and A-SPIQ, of which A-SPIN served as intercept,
# for 4. with AT-SPIN  T-SPIQ, 
#       with AT-SPIN  vs A-SPIN



# ###############################alternatively hypothesis tests on single channel level
df_channelstats_Rename = df_channelstats.copy()
df_channelstats_Rename['Condition'] = df_channelstats_Rename['Condition'].replace('A_SPIN_Sil', 'AA_SPIN_Sil')
df_channelstats_Rename['Condition'] = df_channelstats_Rename['Condition'].replace('T_Sil', 'AA_T_Sil')

 ###################1a: 
#input_data = df_channelstats_Rename.query("Condition in ['Speech_Sil', 'Noise_Sil', 'B_T_Sil']") # same results when testing separate models? ['Speech_Sil','B_T_Sil']") ,  ['Noise_Sil', 'B_T_Sil']")  

input_data = df_channelstats_Rename.query("Condition in ['Speech_Sil', 'AA_T_Sil']") # 
### 1b
input_data = df_channelstats_Rename.query("Condition in ['Noise_Sil', 'AA_T_Sil']") # same results when testing separate models? ['Speech_Sil','B_T_Sil']") ,  ['Noise_Sil', 'B_T_Sil']")  


 ###################2 : A_Noise vs ASPIQ
input_data = df_channelstats_Rename.query("Condition in ['Speech_Sil', 'Noise_Sil']")  #'T'

 ###################3 A_SPIN vs ASPIQ
input_data = df_channelstats_Rename.query("Condition in ['Speech_Sil', 'A_SPIN_Sil']")  #'T'
 ###################4a: AT_SPIN vs A-SPIn
input_data = df_channelstats_Rename.query("Condition in ['AT_SPIN_Sil', 'AA_SPIN_Sil']") # 'T'
 ##                 b: AT_SPIN vs T-SPIQ
input_data = df_channelstats_Rename.query("Condition in ['AT_SPIN_Sil', 'AA_T_Sil']") # 'T'


chroma = "Chroma in ['hbo']"
chroma2 = 'hbo'
grp_results = input_data.query(chroma)
grp_results = grp_results.dropna(subset=['theta'])

chan_glmmodel = smf.mixedlm("theta ~ Condition:ch_name", 
                        grp_results, groups=grp_results["ID"], missing="ignore").fit(method='lbfgs')
chan_glmmodel.summary()
df_lme_glmchanlong = statsmodels_to_results(chan_glmmodel)

df_lme_glmchanlong= df_lme_glmchanlong[df_lme_glmchanlong.index.str.contains(':')]




# FDR correction
pvals = df_lme_glmchanlong['P>|z|'].values
pvals_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)[1]

# Add the corrected p-values back to the dataframe
df_lme_glmchanlong['p_FDRcorrected'] = pvals_corrected
df_lme_glmchanlong = df_lme_glmchanlong.rename(columns={'Coef.': 'Beta', 'P>|z|': 'p-value'})
df_lme_glmchanlong = df_lme_glmchanlong.drop(columns=['Std.Err.', 'z', '[0.025', '0.975]', 'Significant'])
df_lme_glmchanlong['significant?'] = df_lme_glmchanlong['p_FDRcorrected'] <= 0.05


# Apply this function to the specific columns you want to format
#df_lme_glmchanlong['Beta'] = df_lme_glmchanlong['Beta'].apply(lambda x: format_number(x))
    #small betas are replaced by string <0.001 which creates NaNs in plot)
df_lme_glmchanlong['p-value'] = df_lme_glmchanlong['p-value'].apply(lambda x: format_number(x))
df_lme_glmchanlong['p_FDRcorrected'] = df_lme_glmchanlong['p_FDRcorrected'].apply(lambda x: format_number(x))
#df_lme_glmchanlong.to_csv(savepath2 + r'\glm\single channel analyses\RQ1aSPIQvsTSPIQ.csv')




# theta different order than in info!
info = raw_haemo.copy().pick(chroma2).info
channelorder = info.ch_names

# reordering for contrast data frames
# Extract channel names from the index
df_lme_glmchanlong['ch_name'] = df_lme_glmchanlong.index.to_series().str.extract(r'ch_name\[(.*?)\]')[0]
df_lme_glmchanlong.set_index('ch_name', inplace=True)
df_lme_glmchanlong_ordered = df_lme_glmchanlong.reindex(channelorder)
df_lme_glmchanlong_ordered.reset_index(inplace=True)
radius =df_lme_glmchanlong_ordered['significant?'].apply(lambda x: 0.0017 if x else 0.001) # result turn wrong (wrong colors) when choosing thinner values!!!


fig= plot_nirs_source_detector_wAxixLims(
    df_lme_glmchanlong_ordered['Beta'], 
    raw_haemo.copy().pick('hbo').info , fnirs = True,
    radius= radius.values,
subject= 'fsaverage', coord_frame= 'head', trans = 'fsaverage', surfaces=['brain'],vmin = -1, vmax = 1) # changed to 1 to see contrast better

plotter = fig.plotter  # Get the PyVista plotter from the figure

# Right hemisphere view (looking from the right to the left)
right_camera_position = [0.5, 0, 0]  # Adjust the x-coordinate as needed
focal_point = [0, 0, 0]
view_up = [0, 0, 1]

# Set and update for the right hemisphere view
plotter.camera_position = [right_camera_position, focal_point, view_up]
plotter.render()
#plotter.screenshot(savepath2 + r'\glm\single channel analyses\contrastbrain_1aSPIQvsTSPIQ_GLM_rh.png')
#plotter.screenshot(savepath2 + r'\glm\single channel analyses\contrastbrain_1bANoisevsTSPIQ_GLM_rh.png')
#plotter.screenshot(savepath2 + r'\glm\single channel analyses\contrastbrain_2ASPIQvsANoise_GLM_rh.png')
#plotter.screenshot(savepath2 + r'\glm\single channel analyses\contrastbrain_3ASPIQvsASPIN_GLM_rh.png')
#plotter.screenshot(savepath2 + r'\glm\single channel analyses\contrastbrain_4ATsASPIN_GLM_rh.png')
plotter.screenshot(savepath2 + r'\glm\single channel analyses\contrastbrain_4ATvsT_GLM_rh.png')

# Left hemisphere view (looking from the left to the right)
left_camera_position = [-0.5, 0, 0.07]  # Mirror the x-coordinate

# Set and update for the left hemisphere view
plotter.camera_position = [left_camera_position, focal_point, view_up]
plotter.render()
#plotter.screenshot(savepath2 + r'\glm\single channel analyses\contrastbrain_1aSPIQvsTSPIQ_GLM_lh.png')
#plotter.screenshot(savepath2 + r'\glm\single channel analyses\contrastbrain_1bAnoisevsTSPIQ_GLM_lh.png')
#plotter.screenshot(savepath2 + r'\glm\single channel analyses\contrastbrain_2ASPIQvsANoise_GLM_lh.png')
#plotter.screenshot(savepath2 + r'\glm\single channel analyses\contrastbrain_3ASPIQvsASPIN_GLM_lh.png')
#plotter.screenshot(savepath2 + r'\glm\single channel analyses\contrastbrain_4ATsASPINN_GLM_lh.png')
plotter.screenshot(savepath2 + r'\glm\single channel analyses\contrastbrain_4ATsvsT_GLM_lh.png')



# Define the ROI source-detector pair mappings based on the provided definitions
rois_pairs = {
    'lIFG': lIFG,
    'lSTG': lSTG,
    'rSTG': rSTG,
    'Left_SC': lSC,
    'Right_SC': rSC,
    'leftPostTemp': lpostTemp,
}

# Apply the function to each row in the DataFrame
df_lme_glmchanlong_ordered['ROI'] = df_lme_glmchanlong_ordered['ch_name'].apply(lambda x: determine_roi(x, rois_pairs))

df_lme_glmchanlong_ordered['p-value'] = df_lme_glmchanlong_ordered['p-value'] .replace('<0.001', -0.001).astype(float)

topten_betas = df_lme_glmchanlong_ordered.nsmallest(10, 'p-value')

# brain plot
# extract significant ones and show significant ones only
# df_lme_glmchanlong_ordered.loc[~df_lme_glmchanlong_ordered['significant?'], 'Beta'] = 0
# non_zero_count = df_lme_glmchanlong_ordered[df_lme_glmchanlong_ordered['Beta'] != 0].shape[0]
# positive_count = df_lme_glmchanlong_ordered[df_lme_glmchanlong_ordered['Beta'] > 0].shape[0]
# negative_count = df_lme_glmchanlong_ordered[df_lme_glmchanlong_ordered['Beta'] < 0].shape[0]

# mean_betas = df_lme_glmchanlong_ordered['Beta'].mean()
# std_betas = df_lme_glmchanlong_ordered['Beta'].std()


#ten_largest_betas.to_csv(savepath2 + r'\glm\tenlargestbetas_T_Sil_Sil.csv')



#combine waveforma nd glm ten largest dataframes

#topten_betas['ch_name'] = topten_betas['ch_name'].str.replace(' hbo', '')


#%% same brain channel plot for waveforms

# extract mean amps for each chan 

        # stats: generate data frame from waveform responses
df_waveformamplitudes_chans = pd.DataFrame(columns=['ID', 'ch_name', 'Chroma', 'Condition', 'Value'])

for idx, evoked in enumerate(all_evokeds_allCond):
    for subj_data in all_evokeds_allCond[evoked]:
        subj_id = subj_data.info['subject_info']['his_id'] if 'subject_info' in subj_data.info and 'his_id' in subj_data.info['subject_info'] else 'unknown'
        bads = subj_data.info['bads']
        for ch_name in subj_data.ch_names:
            if ch_name not in bads and ('hbo' in ch_name or 'hbr' in ch_name):
                chroma = 'hbo' if 'hbo' in ch_name else 'hbr'
                data = deepcopy(subj_data).pick(picks=[ch_name])
                value = data.crop(tmin=5.0, tmax=8.0).data.mean() * 1.0e6  
                
                # Append metadata and extracted feature to dataframe
                this_df = pd.DataFrame(
                    {'ID': subj_id, 'ch_name': ch_name.split(' ')[0], 'Chroma': chroma,
                     'Condition': evoked, 'Value': value}, index=[0])
                df_waveformamplitudes_chans = pd.concat([df_waveformamplitudes_chans, this_df], ignore_index=True)

df_waveformamplitudes_chans.reset_index(inplace=True, drop=True)
df_waveformamplitudes_chans['Value'] = pd.to_numeric(df_waveformamplitudes_chans['Value'])  # Convert Value column to numeric



df_waveformamplitudes_chans_Rename = df_waveformamplitudes_chans.copy()
df_waveformamplitudes_chans_Rename['Condition'] =df_waveformamplitudes_chans_Rename['Condition'].replace('A_SPIN_Sil', 'AA_SPIN_Sil')
df_waveformamplitudes_chans_Rename['Condition'] = df_waveformamplitudes_chans_Rename['Condition'].replace('T_Sil', 'AA_T_Sil')

 ###################1a: 
#input_data = df_channelstats_Rename.query("Condition in ['Speech_Sil', 'Noise_Sil', 'B_T_Sil']") # same results when testing separate models? ['Speech_Sil','B_T_Sil']") ,  ['Noise_Sil', 'B_T_Sil']")  

input_data = df_waveformamplitudes_chans_Rename.query("Condition in ['Speech_Sil', 'AA_T_Sil']") # 
input_data = df_waveformamplitudes_chans_Rename.query("Condition in ['Noise_Sil', 'AA_T_Sil']") # same results when testing separate models? ['Speech_Sil','B_T_Sil']") ,  ['Noise_Sil', 'B_T_Sil']")  

 ###################2: A_Noise vs ASPIQ
input_data = df_waveformamplitudes_chans_Rename.query("Condition in ['Speech_Sil', 'Noise_Sil']")  #'T'

 ###################3: A_SPIN vs ASPIQ
input_data = df_waveformamplitudes_chans_Rename.query("Condition in ['Speech_Sil', 'AA_SPIN_Sil']")  #'T'


 ###################4a: AT_SPIN vs A-SPIn
input_data = df_waveformamplitudes_chans_Rename.query("Condition in ['AT_SPIN_Sil', 'AA_SPIN_Sil']") # 'T'
 ##               b: AT_SPIN vs T-SPIQ
input_data = df_waveformamplitudes_chans_Rename.query("Condition in ['AT_SPIN_Sil' ,'AA_T_Sil']") # 'T'


chroma = "Chroma in ['hbo']"
chroma2 = 'hbo'

grp_results = input_data
grp_results = grp_results.query(chroma)

grp_results = grp_results.dropna(subset=['Value'])

chan_wavemodel = smf.mixedlm("Value ~ Condition:ch_name", 
                        grp_results, groups=grp_results["ID"], missing="ignore").fit(method='nm')
chan_wavemodel.summary()
df_lme_chan_wavemodel = statsmodels_to_results(chan_wavemodel)


df_lme_chan_wavemodel = df_lme_chan_wavemodel[df_lme_chan_wavemodel.index.str.contains(':')]


# FDR correction
pvals = df_lme_chan_wavemodel['P>|z|'].values
pvals_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)[1]

# Add the corrected p-values back to the dataframe
df_lme_chan_wavemodel['p_FDRcorrected'] = pvals_corrected
df_lme_chan_wavemodel = df_lme_chan_wavemodel .rename(columns={'Coef.': 'Amplitude', 'P>|z|': 'p-value'})
df_lme_chan_wavemodel = df_lme_chan_wavemodel .drop(columns=['Std.Err.', 'z', '[0.025', '0.975]', 'Significant'])
df_lme_chan_wavemodel['significant?'] = df_lme_chan_wavemodel['p_FDRcorrected'] <= 0.05


# Apply this function to the specific columns you want to format
#df_lme_chan_wavemodel ['Beta'] = df_lme_chan_wavemodel ['Beta'].apply(lambda x: format_number(x))
    #small betas are replaced by string <0.001 which creates NaNs in plot)
df_lme_chan_wavemodel['p-value'] = df_lme_chan_wavemodel['p-value'].apply(lambda x: format_number(x))
df_lme_chan_wavemodel['p_FDRcorrected'] = df_lme_chan_wavemodel['p_FDRcorrected'].apply(lambda x: format_number(x))
#df_lme_chan_wavemodel.to_csv(savepath2 + r'\waveforms\channelresults_Control_Sil.csv')



# channel order different than in info! -> align so plot is correct
info = raw_haemo.copy().pick(chroma2).info
channelorder = info.ch_names
###############################################continue from here

# reordering for contrast data frames
# Extract channel names from the index

df_lme_chan_wavemodel['ch_name'] = df_lme_chan_wavemodel.index.to_series().str.extract(r'ch_name\[(.*?)\]')[0]
df_lme_chan_wavemodel['ch_name'] = df_lme_chan_wavemodel['ch_name']+ ' hbo'
df_lme_chan_wavemodel.set_index('ch_name', inplace=True)

df_lme_chan_wavemodel_ordered = df_lme_chan_wavemodel.reindex(channelorder)
df_lme_chan_wavemodel_ordered.reset_index(inplace=True)
radius =df_lme_chan_wavemodel_ordered['significant?'].apply(lambda x: 0.0017 if x else 0.001) # result turn wrong (wrong colors) when choosing thinner values!!!


fig= plot_nirs_source_detector_wAxixLims(
    df_lme_chan_wavemodel_ordered['Amplitude'], 
    raw_haemo.copy().pick('hbo').info , fnirs = True,
    radius= radius.values,
subject= 'fsaverage', coord_frame= 'head', trans = 'fsaverage', surfaces=['brain'],vmin = -4, vmax = 4)

plotter = fig.plotter  # Get the PyVista plotter from the figure

# Right hemisphere view (looking from the right to the left)
right_camera_position = [0.5, 0, 0]  # Adjust the x-coordinate as needed
focal_point = [0, 0, 0]
view_up = [0, 0, 1]

# Set and update for the right hemisphere view
plotter.camera_position = [right_camera_position, focal_point, view_up]
plotter.render()
#plotter.screenshot(savepath2 + r'\waveforms\single channel analyses\contrastbrain_1aSPIQvsTSPIQ_GLM_rh.png')
#plotter.screenshot(savepath2 + r'\waveforms\single channel analyses\contrastbrain_1bANoisesTSPIQ_GLM_rh.png')
#plotter.screenshot(savepath2 + r'\waveforms\single channel analyses\contrastbrain_2ASPIQvsANoise_GLM_rh.png')
#plotter.screenshot(savepath2 + r'\waveforms\single channel analyses\contrastbrain_3ASPIQvsASPIN_GLM_rh.png')
#plotter.screenshot(savepath2 + r'\waveforms\single channel analyses\contrastbrain_4ATvsASPIN_GLM_rh.png')
plotter.screenshot(savepath2 + r'\waveforms\single channel analyses\contrastbrain_4ATvsT_GLM_rh.png')


# Left hemisphere view (looking from the left to the right)
left_camera_position = [-0.5, 0, 0.07]  # Mirror the x-coordinate

# Set and update for the left hemisphere view
plotter.camera_position = [left_camera_position, focal_point, view_up]
plotter.render()
#plotter.screenshot(savepath2 + r'\waveforms\single channel analyses\contrastbrain_1aSPIQvsTSPIQ_GLM_lh.png')
#plotter.screenshot(savepath2 + r'\waveforms\single channel analyses\contrastbrain_1bANOisevsTSPIQ_GLM_lh.png')
#plotter.screenshot(savepath2 + r'\waveforms\single channel analyses\contrastbrain_2ASPIQvsANoise_GLM_lh.png')
#plotter.screenshot(savepath2 + r'\waveforms\single channel analyses\contrastbrain_3ASPIQvsASPIN_GLM_lh.png')
#plotter.screenshot(savepath2 + r'\waveforms\single channel analyses\contrastbrain_4ATvsASPIN_GLM_lh.png')
plotter.screenshot(savepath2 + r'\waveforms\single channel analyses\contrastbrain_4ATvsT_GLM_lh.png')

# all called glm but they are not



# Define the ROI source-detector pair mappings based on the provided definitions
rois_pairs = {
    'lIFG': lIFG,
    'lSTG': lSTG,
    'rSTG': rSTG,
    'Left_SC': lSC,
    'Right_SC': rSC,
    'leftPostTemp': lpostTemp,
}

# Apply the function to each row in the DataFrame
df_lme_chan_wavemodel_ordered['ROI'] = df_lme_chan_wavemodel_ordered['ch_name'].apply(lambda x: determine_roi(x, rois_pairs))


df_lme_chan_wavemodel_ordered['p-value'] = df_lme_chan_wavemodel_ordered['p-value'] .replace('<0.001', -0.001).astype(float)


topten_amps= df_lme_chan_wavemodel_ordered.nsmallest(10, 'p-value')


#ten_largest_amps.to_csv(savepath2 + r'\waveforms\tenlargestwaves_AT_SPIN_Sil.csv')

# # extract significant ones and show significant ones only
# df_lme_chan_wavemodel_ordered.loc[~df_lme_chan_wavemodel_ordered['significant?'], 'Amplitude'] = 0
# non_zero_count = df_lme_chan_wavemodel_ordered[df_lme_chan_wavemodel_ordered[ 'Amplitude'] != 0].shape[0]
# positive_count = df_lme_chan_wavemodel_ordered[df_lme_chan_wavemodel_ordered[ 'Amplitude'] > 0].shape[0]
# negative_count = df_lme_chan_wavemodel_ordered[df_lme_chan_wavemodel_ordered[ 'Amplitude'] < 0].shape[0]

# mean_amps = df_lme_chan_wavemodel_ordered[ 'Amplitude'].mean()
# std_amps = df_lme_chan_wavemodel_ordered[ 'Amplitude'].std()





Combined_toptensig= pd.merge(topten_amps, topten_betas, on='ch_name', suffixes=('_waveform', '_glm'))
Combined_toptensig = Combined_toptensig.drop(axis = "columns", labels =["Intercept_waveform", "ROI_waveform","Intercept_glm","significant?_glm","significant?_waveform"])
Combined_toptensig["Amplitude"] = Combined_toptensig["Amplitude"].round(2)
Combined_toptensig["Beta"] = Combined_toptensig["Beta"].round(2)
# # Calculating the mean for 'Amplitude' and 'Beta'
# mean_values_largestacti = pd.DataFrame({
#     'ch_name': ['Mean'],
#     'Amplitude': [Combined_tenlargestactivation['Amplitude'].mean().round(1)],
#     'Beta': [Combined_tenlargestactivation['Beta'].mean().round(1)],
#     'Condition': ['Average'],
#     'p_FDR_waveform': [None],
#     'ROI_waveform': [None],
#     'p_FDR_glm': [None],
#     'ROI_glm': [None]
# })



#Combined_toptensig.to_csv(savepath2 + r'\singlechananalysis_tenmostsig_1a_ASPIQvsTSPIQ.csv')
#Combined_toptensig.to_csv(savepath2 + r'\singlechananalysis_tenmostsig_1b_ANoisevsTSPIQ.csv')
#Combined_toptensig.to_csv(savepath2 + r'\singlechananalysis_tenmostsig_2ASPIQvsANoise.csv')
#Combined_toptensig.to_csv(savepath2 + r'\singlechananalysis_tenmostsig_3ASPIQvsASPIN.csv')
#Combined_toptensig.to_csv(savepath2 + r'\singlechananalysis_tenmostsig_4ATvsASPIN.csv')
Combined_toptensig.to_csv(savepath2 + r'\singlechananalysis_tenmostsig_4ATvsT.csv')









#%% second roi analysis AT-A-T
#%% in SPIQ_ROI

spiq_r = [[15,12], [5,2],[14,13],[15,9],[4,4],[14,10],[3,2],[13,13],[3,4]] 
rois = dict(spiq_roi=picks_pair_to_idx(raw_haemo,spiq_r , on_missing ='ignore')) # overwriting old rois to be able to reuse code
          
#%% 1. waveforms
# a) 

        # stats: generate data frame from waveform responses
df_waveformamplitudes = pd.DataFrame(columns=['ID', 'ROI', 'Chroma', 'Condition', 'Value'])

for idx, evoked in enumerate(all_evokeds_allCond):
    for subj_data in all_evokeds_allCond[evoked]:
        subj_id = subj_data.info['subject_info']['his_id'] if 'subject_info' in subj_data.info and 'his_id' in subj_data.info['subject_info'] else 'unknown'
        bads = subj_data.info['bads']
        for roi, picks in rois.items():
            for chroma in ["hbo", "hbr"]:
                # Exclude bad channels from the picks
                good_picks = [pick for pick in picks if subj_data.ch_names[pick] not in bads]
                if good_picks:  # Proceed only if there are good channels
                    data = deepcopy(subj_data).pick(picks=good_picks).pick(chroma)
                    value = data.crop(tmin=5.0, tmax=8.0).data.mean() * 1.0e6  
                    
                    # Append metadata and extracted feature to dataframe
                    this_df = pd.DataFrame(
                        {'ID': subj_id, 'ROI': roi, 'Chroma': chroma,
                        'Condition': evoked, 'Value': value}, index=[0])
                    df_waveformamplitudes = pd.concat([df_waveformamplitudes, this_df], ignore_index=True)

df_waveformamplitudes.reset_index(inplace=True, drop=True)
df_waveformamplitudes['Value'] = pd.to_numeric(df_waveformamplitudes['Value'])



pd.set_option('display.float_format', '{:.2e}'.format)



############################################################## group level comparisons: AT larger than A or T?

input_data = df_waveformamplitudes.query("Condition in ['AT_SPIN_Sil', 'A_SPIN_Sil','T_Sil']")  #'T'
input_data = input_data.query("Chroma in ['hbo']")
roi_model = smf.mixedlm("Value ~ Condition", input_data, groups=input_data["ID"]).fit()

roi_model.summary()
df_lme_waveslongchans= statsmodels_to_results(roi_model)

# larger than T but not larger than A

############ MSI # test for max criterion
# more correcter appraoch: finding max or sum on first level data first before adding into model. 

# 1) find max first, or just testing model subtracting at from both
input_data = df_waveformamplitudes[(df_waveformamplitudes['Chroma'] == 'hbo') & (df_waveformamplitudes['Condition'].isin(['AT_SPIN_Sil', 'A_SPIN_Sil', 'T_Sil']))]
input_data = input_data.pivot_table(index=['ID', 'Chroma', ''], columns='Condition', values='Value', aggfunc='max').reset_index()

# Compute the max between 'A' and 'T'
input_data['Max_A_T'] = input_data[['A_SPIN_Sil', 'T_Sil']].max(axis=1)

# Determine which condition had the max
input_data['max_cond'] = np.where(input_data['A_SPIN_Sil'] >= input_data['T_Sil'], 'A_SPIN_Sil', 'T_Sil')

# Count the number of occurrences for each condition
max_a_count = (input_data['max_cond'] == 'A_SPIN_Sil').sum()
max_t_count = (input_data['max_cond'] == 'T_Sil').sum()

input_data = input_data.melt(id_vars=['ID', 'Chroma'], value_vars=['Max_A_T', 'AT_SPIN_Sil'],
                                   var_name='Condition', value_name='Value')



input_data = input_data.query("Condition in ['AT_SPIN_Sil', 'Max_A_T']")  #'T'
roi_model = smf.mixedlm("Value ~ Condition", input_data, groups=input_data["ID"]).fit()

roi_model.summary()
df_lme_waveslongchans= statsmodels_to_results(roi_model)

#no max effect


#%% betas
# -> needs rerunning the first level analysis with new ROI or grouping results from df_channelstats for selected channels

## preprateion to get the ROI
spiq_roi_names = [f'S{pair[0]}_D{pair[1]} hbo' for pair in spiq_r]
# Filter the DataFrame for the specified conditions and channels
df_glm_spiqroichans = df_channelstats[
    (df_channelstats['Chroma'] == 'hbo') &
    (df_channelstats['ch_name'].isin(spiq_roi_names)) &
    (df_channelstats['Condition'].isin(['AT_SPIN_Sil', 'A_SPIN_Sil', 'T_Sil']))]
df_glm_spiqroichans['ROI'] = 'spiq_roi'
# Compute the mean beta value per condition within the ROI
df_glm_spiqroi = df_glm_spiqroichans.groupby(['Condition', 'ROI','ID']).agg(
    ROI_mean=('theta', 'mean')).reset_index()

# now equivalent code to above for waveforms to define max
# 1) find max first
input_data = df_glm_spiqroi.pivot_table(index=['ID'], columns='Condition', values='ROI_mean', aggfunc='max').reset_index()
# Compute the max between 'A' and 'T'
input_data['Max_A_T'] = input_data[['A_SPIN_Sil', 'T_Sil']].max(axis=1)
# Determine which condition had the max
input_data['max_cond'] = np.where(input_data['A_SPIN_Sil'] >= input_data['T_Sil'], 'A_SPIN_Sil', 'T_Sil')

# Count the number of occurrences for each condition
max_a_count = (input_data['max_cond'] == 'A_SPIN_Sil').sum()
max_t_count = (input_data['max_cond'] == 'T_Sil').sum()

input_data = input_data.melt(id_vars=['ID'], value_vars=['Max_A_T', 'AT_SPIN_Sil'],
                                   var_name='Condition', value_name='Value')

input_data = input_data.query("Condition in ['AT_SPIN_Sil', 'Max_A_T']")  #'T'
roi_model = smf.mixedlm("Value ~ Condition", input_data, groups=input_data["ID"]).fit()

roi_model.summary()
df_lme_GLM_SPIQROI= statsmodels_to_results(roi_model)

# no max effect AT is 0.35 mm larger, p =0.13


#1 AT larger than A, AT?
input_data = df_glm_spiqroi.query("Condition in ['AT_SPIN_Sil', 'A_SPIN_Sil','T_Sil']")  #'T'
roi_model = smf.mixedlm("ROI_mean ~ Condition", input_data, groups=input_data["ID"]).fit()

roi_model.summary()
df_lme_GLMSPIQROI= statsmodels_to_results(roi_model)

# larger than T yes, but not A








## superadditivity not even worth testing..


# # to test for superadditivity: add unisensory reponses
# df_waveformamplitudes_Max = df_waveformamplitudes.copy()
# # Rename conditions
# df_waveformamplitudes_Max['Condition'] = df_waveformamplitudes_Max['Condition'].replace({'A_SPIN': 'Unisensory', 'T': 'Unisensory', 'AT_SPIN': 'Multisensory'})
# # Create a subset for 'Unisensory' data
# unisensory_data = df_waveformamplitudes_MSI.query("Condition == 'Unisensory'")
# # Group by 'ID', 'ROI', 'Chroma', 'Condition' and sum the 'Value'
# unisensory_grouped = unisensory_data.groupby(['ID', 'ROI', 'Chroma', 'Condition'], as_index=False)['Value'].sum()

# # Filter out the original 'A_SPIN' and 'T' rows
# df_waveformamplitudes_MSI = df_waveformamplitudes_MSI.query("Condition == 'Multisensory'")

# # Append the new 'Unisensory' data to the dataframe
# df_waveformamplitudes_MSI_final = pd.concat([df_waveformamplitudes_MSI, unisensory_grouped], ignore_index=True)







# 2) superadditivity




# to test for superadditivity: add unisensory reponses
df_waveformamplitudes_MSI = df_waveformamplitudes.copy()
# Rename conditions
df_waveformamplitudes_MSI['Condition'] = df_waveformamplitudes_MSI['Condition'].replace({'A_SPIN': 'Unisensory', 'T': 'Unisensory', 'AT_SPIN': 'Multisensory'})
# Create a subset for 'Unisensory' data
unisensory_data = df_waveformamplitudes_MSI.query("Condition == 'Unisensory'")
# Group by 'ID', 'ROI', 'Chroma', 'Condition' and sum the 'Value'
unisensory_grouped = unisensory_data.groupby(['ID', 'ROI', 'Chroma', 'Condition'], as_index=False)['Value'].sum()

# Filter out the original 'A_SPIN' and 'T' rows
df_waveformamplitudes_MSI = df_waveformamplitudes_MSI.query("Condition == 'Multisensory'")

# Append the new 'Unisensory' data to the dataframe
df_waveformamplitudes_MSI_final = pd.concat([df_waveformamplitudes_MSI, unisensory_grouped], ignore_index=True)




# ## snippet from previous approach for superadditivity (check C:\GitHub\AudTacLocalizer\Analysis scripts\final pipelines\MNEchanreject\Groupanalysis_finalpipeline_RQ3_ATspeech_MNEbadchanreject.py)

# # Filter A_SPIN and T conditions
# a_spin_data = df_waveformamplitudes.query("Condition == 'A_SPIN_Sil' and Chroma == 'hbo' and ROI == 'spiq_roi'")  #!!
# t_data = df_waveformamplitudes.query("Condition == 'T_Sil' and Chroma == 'hbo' and ROI == 'spiq_roi'")#         !!!!


# # a_spin_data = df_waveformamplitudes.query("Condition == 'A_SPIN' and Chroma == 'hbo' and ROI == 'ATROI_waves'")
# # t_data = df_waveformamplitudes.query("Condition == 'T' and Chroma == 'hbo' and ROI == 'ATROI_waves'")

# # Merge the two dataframes on ID to compare their values
# comparison_df = a_spin_data.merge(t_data, on='ID', suffixes=('_A', '_T'))

# # Check if A_SPIN value is always greater or equal to T value
# comparison_df['A_greater_T'] = comparison_df['Value_A'] >= comparison_df['Value_T']
# if comparison_df['A_greater_T'].all():
#     print("A_SPIN is always greater or equal to T.")
# else:
#     print("There are cases where T is greater than A_SPIN.")



# # Create a copy of the original dataframe
# df_waveformamplitudes_MSI = df_waveformamplitudes.copy()

# # Map the conditions to a common group for aggregation while keeping the original condition
# df_waveformamplitudes_MSI['MSI_Condition'] = df_waveformamplitudes_MSI['Condition'].replace({'A_SPIN_Sil': 'Unisensory', 'T_Sil': 'Unisensory', 'AT_SPIN_Sil': 'Multisensory'}) 
# # For Unisensory data, find the row with the maximum value for each group and keep the original condition
# unisensory_max = df_waveformamplitudes_MSI[df_waveformamplitudes_MSI['MSI_Condition'] == 'Unisensory'].groupby(['ID', 'ROI', 'Chroma'], as_index=False).apply(lambda x: x.loc[x['Value'].idxmax()])

# # Filter to keep only Multisensory condition data
# multisensory_data = df_waveformamplitudes_MSI[df_waveformamplitudes_MSI['Condition'] == 'AT_SPIN_Sil']

# # Combine the maximum Unisensory data with the Multisensory data
# df_waveformamplitudes_MSI_final = pd.concat([unisensory_max, multisensory_data], ignore_index=True)#.drop(columns=[])


# input_data = df_waveformamplitudes_MSI_final.query("MSI_Condition in ['Multisensory', 'Unisensory']")  
# input_data = input_data.query("Chroma in ['hbo']")
# input_data = input_data.query("ROI in ['spiq_roi']") # ATROI_waves
# roi_model = smf.mixedlm("Value ~ MSI_Condition", input_data, groups=input_data["ID"]).fit()

# roi_model.summary()
# df_lme_waveslongchans= statsmodels_to_results(roi_model)


#%% what now? mean criterion? find channels that do exceed max or A significatnly?

#%%% #%% in which channels AT larger than A - waveforms / 

condition = "Condition in ['A_SPIN_Sil', 'AT_SPIN_Sil']" 
chroma = "Chroma in ['hbo']"
chroma2 = 'hbo'

grp_results = df_waveformamplitudes_chans.query(condition)
grp_results = grp_results.query(chroma)

grp_results = grp_results.dropna(subset=['Value'])

chan_wavemodel = smf.mixedlm("Value ~ Condition:ch_name", 
                        grp_results, groups=grp_results["ID"], missing="ignore").fit(method='nm')
chan_wavemodel.summary()
df_lme_chan_wavemodel = statsmodels_to_results(chan_wavemodel)

# FDR correction
pvals = df_lme_chan_wavemodel['P>|z|'].values
pvals_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)[1]

# Add the corrected p-values back to the dataframe
df_lme_chan_wavemodel['p_FDRcorrected'] = pvals_corrected
df_lme_chan_wavemodel = df_lme_chan_wavemodel .rename(columns={'Coef.': 'Amplitude', 'P>|z|': 'p-value'})
df_lme_chan_wavemodel = df_lme_chan_wavemodel .drop(columns=['Std.Err.', 'z', '[0.025', '0.975]', 'Significant'])
df_lme_chan_wavemodel['significant?'] = df_lme_chan_wavemodel['p_FDRcorrected'] <= 0.05


# Apply this function to the specific columns you want to format
#df_lme_chan_wavemodel ['Beta'] = df_lme_chan_wavemodel ['Beta'].apply(lambda x: format_number(x))
    #small betas are replaced by string <0.001 which creates NaNs in plot)
df_lme_chan_wavemodel['p-value'] = df_lme_chan_wavemodel['p-value'].apply(lambda x: format_number(x))
df_lme_chan_wavemodel['p_FDRcorrected'] = df_lme_chan_wavemodel['p_FDRcorrected'].apply(lambda x: format_number(x))

# --> only main chanel effects, no interaction effects# > nothing

# ######################alterntaively test in which AT larger than max? -> not worth it if previous one did not result in any channels 

# 1) find max first, or just testing model subtracting at from both
input_data = df_waveformamplitudes_chans[(df_waveformamplitudes_chans['Chroma'] == 'hbo') & (df_waveformamplitudes_chans['Condition'].isin(['AT_SPIN_Sil', 'A_SPIN_Sil', 'T_Sil']))]
input_data = input_data.pivot_table(index=['ID', 'Chroma','ch_name'], columns='Condition', values='Value', aggfunc='max').reset_index()

# Compute the max between 'A' and 'T'
input_data['Max_A_T'] = input_data[['A_SPIN_Sil', 'T_Sil']].max(axis=1)

# Determine which condition had the max
input_data['max_cond'] = np.where(input_data['A_SPIN_Sil'] >= input_data['T_Sil'], 'A_SPIN_Sil', 'T_Sil')

# Count the number of occurrences for each condition
max_a_count = (input_data['max_cond'] == 'A_SPIN_Sil').sum() #496
max_t_count = (input_data['max_cond'] == 'T_Sil').sum() # 486 
# on channel level, half half in which A or T is max

input_data = input_data.melt(id_vars=['ID', 'Chroma','ch_name'], value_vars=['Max_A_T', 'AT_SPIN_Sil'],
                                    var_name='Condition', value_name='Value')



input_data = input_data.query("Condition in ['AT_SPIN_Sil', 'Max_A_T']")  #'T'
roi_model = smf.mixedlm("Value ~ Condition:ch_name", input_data, groups=input_data["ID"]).fit()

roi_model.summary()
df_lme_waveslongchans= statsmodels_to_results(roi_model)



###### betas


condition = "Condition in ['A_SPIN_Sil', 'AT_SPIN_Sil']" 
chroma = "Chroma in ['hbo']"
chroma2 = 'hbo'

grp_results = df_channelstats.query(condition)
grp_results = grp_results.query(chroma)

grp_results = grp_results.dropna(subset=['theta'])

chan_glmmodel = smf.mixedlm("theta ~ Condition:ch_name", 
                        grp_results, groups=grp_results["ID"], missing="ignore").fit(method='nm')
chan_glmmodel.summary()
df_lme_glmchanlong = statsmodels_to_results(chan_glmmodel)

# FDR correction
pvals = df_lme_glmchanlong['P>|z|'].values
pvals_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)[1]

# Add the corrected p-values back to the dataframe
df_lme_glmchanlong['p_FDRcorrected'] = pvals_corrected
df_lme_glmchanlong = df_lme_glmchanlong.rename(columns={'Coef.': 'Beta', 'P>|z|': 'p-value'})
df_lme_glmchanlong = df_lme_glmchanlong.drop(columns=['Std.Err.', 'z', '[0.025', '0.975]', 'Significant'])
df_lme_glmchanlong['significant?'] = df_lme_glmchanlong['p_FDRcorrected'] <= 0.05


# Apply this function to the specific columns you want to format
#df_lme_glmchanlong['Beta'] = df_lme_glmchanlong['Beta'].apply(lambda x: format_number(x))
    #small betas are replaced by string <0.001 which creates NaNs in plot)
df_lme_glmchanlong['p-value'] = df_lme_glmchanlong['p-value'].apply(lambda x: format_number(x))
df_lme_glmchanlong['p_FDRcorrected'] = df_lme_glmchanlong['p_FDRcorrected'].apply(lambda x: format_number(x))
#df_lme_glmchanlong.to_csv(savepath2 + r'\glm\channelbetaresults_AT_SPIN_Sil.csv')


########### betas max criterion

# 1) find max first, or just testing model subtracting at from both
input_data = df_channelstats[(df_channelstats['Chroma'] == 'hbo') & (df_channelstats['Condition'].isin(['AT_SPIN_Sil', 'A_SPIN_Sil', 'T_Sil']))]
input_data = input_data.pivot_table(index=['ID', 'Chroma','ch_name'], columns='Condition', values='theta', aggfunc='max').reset_index()

# Compute the max between 'A' and 'T'
input_data['Max_A_T'] = input_data[['A_SPIN_Sil', 'T_Sil']].max(axis=1)

# Determine which condition had the max
input_data['max_cond'] = np.where(input_data['A_SPIN_Sil'] >= input_data['T_Sil'], 'A_SPIN_Sil', 'T_Sil')

# Count the number of occurrences for each condition
max_a_count = (input_data['max_cond'] == 'A_SPIN_Sil').sum() #496
max_t_count = (input_data['max_cond'] == 'T_Sil').sum() # 486 
# on channel level, half half in which A or T is max

input_data = input_data.melt(id_vars=['ID', 'Chroma','ch_name'], value_vars=['Max_A_T', 'AT_SPIN_Sil'],
                                    var_name='Condition', value_name='theta')



input_data = input_data.query("Condition in ['AT_SPIN_Sil', 'Max_A_T']")  #'T'
roi_model = smf.mixedlm("theta ~ Condition:ch_name", input_data, groups=input_data["ID"]).fit()

roi_model.summary()
df_lme_glmlongchans= statsmodels_to_results(roi_model)









## plotting

# channel order different than in info! -> align so plot is correct
info = raw_haemo.copy().pick(chroma2).info
channelorder = info.ch_names

# Remove the chroma part from channelorder
channelorder = [ch.split(' ')[0] for ch in channelorder]

df_lme_chan_wavemodel_ordered = df_lme_chan_wavemodel.set_index('ch_name').reindex(channelorder).reset_index()


radius =df_lme_chan_wavemodel_ordered['significant?'].apply(lambda x: 0.0017 if x else 0.001) # result turn wrong (wrong colors) when choosing thinner values!!!

# can we plot contrast? --> worked for TSil vs Tnoi
fig= plot_nirs_source_detector_wAxixLims(
    df_lme_chan_wavemodel_ordered['Amplitude'], 
    raw_haemo.copy().pick('hbo').info , fnirs = True,
    radius= radius.values,
subject= 'fsaverage', coord_frame= 'head', trans = 'fsaverage', surfaces=['brain'],vmin = -4, vmax = 4)

plotter = fig.plotter  # Get the PyVista plotter from the figure

# Right hemisphere view (looking from the right to the left)
right_camera_position = [0.5, 0, 0]  # Adjust the x-coordinate as needed
focal_point = [0, 0, 0]
view_up = [0, 0, 1]

# Set and update for the right hemisphere view
plotter.camera_position = [right_camera_position, focal_point, view_up]
plotter.render()
plotter.screenshot(savepath + r'\waveforms\brainplots_T_Sil_right_hemisphere_view.png')

# Left hemisphere view (looking from the left to the right)
left_camera_position = [-0.5, 0, 0.07]  # Mirror the x-coordinate

# Set and update for the left hemisphere view
plotter.camera_position = [left_camera_position, focal_point, view_up]
plotter.render()
plotter.screenshot(savepath + r'\waveforms\brainplots_T_Sil_left_hemisphere_view.png')





# Define the ROI source-detector pair mappings based on the provided definitions
rois_pairs = {
    'lIFG': lIFG,
    'lSTG': lSTG,
    'rSTG': rSTG,
    'Left_SC': lSC,
    'Right_SC': rSC,
    'leftPostTemp': lpostTemp,
}

# Apply the function to each row in the DataFrame
df_lme_chan_wavemodel_ordered['ROI'] = df_lme_chan_wavemodel_ordered['ch_name'].apply(lambda x: determine_roi(x, rois_pairs))
three_largest_amps= df_lme_chan_wavemodel_ordered.nlargest(10, 'Amplitude')


df_lme_chan_wavemodel_ordered.to_csv(savepath + r'\waveforms\df_RQ3_chanstats_A_SPIN_10largestamps.csv')


# extract significant ones and show significant ones only
df_lme_chan_wavemodel_ordered.loc[~df_lme_chan_wavemodel_ordered['significant?'], 'Amplitude'] = 0
non_zero_count = df_lme_chan_wavemodel_ordered[df_lme_chan_wavemodel_ordered[ 'Amplitude'] != 0].shape[0]
positive_count = df_lme_chan_wavemodel_ordered[df_lme_chan_wavemodel_ordered[ 'Amplitude'] > 0].shape[0]
negative_count = df_lme_chan_wavemodel_ordered[df_lme_chan_wavemodel_ordered[ 'Amplitude'] < 0].shape[0]

mean_betas = df_lme_chan_wavemodel_ordered[ 'Amplitude'].mean()
std_betas = df_lme_chan_wavemodel_ordered[ 'Amplitude'].std()










