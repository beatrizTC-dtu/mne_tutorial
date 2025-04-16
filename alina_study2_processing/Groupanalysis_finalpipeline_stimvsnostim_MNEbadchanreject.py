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


"""
Created on Mon Mar 11 15:10:31 2024
@author: aicu

Group analysis script for audio-tactile localizer data:
- Same preprocessing for glm and waveforms with short channels being scaled by separate glm and then subtracted 
- bad channels are cleaned as defined in matlab

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

# choose stimulus grouping 
stimrenaming = {'1.0': 'Stimulus', '2.0':'Stimulus', '3.0':'No Stimulus', '4.0':'Stimulus', '5.0': 'Stimulus','6.0': 'Stimulus', '7.0': 'Stimulus', '8.0': 'No Stimulus'}
T_Noi_rename = {'9.0': 'Stimulus'}

# stimrenaming = {'1.0': 'Speech_Sil', '2.0':'Noise_Sil', '3.0':'Control_Sil', '4.0':'A_SPIN_Sil', '5.0': 'T_Sil','6.0': 'AT_SPIN_Sil', '7.0': 'Speech_Noi', '8.0': 'Control_Noi'}
# T_Noi_rename = {'9.0': 'T_Noi'}

# stimrenaming = {'1.0': 'Speech', '2.0':'Noise', '3.0':'Control', '4.0':'Speech', '5.0': 'Tactile','6.0': 'Speech', '7.0': 'Speech', '8.0': 'Control'}
# T_Noi_rename = {'9.0': 'Tactile'}

plots = True
glm = True
waveform = True

df_channelstats = pd.DataFrame() 
df_roiglm = pd.DataFrame() 
df_shortchanglm = pd.DataFrame() 
all_evokeds_allCond = defaultdict(list)
all_evokeds_TempRoi = defaultdict(list) # all cond but only temp channels epoched
all_evokeds_ShortChans = defaultdict(list)

epochrejectCutOffs =[]
nr_of_bad_epochs = []
bad_channels =[]

normalization = False # only for NAHA breathing task is available

filelist = [r'01\2023-04-18_002',r'02\2023-04-19_001',r'03\2023-04-21_001',r'04\2023-04-24_001',r'07\2023-05-04_001',r'08\2023-12-18_002',r'09\2023-12-21_001',r'10\2024-01-04_001', r'11\2024-01-09_001',r'12\2024-01-16_003',r'13\2024-01-19_001',r'14\2024-01-19_002',r'15\2024-01-23_001',r'16\2024-01-23_002',r'17\2024-01-24_003',  r'19\2024-01-26_002',  r'20\2024-01-29_001', r'21\2024-01-30_001', r'22\2024-02-01_001', r'23\2024-02-08_001', r'25\2024-02-15_001'] # subject 5 excluded, subject 6 not loadable, 5\2023-04-27_005',,'6\2023-05-02_001', subj 18 trigger problem., subject 24 bad


for sub in range(0,len(filelist)): 
    raw_haemo,short_chs, epochs_TempRoi, epochs_allCond, channelstats, roiglm, epochs_shortchans, glm_shortchans,  largerthanstdsALLcond, bads_SCI_PP = individual_analysis(filelist[sub], plots, stimrenaming, T_Noi_rename, glm, waveform)
        # Save individual-evoked participant data along with others in all_evokeds
    df_channelstats = pd.concat([df_channelstats, channelstats], ignore_index=True)
    df_roiglm = pd.concat([df_roiglm, roiglm], ignore_index=True)
    df_shortchanglm = pd.concat([ df_shortchanglm, glm_shortchans], ignore_index=True)
    
    # Save individual-evoked participant data along with others in all_evokeds
    for cidx, condition in enumerate(epochs_allCond.event_id):
        all_evokeds_allCond[condition].append(epochs_allCond[condition].average())
            
    for cidx, condition in enumerate(epochs_TempRoi.event_id):
        all_evokeds_TempRoi[condition].append(epochs_TempRoi[condition].average())
        
    for cidx, condition in enumerate(epochs_shortchans.event_id):
        all_evokeds_ShortChans[condition].append(epochs_shortchans[condition].average())

    # collect info about what channels were bad, nr of epochs dropped, what was the 5*std value used as threshold? 
    epochrejectCutOffs.append(largerthanstdsALLcond)
    bad_channels.append(bads_SCI_PP)
    #add
  #  nr_of_bad_epochs.append(currentnr_of_bad_epochs)

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

         
# save results

# savepath = r"C:\Users\AICU\OneDrive - Demant\Documents - Eriksholm Research Centre\Research Projects\AudTacLoc\fNIRS results - reanalysis with more subjects\Final files_21_03_24\newMNEbadchanreject\RQ0_Allstimgrouped"
# import pickle 
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


# # as csv
# df_roiglm.to_csv(savepath + r'\glm\df_roiGLM_allcond.csv') 
# df_channelstats.to_csv(savepath + r'\glm\df_channelstats_allcond.csv') 
# df_shortchanglm.to_csv(savepath + r'\glm\df_shortchanglm.csv') 


#%% load
savepath = r"C:\Users\AICU\OneDrive - Demant\Documents - Eriksholm Research Centre\Research Projects\AudTacLoc\fNIRS results - reanalysis with more subjects\Final files_21_03_24\newMNEbadchanreject\RQ0_Allstimgrouped"
import pickle
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

#%% define ROIs for waveforms (make sure to use the same as inside first level function is used for glm)

lIFG = [[8,6],[9,6],[9,7],[10,6],[10,7]]; # 8-6 and 9-6 are the same as in luke
#lSTG = [[13,9],[13,13],[14,9],[14,13],[14,14]] # exactly as in robs paper
lSTG = [[13,9],[13,13],[14,9],[14,13],[14,14],[15,10],[15,9],[15,14],[14,15]] # exactly as in robs paper + extra ones inbetween
#rSTG = [[7,5],[5,5],[3,5],[5,2],[3,2]] # as in luke except one chan that we don't have
rSTG = [[7,5],[5,5],[3,5],[5,2],[3,2],[4,4],[4,5],[3,4]]# 6_4 and 7_4 additonal channels in our montage (posterior)

bilatSTG = lSTG + rSTG
lSC =[[9,9],[11,8],[11,9],[11,11],[11,12],[12,8],[12,12]]
rSC =[[2,3],[4,3],[6,3]]
bilatSC= lSC + rSC
lpostTemp = [[15,12], [15,14],[15,15],[14,14],[14,15],[14,13],[16,13],[16,14],[14,11]] # maybe not 14,11


rois = dict(lSTG=picks_pair_to_idx(raw_haemo, lSTG, on_missing ='ignore'),
        rSTG =picks_pair_to_idx(raw_haemo, rSTG, on_missing ='ignore'),
        bilateralSTG =picks_pair_to_idx(raw_haemo, bilatSTG, on_missing ='ignore'),
        Left_SC=picks_pair_to_idx(raw_haemo, lSC, on_missing ='ignore'),
        Right_SC=picks_pair_to_idx(raw_haemo, rSC, on_missing ='ignore'),
        Bilateral_SC=picks_pair_to_idx(raw_haemo, bilatSC, on_missing ='ignore'),
        leftIFG=picks_pair_to_idx(raw_haemo, lIFG, on_missing ='ignore'),
        leftPostTemp=picks_pair_to_idx(raw_haemo, lpostTemp, on_missing ='ignore'))



#%% Short channels analysed as if they were real channels
    
# short chan rois
allshortchans = [[1,16],[2,17],[6,18],[7,19],[8,20],[9,21],[12,22],[16,23]]
# very narrow ROIs: choosen based on GLM results or highest spec
shortchanstemppost = [[6,18],[7,19],[16,23],[12,22]]
shortchansfront = [[1,16],[2,17],[8,20],[9,21]]
#veryfrontal = [[1,16],[8,20]]
# very narrow ROIs: choosen based on GLM results or highest spec
condition = 'Stimulus' #  does not matter which, but needs one condition key of data frame in order to index into channels


SC_ROI= dict(all_shortchans=picks_pair_to_idx(all_evokeds_ShortChans[condition][0], allshortchans),
             frontal =picks_pair_to_idx(all_evokeds_ShortChans[condition][0], shortchansfront),
             posterior_temporal = picks_pair_to_idx(all_evokeds_ShortChans[condition][0],shortchanstemppost))


#          # waveforms
# ## 1: from epochs on preselected channels only: all events temp channels
# fig, axes = plt.subplots(nrows=1, ncols=len(all_evokeds_ShortChans), figsize=(10, 5))
# lims = dict(hbo=[-5, 5], hbr=[-5, 5])

# for (pick, color) in zip(['hbo', 'hbr'], ['r', 'b']):
#     for idx, evoked in enumerate(all_evokeds_ShortChans):
#         plot_compare_evokeds({evoked: all_evokeds_ShortChans[evoked]}, combine='mean',
#                              picks=pick, axes=axes[idx], show=False,
#                              colors=[color], legend=False, ylim=lims, ci=0.95,
#                              show_sensors=idx == 1)
#         axes[idx].set_title('{}'.format(evoked))
# axes[0].legend(["Oxyhaemoglobin","95% CI", "Stimulus onset","Deoxyhaemoglobin"])

# #all chan plot is not based on all chans..!!




#short chans
# plot above is not selecting all chans (only all that are available in first sub --> forcing all channels by roi with all chans
SC_ROI= dict(all_shortchans=picks_pair_to_idx(all_evokeds_ShortChans[condition][0], allshortchans),
            frontal =picks_pair_to_idx(all_evokeds_ShortChans[condition][0], shortchansfront))
         #    posterior_temporal = picks_pair_to_idx(all_evokeds_ShortChans[condition][0],shortchanstemppost))


fig, axes = plt.subplots(nrows=len(SC_ROI), ncols=len(all_evokeds_ShortChans))
lims = dict(hbo=[-5, 5], hbr=[-5, 5])

for (pick, color) in zip(['hbo', 'hbr'], ['r', 'b']):
    for ridx, roi in enumerate(SC_ROI):
        for cidx, evoked in enumerate(all_evokeds_ShortChans):
            if pick == 'hbr':
                picks = SC_ROI[roi][1::2]  # Select only the hbr channels
            else:
                picks = SC_ROI[roi][0::2]  # Select only the hbo channel
            plot_compare_evokeds({evoked: all_evokeds_ShortChans[evoked]}, combine='mean',
                                 picks=picks, axes=axes[ridx, cidx],
                                 show=False, colors=[color], legend=False,
                                 ylim=lims, ci=0.95, show_sensors=cidx == 1)
            axes[ridx, cidx].set_title("")
            axes[0, cidx].set_title('{}'.format(evoked))
        axes[ridx, 0].set_ylabel(f"{roi}\nChromophore (ΔμMol)")
#axes[0, 0].legend(["Oxyhaemoglobin", "Deoxyhaemoglobin"])
fig.savefig(savepath + r"\waveforms\short_chswaveforms_stim_nostim.svg", format='svg')
# take first one for all chan 


#%% short chan dataframe

df_shortchan_amplitudes = pd.DataFrame(columns=['ID', 'ROI', 'Chroma', 'Condition', 'Value'])

for idx, evoked in enumerate(all_evokeds_ShortChans):
    for subj_data in all_evokeds_ShortChans[evoked]:
        subj_id = subj_data.info['subject_info']['his_id'] if 'subject_info' in subj_data.info and 'his_id' in subj_data.info['subject_info'] else 'unknown'
        bads = subj_data.info['bads']
        for roi, picks in SC_ROI.items():
            for chroma in ["hbo", "hbr"]:
                good_picks = [pick for pick in picks if subj_data.ch_names[pick] not in bads]
                if good_picks:  # Proceed only if there are good channels
                    data = deepcopy(subj_data).pick(picks=good_picks).pick(chroma)
                    if len(data.ch_names) > 0:
                        value = data.crop(tmin=5.0, tmax=8.0).data.mean() * 1.0e6
                        this_df = pd.DataFrame({'ID': subj_id, 'ROI': roi, 'Chroma': chroma, 'Condition': evoked, 'Value': value}, index=[0])
                        df_shortchan_amplitudes = pd.concat([df_shortchan_amplitudes, this_df], ignore_index=True)

df_shortchan_amplitudes.reset_index(inplace=True, drop=True)
df_shortchan_amplitudes['Value'] = pd.to_numeric(df_shortchan_amplitudes['Value'])

#%%

# compare statsitically for all chan (hbo only)
input_data = df_shortchan_amplitudes.query("Condition in ['Stimulus', 'No Stimulus']")
input_data = input_data.query("Chroma in ['hbo']")
input_data = input_data.query("ROI in ['all_shortchans']")
roi_model = smf.mixedlm("Value ~-1+ Condition:ROI", input_data,
                        groups=input_data["ID"]).fit()
roi_model.summary()
df_lme_shortchans= statsmodels_to_results(roi_model)

df_lme_shortchans.reset_index(drop=True, inplace=True)
df_lme_shortchans = df_lme_shortchans.set_index(['ROI', 'Condition'])



df_lme_shortchans.to_csv(savepath + r'\waveforms\df_waveformmeanamps_shortchans.csv')


# FDR correction
pvals = df_lme_shortchans['P>|z|'].values
pvals_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)[1]

# Add the corrected p-values back to the dataframe
df_lme_shortchans['p_FDRcorrected'] = pvals_corrected
df_lme_shortchans = df_lme_shortchans.rename(columns={'Coef.': 'Amplitude', 'P>|z|': 'p-value'})
df_lme_shortchans = df_lme_shortchans.drop(columns=['Std.Err.', 'z', '[0.025', '0.975]', 'Significant'])
df_lme_shortchans['significant?'] = df_lme_shortchans['p_FDRcorrected'] <= 0.05

# Apply this function to the specific columns you want to format
df_lme_shortchans['Amplitude'] = df_lme_shortchans['Amplitude'].apply(lambda x: format_number(x))
df_lme_shortchans['p-value'] = df_lme_shortchans['p-value'].apply(lambda x: format_number(x))
df_lme_shortchans['p_FDRcorrected'] = df_lme_shortchans['p_FDRcorrected'].apply(lambda x: format_number(x))

df_lme_shortchans['Amplitude'] = pd.to_numeric(df_lme_shortchans['Amplitude'], errors='coerce')
df_lme_shortchans.to_csv(savepath + r'\waveforms\df_waveformmeanamps_shortchans.csv')


# see all subjects (--> plot from joerg or this)
grp_results = df_shortchan_amplitudes.query("Condition in ['Stimulus', 'No Stimulus']")
grp_results = grp_results.query("Chroma in ['hbo']")

fig = sns.catplot(x="ID", y="Value", col="Condition",  hue="ROI", data=grp_results, col_wrap=2, errorbar=None, palette="muted", height=9, s=60)




        #%% short chan glm
grp_results = df_shortchanglm.query("Condition in ['Stimulus', 'No Stimulus']")
grp_results = grp_results.query("Chroma in ['hbo']")
grp_results = grp_results.query("ROI in ['all_shortchans']")
shortchan_model = smf.mixedlm("theta ~ -1 + Condition:ROI", 
                        grp_results, groups=grp_results["ID"]).fit(method='nm')
shortchan_model.summary()
df_lme_glm_shortchans= statsmodels_to_results(shortchan_model)

df_lme_glm_shortchans.reset_index(drop=True, inplace=True)
df_lme_glm_shortchans = df_lme_glm_shortchans.set_index(['ROI', 'Condition'])


# FDR correction
pvals = df_lme_glm_shortchans['P>|z|'].values
pvals_corrected = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)[1]

# Add the corrected p-values back to the dataframe
df_lme_glm_shortchans['p_FDRcorrected'] = pvals_corrected
df_lme_glm_shortchans = df_lme_glm_shortchans.rename(columns={'Coef.': 'Beta', 'P>|z|': 'p-value'})
df_lme_glm_shortchans = df_lme_glm_shortchans.drop(columns=['Std.Err.', 'z', '[0.025', '0.975]', 'Significant'])
df_lme_glm_shortchans['significant?'] = df_lme_glm_shortchans['p_FDRcorrected'] <= 0.05

# Apply this function to the specific columns you want to format
df_lme_glm_shortchans['Beta'] = df_lme_glm_shortchans['Beta'].apply(lambda x: format_number(x))
df_lme_glm_shortchans['p-value'] = df_lme_glm_shortchans['p-value'].apply(lambda x: format_number(x))
df_lme_glm_shortchans['p_FDRcorrected'] = df_lme_glm_shortchans['p_FDRcorrected'].apply(lambda x: format_number(x))

df_lme_glm_shortchans['Beta'] = pd.to_numeric(df_lme_glm_shortchans['Beta'], errors='coerce')


df_lme_glm_shortchans.to_csv(savepath + r'\glm\df_glm_lme_shortchans.csv')

# for brain plots of short channels export short_ch var and channel stats glm from function to beable to get channel infos
# theta different order than in info!
#create short channel stats df..
# info = short_chs.copy().pick('hbo').info
# channelorder = info.ch_names
# df_lme_glmchanshort_ordered = df_lme_glmchanshort.set_index('ch_name').reindex(channelorder).reset_index()

# fig= mne_nirs.visualisation.plot_nirs_source_detector(
#     df_lme_glmchanshort_ordered['Coef.'], # loop through theta-hbo group result thetas
#     raw_haemo.copy().pick('hbo').info , fnirs = True, 
# subject= 'fsaverage', coord_frame= 'head', trans = 'fsaverage', surfaces=['brain'])



#%% Long channels

         # waveforms
## 1: from epochs on preselected channels only: all events temp channels
fig, axes = plt.subplots(nrows=1, ncols=len(all_evokeds_allCond), figsize=(10, 5))
lims = dict(hbo=[-5, 5], hbr=[-5, 5])

for (pick, color) in zip(['hbo', 'hbr'], ['r', 'b']):
    for idx, evoked in enumerate(all_evokeds_allCond):
        plot_compare_evokeds({evoked: all_evokeds_allCond[evoked]}, combine='mean',
                             picks=pick, axes=axes[idx], show=False,
                             colors=[color], legend=False, ylim=lims, ci=0.95,
                             show_sensors=idx == 1)
        axes[idx].set_title('{}'.format(evoked))
axes[0].legend(["Oxyhaemoglobin","95% CI", "Stimulus onset","Deoxyhaemoglobin"])

fig.savefig(savepath + r"\waveforms\longchas_aveforms_stim_nostim.svg", format='svg')


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




# compare statsitically for all chan (hbo only)
input_data = df_waveformamplitudes.query("Condition in ['Stimulus', 'No Stimulus']")
input_data = input_data.query("Chroma in ['hbo']")
# no roi means all rois together?
roi_model = smf.mixedlm("Value ~-1+ Condition", input_data,
                        groups=input_data["ID"]).fit()
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


df_lme_waveslongchans.to_csv(savepath + r'\waveforms\df_lme_waves_longchans.csv')




# see all subjects (--> plot from joerg with waveforms or this)
grp_results = df_waveformamplitudes.query("Condition in ['Stimulus', 'No Stimulus']")
grp_results = grp_results.query("Chroma in ['hbo']")

fig = sns.catplot(x="ID", y="Value", col="Condition",  hue="ROI", data=grp_results, col_wrap=2, errorbar=None, palette="muted", height=9, s=60)


        # glm lme
grp_results = df_roiglm.query("Condition in ['Stimulus', 'No Stimulus']")
grp_results = grp_results.query("Chroma in ['hbo']")
roi_glmmodel = smf.mixedlm("theta ~ -1 + Condition", 
                        grp_results, groups=grp_results["ID"]).fit(method='nm')
roi_glmmodel.summary()
df_lme_glmlong = statsmodels_to_results(roi_glmmodel)



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

df_lme_glmlong['Beta'] = pd.to_numeric(df_lme_glmlong['Beta'], errors='coerce')


df_lme_glmlong.to_csv(savepath + r'\glm\df_glm_lme_longchans.csv')





# samw scatter plot as befor but now for thetas instead of mean or peak vals
grp_results = df_roiglm.query("Condition in ['Stimulus', 'No Stimulus']")
grp_results = grp_results.query("Chroma in ['hbo']")

fig = sns.catplot(x="ID", y="theta", col="Condition",  hue="ROI", data=grp_results, col_wrap=2, errorbar=None, palette="muted", height=9, s=60)



#%% brain plots


condition = "Condition in ['Stimulus']" #"Condition in ['Auditory Speech']","Condition in ['Noise']" 
chroma = "Chroma in ['hbo']"
chroma2 = 'hbo'

grp_results = df_channelstats.query(condition)
grp_results = grp_results.query(chroma)

grp_results = grp_results.dropna(subset=['theta'])

chan_glmmodel = smf.mixedlm("theta ~ -1 + Condition:ch_name", 
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



# theta different order than in info!
info = raw_haemo.copy().pick(chroma2).info
channelorder = info.ch_names
df_lme_glmchanlong_ordered = df_lme_glmchanlong.set_index('ch_name').reindex(channelorder).reset_index()


df_lme_glmchanlong_ordered['Beta'] = pd.to_numeric(df_lme_glmchanlong_ordered['Beta'], errors='coerce')


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
plotter.screenshot(savepath + r'\glm\brainchanplots_AllStim_GLM_rh.png')

# Left hemisphere view (looking from the left to the right)
left_camera_position = [-0.5, 0, 0.07]  # Mirror the x-coordinate

# Set and update for the left hemisphere view
plotter.camera_position = [left_camera_position, focal_point, view_up]
plotter.render()
plotter.screenshot(savepath + r'\glm\brainchanplots_AllStim_GLM_lh.png')



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

ten_largest_betas = df_lme_glmchanlong_ordered.nlargest(10, 'Beta')



# brain plot
# extract significant ones and show significant ones only
df_lme_glmchanlong_ordered.loc[~df_lme_glmchanlong_ordered['significant?'], 'Beta'] = 0
non_zero_count = df_lme_glmchanlong_ordered[df_lme_glmchanlong_ordered['Beta'] != 0].shape[0]
positive_count = df_lme_glmchanlong_ordered[df_lme_glmchanlong_ordered['Beta'] > 0].shape[0]
negative_count = df_lme_glmchanlong_ordered[df_lme_glmchanlong_ordered['Beta'] < 0].shape[0]

mean_betas = df_lme_glmchanlong_ordered['Beta'].mean()
std_betas = df_lme_glmchanlong_ordered['Beta'].std()




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
condition = "Condition in ['Stimulus']" #"Condition in ['Auditory Speech']","Condition in ['Noise']" 
chroma = "Chroma in ['hbo']"
chroma2 = 'hbo'

grp_results = df_waveformamplitudes_chans.query(condition)
grp_results = grp_results.query(chroma)

grp_results = grp_results.dropna(subset=['Value'])

chan_wavemodel = smf.mixedlm("Value ~ -1 + Condition:ch_name", 
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
#plotter.screenshot(savepath + r'\waveforms\brainplots_AudSpeech_Sil_rh.png')
plotter.screenshot(savepath + r'\waveforms\brainplots_AllStim_rh.png')

# Left hemisphere view (looking from the left to the right)
left_camera_position = [-0.5, 0, 0.07]  # Mirror the x-coordinate

# Set and update for the left hemisphere view
plotter.camera_position = [left_camera_position, focal_point, view_up]
plotter.render()
#plotter.screenshot(savepath + r'\waveforms\brainplots_AudSpeech_Sil_lh.png')
plotter.screenshot(savepath + r'\waveforms\brainplots_AllStim_lh.png')




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


#df_lme_chan_wavemodel_ordered.to_csv(r'C:\Users\AICU\OneDrive - Demant\Documents - Eriksholm Research Centre\Research Projects\AudTacLoc\fNIRS results - reanalysis with more subjects\Final files_MNE analysis\RQ1_audStimuli\glm\df_RQ1.3_waveformampchanstats_A_Speech_whichchans.csv')


# extract significant ones and show significant ones only
df_lme_chan_wavemodel_ordered.loc[~df_lme_chan_wavemodel_ordered['significant?'], 'Amplitude'] = 0
non_zero_count = df_lme_chan_wavemodel_ordered[df_lme_chan_wavemodel_ordered[ 'Amplitude'] != 0].shape[0]
positive_count = df_lme_chan_wavemodel_ordered[df_lme_chan_wavemodel_ordered[ 'Amplitude'] > 0].shape[0]
negative_count = df_lme_chan_wavemodel_ordered[df_lme_chan_wavemodel_ordered[ 'Amplitude'] < 0].shape[0]

mean_amps = df_lme_chan_wavemodel_ordered[ 'Amplitude'].mean()
std_amps = df_lme_chan_wavemodel_ordered[ 'Amplitude'].std()




