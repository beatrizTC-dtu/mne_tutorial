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
sys.path.append(r'C:\GitHub\AudTacLocalizer\Analysis scripts\final pipelines')
from preprocessing_and_first_level_MNEbadchanreject import individual_analysis # first level function for both glm and waveforms

sys.path.append(r'C:\GitHub\audiohaptics_exp1\selfwrittenFunctions\functions complementing mne')
from plot_nirs_source_detector_wAxixLims import plot_nirs_source_detector_wAxixLims # first level function for both glm and waveforms


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


"""

Created on Mon Mar 11 15:10:31 2024
@author: aicu

Group analysis script for audio-tactile localizer data:
- Same preprocessing for glm and waveforms with short channels being scaled by separate glm and then subtracted 
- bad channels are cleaned as defined in matlab

"""


#%% Preprocessing and first level analysis

# choose stimulus grouping 

  # treating all speech st
 
stimrenaming = ({'1.0': 'Delete', '2.0':'Delete', '3.0':'Delete', '4.0':'A_SPIN', '5.0': 'T','6.0': 'AT_SPIN', '7.0': 'Delete', '8.0': 'Delete'})
T_Noi_rename = {'9.0': 'Delete'}

# stimrenaming = {'1.0': 'Speech', '2.0':'Noise', '3.0':'Control', '4.0':'Speech', '5.0': 'Tactile','6.0': 'Speech', '7.0': 'Speech', '8.0': 'Control'}
# T_Noi_rename = {'9.0': 'Tactile'}

plots = False # if true, make sure to adjust path inside function to store plots in correct folder for this analysis
glm = True
waveform = True
normalization = False # only for NAHA breathing task is available


epochrejectCutOffs =[]
df_channelstats = pd.DataFrame() 
df_roiglm = pd.DataFrame() 
df_shortchanglm = pd.DataFrame() 
all_evokeds_allCond = defaultdict(list)
all_evokeds_TempRoi = defaultdict(list) # all cond but only temp channels epoched
all_evokeds_ShortChans = defaultdict(list)

nr_of_bad_epochs = []
bad_channels =[]
percent_bad_epochs = []



filelist = [r'01\2023-04-18_002',r'02\2023-04-19_001',r'03\2023-04-21_001',r'04\2023-04-24_001',r'07\2023-05-04_001' ,r'08\2023-12-18_002',r'09\2023-12-21_001',r'10\2024-01-04_001', r'11\2024-01-09_001',r'12\2024-01-16_003',r'13\2024-01-19_001',r'14\2024-01-19_002',r'15\2024-01-23_001',r'16\2024-01-23_002',r'17\2024-01-24_003',  r'19\2024-01-26_002',  r'20\2024-01-29_001', r'21\2024-01-30_001', r'22\2024-02-01_001', r'23\2024-02-08_001', r'25\2024-02-15_001'] # subject 5 excluded, subject 6 not loadable, 5\2023-04-27_005',,'6\2023-05-02_001', subj 18 trigger problem., subject 24 bad
#filelist =[r'10\2024-01-04_001']

for sub in range(0,len(filelist)): 
    raw_haemo,short_chs, epochs_TempRoi, epochs_allCond, channelstats, roiglm, epochs_shortchans, glm_shortchans, epochrejectCutOff, bads_SCI_PP, bad_epochs  = individual_analysis(filelist[sub], plots, stimrenaming, T_Noi_rename, glm, waveform)
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

    epochrejectCutOffs.append(epochrejectCutOff)
    bad_channels.append(bads_SCI_PP)
    percent_bad_epochs.append(bads_epochs)
    
# save results
savepath = r"C:\Users\AICU\OneDrive - Demant\Documents - Eriksholm Research Centre\Research Projects\AudTacLoc\fNIRS results - reanalysis with more subjects\Final files_21_03_24\newMNEbadchanreject\RQ3AT"


  #    waveforms
with open(savepath + r'\waveforms\all_evokeds.pkl', 'wb') as file:
    pickle.dump(all_evokeds_allCond, file)

raw_haemo.save(savepath + r'\raw_haemo.fif', overwrite= True)

    
    # glm data frames
#df_shortchanglm.to_pickle(savepath + r'\short channel evoked responses\df_shortchanglm.pkl')  
df_roiglm.to_pickle(savepath + r'\glm\df_roiglm.pkl')
df_channelstats.to_pickle(savepath + r'\glm\df_channelstats.pkl')

# all conditions

# #export
df_roiglm.to_csv(savepath + r'\glm\df_roiGLM_allcond.csv') 
df_channelstats.to_csv(savepath + r'\glm\df_channelstats_allcond.csv') 
     

#%% 
savepath = r"C:\Users\AICU\OneDrive - Demant\Documents - Eriksholm Research Centre\Research Projects\AudTacLoc\fNIRS results - reanalysis with more subjects\Final files_21_03_24\newMNEbadchanreject\RQ3AT"


# load results

with open(savepath + r'\waveforms\all_evokeds.pkl', 'rb') as file:
    all_evokeds_allCond =pickle.load( file)

with open(savepath + r'\glm\df_roiglm.pkl', 'rb') as file:
    df_roiglm =pickle.load( file)
with open(savepath + r'\glm\df_channelstats.pkl', 'rb') as file:
    df_channelstats =pickle.load( file)
    
raw_haemo = mne.io.read_raw_fif(savepath + r'\raw_haemo.fif')


#define ROIs for waveforms (make sure to use the same as inside first level function is used for glm)

lIFG = [[8,6],[9,6],[9,7],[10,6],[10,7]]; # 8-6 and 9-6 are the same as in luke
#lSTG = [[13,9],[13,13],[14,9],[14,13],[14,14]] # exactly as in robs paper
lSTG = [[13,9],[13,13],[14,9],[14,10],[14,13],[14,14],[15,10],[15,9],[15,14],[14,15]] # exactly as in robs paper + extra ones inbetween
#rSTG = [[7,5],[5,5],[3,5],[5,2],[3,2]] # as in luke except one chan that we don't have
rSTG = [[7,5],[5,5],[3,5],[5,2],[3,2],[4,4],[4,5],[3,4]]# 6_4 and 7_4 additonal channels in our montage (posterior)

bilatSTG = lSTG + rSTG
lSC =[[9,9],[11,8],[11,9],[11,11],[11,12],[12,8],[12,12],[12,11]]
rSC =[[2,3],[4,3],[6,3]]
bilatSC= lSC + rSC
lpostTemp = [[15,12], [15,14],[15,15],[14,14],[14,15],[14,13],[16,13],[16,14],[14,11]] # maybe not 14,11


# rois = dict(lSTG=picks_pair_to_idx(raw_haemo, lSTG, on_missing ='ignore'),
#         rSTG =picks_pair_to_idx(raw_haemo, rSTG, on_missing ='ignore'),
#         bilateralSTG =picks_pair_to_idx(raw_haemo, bilatSTG, on_missing ='ignore'),
#         Left_SC=picks_pair_to_idx(raw_haemo, lSC, on_missing ='ignore'),
#         Right_SC=picks_pair_to_idx(raw_haemo, rSC, on_missing ='ignore'),
#         Bilateral_SC=picks_pair_to_idx(raw_haemo, bilatSC, on_missing ='ignore'),
#         leftIFG=picks_pair_to_idx(raw_haemo, lIFG, on_missing ='ignore'),
#         leftPostTemp=picks_pair_to_idx(raw_haemo, lpostTemp, on_missing ='ignore'))


rois = dict(lSTG=picks_pair_to_idx(raw_haemo, lSTG, on_missing ='ignore'),
        rSTG =picks_pair_to_idx(raw_haemo, rSTG, on_missing ='ignore'),
        Left_SC=picks_pair_to_idx(raw_haemo, lSC, on_missing ='ignore'),
        Right_SC=picks_pair_to_idx(raw_haemo, rSC, on_missing ='ignore'),
        leftIFG=picks_pair_to_idx(raw_haemo, lIFG, on_missing ='ignore'),
        leftPostTemp=picks_pair_to_idx(raw_haemo, lpostTemp, on_missing ='ignore'))
         
       

for evoked in all_evokeds_allCond.values():
    for evo in evoked:
        for bad_chan in evo.info['bads']:
            if bad_chan in evo.ch_names:
                idx = evo.ch_names.index(bad_chan)
                evo.data[idx, :] = np.nan # Set all time points for this channel to NaN  



#%% Short channels analysed as if they were real channels # not for this grouping
# short chan rois
allshortchans = [[1,16],[2,17],[6,18],[7,19],[8,20],[9,21],[12,22],[16,23]]
# very narrow ROIs: choosen based on GLM results or highest spec
shortchanstemppost = [[6,18],[7,19],[16,23],[12,22]]
shortchansfront = [[1,16],[2,17],[8,20],[9,21]]
#veryfrontal = [[1,16],[8,20]]
# very narrow ROIs: choosen based on GLM results or highest spec
condition = 'Speech_Sil' #  does not matter which, but needs one condition key of data frame in order to index into channels


SC_ROI= dict(all_shortchans=picks_pair_to_idx(all_evokeds_ShortChans[condition][0], allshortchans),
             frontal =picks_pair_to_idx(all_evokeds_ShortChans[condition][0], shortchansfront))
 #  posterior_temporal = picks_pair_to_idx(all_evokeds_ShortChans[condition][0],shortchanstemppost)
     # very noisy


# waveform all channels plot picks only channels availabe in first subject of the list -> do roi plots to ask for selection of channels (all channels) explicitly

#short chans
fig, axes = plt.subplots(nrows=len(SC_ROI), ncols=len(all_evokeds_ShortChans),
                         figsize=(25, 10))
lims = dict(hbo=[-10, 10], hbr=[-10, 10])

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
                                 ylim=lims, ci=0.95, show_sensors=cidx == 2)
            axes[ridx, cidx].set_title("")
            axes[0, cidx].set_title('{}'.format(evoked))
        axes[ridx, 0].set_ylabel(f"{roi}\nChromophore (ΔμMol)")
#axes[0, 0].legend(["Oxyhaemoglobin", "Deoxyhaemoglobin"])




# stats in grouped script : stim vs no stim

#%% Long channels

         # waveforms
## 1: all channels merged
selected_conditions = ['AT_SPIN', 'A_SPIN','T']

fig, axes = plt.subplots(nrows=1, ncols=len(selected_conditions), figsize=(20, 5))
lims = dict(hbo=[-5, 5], hbr=[-5, 5])

for (pick, color) in zip(['hbo', 'hbr'], ['r', 'b']):
    for idx, evoked in enumerate(selected_conditions ):
        plot_compare_evokeds({evoked: all_evokeds_allCond[evoked]}, combine='mean',
                             picks=pick, axes=axes[idx], show=False,
                             colors=[color], legend=False, ylim=lims, ci=0.95,
                             show_sensors=idx == 2)
        axes[idx].set_title('{}'.format(evoked))
axes[0].legend(["Oxyhaemoglobin","95% CI", "Stimulus onset","Deoxyhaemoglobin"])


fig.savefig(savepath + r"\waveforms\waves_allchan.svg", format='svg')


# in ROIs # mean is now nanmean in evoked functin
selected_conditions = ['AT_SPIN', 'A_SPIN','T']
fig, axes = plt.subplots(nrows=len(rois), ncols=len(selected_conditions), figsize=(10, 18))
lims = dict(hbo=[-5, 5], hbr=[-5, 5])

for (pick, color) in zip(['hbo', 'hbr'], ['r', 'b']):
    for ridx, roi in enumerate(rois):
        for cidx, evoked in enumerate(selected_conditions):  # Iterate over selected conditions
            if evoked in all_evokeds_allCond:  # Check if the condition exists in your data
                if pick == 'hbr':
                    picks = rois[roi][1::2]  # Select only the hbr channels
                else:
                    picks = rois[roi][0::2]  # Select only the hbo channel
                plot_compare_evokeds({evoked: all_evokeds_allCond[evoked]}, combine='mean',
                                     picks=picks, axes=axes[ridx, cidx],
                                     show=False, colors=[color], legend=False,
                                     ylim=lims, ci=0.95, show_sensors=cidx == 2)
                axes[ridx, cidx].set_title("")
                axes[0, cidx].set_title('{}'.format(evoked))
            axes[ridx, 0].set_ylabel(f"{roi}\nChromophore (ΔμMol)")
#axes[0, 0].legend(["Oxyhaemoglobin", "Deoxyhaemoglobin"])

fig.savefig(savepath + r"\waveforms\waves_ROIS.svg", format='svg')





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
                    if len(data.ch_names) > 0:
                        value = data.crop(tmin=5.0, tmax=8.0).data.mean() * 1.0e6
                        this_df = pd.DataFrame({'ID': subj_id, 'ROI': roi, 'Chroma': chroma, 'Condition': evoked, 'Value': value}, index=[0])
                        df_waveformamplitudes = pd.concat([df_waveformamplitudes, this_df], ignore_index=True)

df_waveformamplitudes.reset_index(inplace=True, drop=True)
df_waveformamplitudes['Value'] = pd.to_numeric(df_waveformamplitudes['Value'])




# compare statsitically for all chan (hbo only)
input_data = df_waveformamplitudes.query("Condition in ['AT_SPIN', 'A_SPIN','T']")  #'T'
input_data = input_data.query("Chroma in ['hbo']")
input_data = input_data.query("ROI in ['lSTG',  'rSTG' , 'Left_SC',   'Right_SC',  'leftIFG', 'leftPostTemp']")
roi_model = smf.mixedlm("Value ~-1+ Condition:ROI", input_data,
                        groups=input_data["ID"]).fit()
roi_model.summary()
df_lme_waveslongchans= statsmodels_to_results(roi_model)
df_lme_waveslongchans.reset_index(drop=True, inplace=True)
df_lme_waveslongchans = df_lme_waveslongchans.set_index(['ROI', 'Condition'])


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

df_lme_waveslongchans.to_csv(savepath + r'\waveforms\df_waveforms_RQ3_AT.csv')




# see all subjects (--> plot from joerg or this)
grp_results = df_waveformamplitudes.query("Condition in ['AT_SPIN', 'A_SPIN','T']")
grp_results = grp_results.query("Chroma in ['hbo']")

fig = sns.catplot(x="ID", y="Value", col="Condition",  hue="ROI", data=grp_results, col_wrap=2, errorbar=None, palette="muted", height=9, s=60)


#%% glm exact the same but for beta estimates
input_data = df_roiglm.query("Condition in ['AT_SPIN', 'A_SPIN']") # 'T'
input_data = input_data.query("Chroma in ['hbo']")
input_data = input_data.query("ROI in ['lSTG',  'rSTG' , 'Left_SC',   'Right_SC',  'leftIFG', 'leftPostTemp']")
roi_model = smf.mixedlm("theta ~-1+ Condition:ROI", input_data,
                        groups=input_data["ID"]).fit()
roi_model.summary()
df_lme_GLMROIlongchans= statsmodels_to_results(roi_model)
df_lme_GLMROIlongchans.reset_index(drop=True, inplace=True)
df_lme_GLMROIlongchans = df_lme_GLMROIlongchans.set_index(['ROI', 'Condition'])


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
df_lme_GLMROIlongchans.to_csv(savepath + r'\glm\df_glm_RQ3_AT.csv')




#%% RQ: AT larger than A?

# waveform
# compare statsitically for all chan (hbo only)
input_data = df_waveformamplitudes.query("Condition in ['AT_SPIN', 'A_SPIN']")
input_data = input_data.query("Chroma in ['hbo']")
input_data = input_data.query("ROI in ['leftPostTemp']") # 'lSTG'] 'rSTG'
roi_model = smf.mixedlm("Value ~ Condition", input_data,
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


df_lme_waveslongchans['Amplitude'] = df_lme_waveslongchans['Amplitude'].apply(lambda x: format_number(x))
df_lme_waveslongchans['p-value'] = df_lme_waveslongchans['p-value'].apply(lambda x: format_number(x))
df_lme_waveslongchans['p_FDRcorrected'] = df_lme_waveslongchans['p_FDRcorrected'].apply(lambda x: format_number(x))

df_lme_waveslongchans['Amplitude'] = pd.to_numeric(df_lme_waveslongchans['Amplitude'], errors='coerce')
# df_lme_waveslongchans['p-value'] = pd.to_numeric(df_lme_waveslongchans['p-value'], errors='coerce')
# df_lme_waveslongchans['p_FDRcorrected'] = pd.to_numeric(df_lme_waveslongchans['p_FDRcorrected'], errors='coerce')


#df_lme_waveslongchans.to_csv(r'C:\Users\AICU\OneDrive - Demant\Documents - Eriksholm Research Centre\Research Projects\AudTacLoc\fNIRS results - reanalysis with more subjects\Final files_MNE analysis\RQ1_audStimuli\waveforms\df_RQ1.2SpeechlargerNoise_lSTG.csv')






#glm
input_data = df_roiglm.query("Condition in  ['AT_SPIN', 'A_SPIN']")
input_data = input_data.query("Chroma in ['hbo']")
input_data = input_data.query("ROI in ['rSTG']")  # "ROI in ['lSTG']"'leftPostTemp'
roi_model = smf.mixedlm("theta ~ Condition", input_data,
                        groups=input_data["ID"]).fit()
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
# df_lme_GLMROIlongchans


#%%  brain plots glm  and channel stats
condition = "Condition in ['A_SPIN']" #"Condition in ['T_Sil']","Condition in ['Control_Sil']" 
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
##

# adjust radius for insignificant channels
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
plotter.screenshot(savepath + r'\glm\brainchanplots_A_SPIN__GLM_rh.png')

# Left hemisphere view (looking from the left to the right)
left_camera_position = [-0.5, 0, 0.07]  # Mirror the x-coordinate

# Set and update for the left hemisphere view
plotter.camera_position = [left_camera_position, focal_point, view_up]
plotter.render()
plotter.screenshot(savepath + r'\glm\brainchanplots_A_SPIN_GLM_lh.png')



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

three_largest_betas = df_lme_glmchanlong_ordered.nlargest(10, 'Beta')



df_lme_glmchanlong_ordered.to_csv(savepath + r'\glm\df_RQ3_chanstats_AT_SPIN_10largestbeta.csv')


# brain plot
# extract significant ones and show significant ones only
df_lme_glmchanlong_ordered.loc[~df_lme_glmchanlong_ordered['significant?'], 'Beta'] = 0
non_zero_count = df_lme_glmchanlong_ordered[df_lme_glmchanlong_ordered['Beta'] != 0].shape[0]
positive_count = df_lme_glmchanlong_ordered[df_lme_glmchanlong_ordered['Beta'] > 0].shape[0]
negative_count = df_lme_glmchanlong_ordered[df_lme_glmchanlong_ordered['Beta'] < 0].shape[0]

mean_betas = df_lme_glmchanlong_ordered['Beta'].mean()
std_betas = df_lme_glmchanlong_ordered['Beta'].std()

#%% Define new ROI for channels responding highest to  a) A SPIN, b ) AT SPIN

AROI_glm  = [tuple(map(int, x.replace(' hbo', '').replace('S', '').replace('_D', ' ').split())) for x in three_largest_betas['ch_name']]
ATROI_glm = [tuple(map(int, x.replace(' hbo', '').replace('S', '').replace('_D', ' ').split())) for x in three_largest_betas['ch_name']]



# # save
# with open(savepath + r'\glm\AROI_glm.pkl', 'wb') as file:
#     pickle.dump(AROI_glm, file)
    
# with open(savepath + r'\glm\ATROI_glm.pkl', 'wb') as file:
#     pickle.dump(ATROI_glm, file)

#%% in which channels is AT larger than A?
condition = "Condition in ['AT_SPIN','A_SPIN']" #'AT_SPIN','A_SPIN','T'
chroma = "Chroma in ['hbo']"
chroma2 = 'hbo'

grp_results = df_channelstats.query(condition)
grp_results = grp_results.query(chroma)

grp_results = grp_results.dropna(subset=['theta'])

chan_glmmodel = smf.mixedlm("theta ~Condition:ch_name", 
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
##

# adjust radius for insignificant channels
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
plotter.screenshot(savepath + r'\glm\brainchanplots_A_SPIN__GLM_rh.png')

# Left hemisphere view (looking from the left to the right)
left_camera_position = [-0.5, 0, 0.07]  # Mirror the x-coordinate

# Set and update for the left hemisphere view
plotter.camera_position = [left_camera_position, focal_point, view_up]
plotter.render()
plotter.screenshot(savepath + r'\glm\brainchanplots_A_SPIN_GLM_lh.png')



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

three_largest_betas = df_lme_glmchanlong_ordered.nlargest(10, 'Beta')



#df_lme_glmchanlong_ordered.to_csv(savepath + r'\glm\df_RQ3_chanstats_AT_SPIN_10largestbeta.csv')


# brain plot
# extract significant ones and show significant ones only
df_lme_glmchanlong_ordered.loc[~df_lme_glmchanlong_ordered['significant?'], 'Beta'] = 0
non_zero_count = df_lme_glmchanlong_ordered[df_lme_glmchanlong_ordered['Beta'] != 0].shape[0]
positive_count = df_lme_glmchanlong_ordered[df_lme_glmchanlong_ordered['Beta'] > 0].shape[0]
negative_count = df_lme_glmchanlong_ordered[df_lme_glmchanlong_ordered['Beta'] < 0].shape[0]

mean_betas = df_lme_glmchanlong_ordered['Beta'].mean()
std_betas = df_lme_glmchanlong_ordered['Beta'].std()





#%% Waveform channel amplitude plots, find> highest responsive chans for waveforms too

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
condition = "Condition in ['A_SPIN']" 
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
plotter.screenshot(savepath + r'\waveforms\brainplots_A_SPIN_right_hemisphere_view.png')

# Left hemisphere view (looking from the left to the right)
left_camera_position = [-0.5, 0, 0.07]  # Mirror the x-coordinate

# Set and update for the left hemisphere view
plotter.camera_position = [left_camera_position, focal_point, view_up]
plotter.render()
plotter.screenshot(savepath + r'\waveforms\brainplots_A_SPIN_left_hemisphere_view.png')





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



#%% in which channels AT larger than A - waveforms

# can we plot contrast? --> not working yet
condition = "Condition in ['A_SPIN', 'AT_SPIN']" 
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



# channel order different than in info! -> align so plot is correct
info = raw_haemo.copy().pick(chroma2).info
channelorder = info.ch_names

# Remove the chroma part from channelorder
channelorder = [ch.split(' ')[0] for ch in channelorder]

df_lme_chan_wavemodel_ordered = df_lme_chan_wavemodel.set_index('ch_name').reindex(channelorder).reset_index()


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





#%% second roi analysis
#%% Define new ROI for channels responding highest to  a) A SPIN, b ) AT SPIN


# Extracting channel numbers and converting to a list of tuples
AROI_waves = [tuple(map(int, x.replace('S', '').replace('_D', ' ').split())) for x in three_largest_amps['ch_name']]

ATROI_waves=  [tuple(map(int, x.replace('S', '').replace('_D', ' ').split())) for x in three_largest_amps['ch_name']]


# save
# with open(savepath + r'\waveforms\AROI_waves.pkl', 'wb') as file:
#     pickle.dump(AROI_waves, file)
    
# with open(savepath + r'\waveforms\ATROI_waves.pkl', 'wb') as file:
#     pickle.dump(ATROI_waves, file)

#load


#%% ROI analysis based on A/ AT activaton for glm and wavforms:
    

rois = dict(AROI_waves=picks_pair_to_idx(raw_haemo, AROI_waves, on_missing ='ignore'),
            ATROI_waves=picks_pair_to_idx(raw_haemo, ATROI_waves, on_missing ='ignore'))
          
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


# compare statsitically for all chan (hbo only)
input_data = df_waveformamplitudes.query("Condition in ['AT_SPIN', 'A_SPIN','T']")  #'T'
input_data = input_data.query("Chroma in ['hbo']")
input_data = input_data.query("ROI in ['ATROI_waves']") # ATROI_waves
roi_model = smf.mixedlm("Value ~-1+ Condition:ROI", input_data, #roi_model = smf.mixedlm("Value ~-1+ Condition:ROI", input_data,
                        groups=input_data["ID"]).fit()
roi_model.summary()
df_lme_waveslongchans= statsmodels_to_results(roi_model)
df_lme_waveslongchans.reset_index(drop=True, inplace=True)
df_lme_waveslongchans = df_lme_waveslongchans.set_index(['ROI', 'Condition'])


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

df_lme_waveslongchans.to_csv(savepath + r'\waveforms\df_waveforms_RQ3_AT_activationROI.csv')


#%% MSI #


input_data = df_waveformamplitudes.query("Condition in ['AT_SPIN', 'A_SPIN','T']")  #'T'
input_data = input_data.query("Chroma in ['hbo']")
input_data = input_data.query("ROI in ['AROI_waves']") #  !!!!ATROI_waves
roi_model = smf.mixedlm("Value ~ Condition", input_data, groups=input_data["ID"]).fit()

roi_model.summary()
df_lme_waveslongchans= statsmodels_to_results(roi_model)




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



# Max criterion: keep only max of A or T (most likely the same as considering AT and A in the first place)
# removing T condition changes 


# Filter A_SPIN and T conditions


a_spin_data = df_waveformamplitudes.query("Condition == 'A_SPIN' and Chroma == 'hbo' and ROI == 'AROI_waves'")  #!!
t_data = df_waveformamplitudes.query("Condition == 'T' and Chroma == 'hbo' and ROI == 'AROI_waves'")#         !!!!


# a_spin_data = df_waveformamplitudes.query("Condition == 'A_SPIN' and Chroma == 'hbo' and ROI == 'ATROI_waves'")
# t_data = df_waveformamplitudes.query("Condition == 'T' and Chroma == 'hbo' and ROI == 'ATROI_waves'")

# Merge the two dataframes on ID to compare their values
comparison_df = a_spin_data.merge(t_data, on='ID', suffixes=('_A', '_T'))

# Check if A_SPIN value is always greater or equal to T value
comparison_df['A_greater_T'] = comparison_df['Value_A'] >= comparison_df['Value_T']
if comparison_df['A_greater_T'].all():
    print("A_SPIN is always greater or equal to T.")
else:
    print("There are cases where T is greater than A_SPIN.")



# Create a copy of the original dataframe
df_waveformamplitudes_MSI = df_waveformamplitudes.copy()

# Map the conditions to a common group for aggregation while keeping the original condition
df_waveformamplitudes_MSI['MSI_Condition'] = df_waveformamplitudes_MSI['Condition'].replace({'A_SPIN': 'Unisensory', 'T': 'Unisensory', 'AT_SPIN': 'Multisensory'}) 
# For Unisensory data, find the row with the maximum value for each group and keep the original condition
unisensory_max = df_waveformamplitudes_MSI[df_waveformamplitudes_MSI['MSI_Condition'] == 'Unisensory'].groupby(['ID', 'ROI', 'Chroma'], as_index=False).apply(lambda x: x.loc[x['Value'].idxmax()])

# Filter to keep only Multisensory condition data
multisensory_data = df_waveformamplitudes_MSI[df_waveformamplitudes_MSI['Condition'] == 'AT_SPIN']

# Combine the maximum Unisensory data with the Multisensory data
df_waveformamplitudes_MSI_final = pd.concat([unisensory_max, multisensory_data], ignore_index=True)#.drop(columns=[])





input_data = df_waveformamplitudes_MSI_final.query("MSI_Condition in ['Multisensory', 'Unisensory']")  
input_data = input_data.query("Chroma in ['hbo']")
input_data = input_data.query("ROI in ['AROI_waves']") # ATROI_waves
roi_model = smf.mixedlm("Value ~ MSI_Condition", input_data, groups=input_data["ID"]).fit()

roi_model.summary()
df_lme_waveslongchans= statsmodels_to_results(roi_model)



#%% 2. GLM    
# unlike for waveform data, in which the ROIs are always defined on group level, in the normal glm analysis, we generate the roi_glm dataframe during first level analysis
# here we have new ROIs based on the group results so we need to create a new roi_glm df (as we did for waveform data), based on the channelstats:

    # pass these to first level function to create another roi_glm_df there

#new rois
rois = dict(AROI_glm=picks_pair_to_idx(raw_haemo, AROI_glm, on_missing ='ignore'),
            ATROI_glm=picks_pair_to_idx(raw_haemo, ATROI_glm, on_missing ='ignore'))
        


#%%
  


# compare statsitically for all chan (hbo only)
input_data = df_channelstats.query("Condition in ['AT_SPIN', 'A_SPIN','T']")  #'T'
input_data = input_data.query("Chroma in ['hbo']")
input_data = input_data.query("ROI in ['ATROI_glm']") # ATROI_waves
roi_model = smf.mixedlm("Beta ~-1+ Condition:ROI", input_data, #roi_model = smf.mixedlm("Value ~-1+ Condition:ROI", input_data,
                        groups=input_data["ID"]).fit()
roi_model.summary()
df_lme_glm= statsmodels_to_results(roi_model)
df_lme_glm.reset_index(drop=True, inplace=True)
df_lme_glm = df_lme_waveslongchans.set_index(['ROI', 'Condition'])


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

df_lme_waveslongchans.to_csv(savepath + r'\waveforms\df_waveforms_RQ3_AT_activationROI.csv')


#%% MSI #


input_data = df_waveformamplitudes.query("Condition in ['AT_SPIN', 'A_SPIN','T']")  #'T'
input_data = input_data.query("Chroma in ['hbo']")
input_data = input_data.query("ROI in ['AROI_waves']") #  !!!!ATROI_waves
roi_model = smf.mixedlm("Value ~ Condition", input_data, groups=input_data["ID"]).fit()

roi_model.summary()
df_lme_waveslongchans= statsmodels_to_results(roi_model)




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



# Max criterion: keep only max of A or T (most likely the same as considering AT and A in the first place)
# removing T condition changes 


# Filter A_SPIN and T conditions


a_spin_data = df_waveformamplitudes.query("Condition == 'A_SPIN' and Chroma == 'hbo' and ROI == 'AROI_waves'")  #!!
t_data = df_waveformamplitudes.query("Condition == 'T' and Chroma == 'hbo' and ROI == 'AROI_waves'")#         !!!!


# a_spin_data = df_waveformamplitudes.query("Condition == 'A_SPIN' and Chroma == 'hbo' and ROI == 'ATROI_waves'")
# t_data = df_waveformamplitudes.query("Condition == 'T' and Chroma == 'hbo' and ROI == 'ATROI_waves'")

# Merge the two dataframes on ID to compare their values
comparison_df = a_spin_data.merge(t_data, on='ID', suffixes=('_A', '_T'))

# Check if A_SPIN value is always greater or equal to T value
comparison_df['A_greater_T'] = comparison_df['Value_A'] >= comparison_df['Value_T']
if comparison_df['A_greater_T'].all():
    print("A_SPIN is always greater or equal to T.")
else:
    print("There are cases where T is greater than A_SPIN.")



# Create a copy of the original dataframe
df_waveformamplitudes_MSI = df_waveformamplitudes.copy()

# Map the conditions to a common group for aggregation while keeping the original condition
df_waveformamplitudes_MSI['MSI_Condition'] = df_waveformamplitudes_MSI['Condition'].replace({'A_SPIN': 'Unisensory', 'T': 'Unisensory', 'AT_SPIN': 'Multisensory'}) 
# For Unisensory data, find the row with the maximum value for each group and keep the original condition
unisensory_max = df_waveformamplitudes_MSI[df_waveformamplitudes_MSI['MSI_Condition'] == 'Unisensory'].groupby(['ID', 'ROI', 'Chroma'], as_index=False).apply(lambda x: x.loc[x['Value'].idxmax()])

# Filter to keep only Multisensory condition data
multisensory_data = df_waveformamplitudes_MSI[df_waveformamplitudes_MSI['Condition'] == 'AT_SPIN']

# Combine the maximum Unisensory data with the Multisensory data
df_waveformamplitudes_MSI_final = pd.concat([unisensory_max, multisensory_data], ignore_index=True)#.drop(columns=[])





input_data = df_waveformamplitudes_MSI_final.query("MSI_Condition in ['Multisensory', 'Unisensory']")  
input_data = input_data.query("Chroma in ['hbo']")
input_data = input_data.query("ROI in ['AROI_waves']") # ATROI_waves
roi_model = smf.mixedlm("Value ~ MSI_Condition", input_data, groups=input_data["ID"]).fit()

roi_model.summary()
df_lme_waveslongchans= statsmodels_to_results(roi_model)

    
    
    
    
    
    
    