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
from mne_nirs.preprocessing import peak_power, scalp_coupling_index_windowed
from mne_nirs.visualisation import plot_timechannel_quality_metric

import pickle

import sys
#sys.path.append(r'C:\GitHub\audiohaptics_exp1\selfwrittenFunctions\functions complementing mne')
#from short_channel_regression_GLM_ALL import short_channel_regression_GLM_ALL

#sys.path.append(r'C:\GitHub\audiohaptics_exp1\external')
#from SignalQualityIndex import SQI

#%% 1st level analysis: same preprocessing for glm and block
    
def individual_analysis(subjfolder, plots, stimrenaming, T_Noi_rename, glm, waveform):

    #%% preprocessing
    
    #set rejection parameters 
    Nstdtoforpeaktopeakreject = 9 # only for short chans?
    fixed_cutoff = dict(hbo=45e-6)
    sci_cutoff = 0.7
    cv_cutoff= 0.2
    
#1. load data ###############################################################
    #root= r'C:\Users\AICU\OneDrive - Demant\Desktop\audtacloc study data\Subject'
    root = r"C:\Users\bede\OneDrive - Danmarks Tekniske Universitet\fNIRS - PhD Project\Data\Alina's study\study data"
    raw = mne.io.read_raw_nirx(root+subjfolder, verbose=True, preload=True)
    #raw = raw.resample(1) # resampling acts like a low-pass filter. good to reduce autocorrelations, and speed up processing time, risk of loosing data, too much smoothing, aliasing
            #huppert:loss of stats power, he uses normally 5 Hz
# 2. delete exp start ######################################################
    annotations = raw.annotations
    eventlist = annotations.description
    raw.annotations.delete(np.where(eventlist =='11.0')) # exp start
#3. set stimulus duration for boxcar funct###################################
    stimdur = raw.info['sfreq']  # 3.4# try shorter boxcar original stim dur 5.7
    raw.annotations.set_durations(stimdur)   

#4.  rename events based on given input #####################################
    raw.annotations.rename(stimrenaming)
        
#5. trim baseline ##########################################################
    first_event_time = raw.annotations.onset[0]  # Time of the first event
    last_event_time = raw.annotations.onset[-1]  # Time of the last event
    # Define pre and post baseline durations
    pre_baseline = 10  # 10 seconds before the first event
    post_baseline = 18  # 18 seconds after the last event
    # Calculate the start and end times for cropping
    start_time = max(first_event_time - pre_baseline, 0)  # Ensure start_time is not negative
    end_time = min(last_event_time + post_baseline, raw.times[-1])  # Ensure end_time does not exceed the recording
    # Crop the raw object
    raw = raw.copy().crop(tmin=start_time, tmax=end_time)


 # [6. delete channels with unreasonable distance (that were accidently in montage for this data set)]
    dists = mne.preprocessing.nirs.source_detector_distances(raw.info)
    ch_names_all = raw.ch_names 
    ch_names_keep = np.array(ch_names_all)[(dists > 0.021) | (dists < 0.01)].tolist() # should be 114, includes 16 short chans (hbo + hbr), 98 long chans
    raw = raw.copy().pick(ch_names_keep) 
     
#7. no resampling # the higher the better (but not real improvement for sampling rates > 0.6)  (luke et al.2021)

#8. converson to  to OD  ##########################################################
    raw_od = mne.preprocessing.nirs.optical_density(raw)
    
    
#9.  Bad channel rejectsion
#old matlab chan reject, just for comparisons to new MNE sCi metric:
    # 1) matlab bads and define id
    if subjfolder ==r'01\_002':
         raw_od.info['bads']= raw_od.info['bads']+['S1_D16 760', 'S1_D16 850','S2_D17 760', 'S2_D17 850','S6_D18 760', 'S6_D18 850','S9_D21 760','S9_D21 850','S16_D23 760','S16_D23 850']
         ID = 1
    elif subjfolder ==r'02\2023-04-19_001':
        ID = 2
        raw_od.info['bads']= raw_od.info['bads']+['S7_D4 760', 'S7_D4 850','S7_D5 760', 'S7_D5 850','S7_D19 760', 'S7_D19 850']
    elif subjfolder ==r'03\2023-04-21_001':
        raw_od.info['bads'] = raw_od.info['bads']+['S2_D1 760', 'S2_D1 850', 'S2_D17 760', 'S2_D17 850','S9_D21 760', 'S9_D21 850']
        ID = 3 # 2-1 added because it causes half of the epochs to be rejected
    elif subjfolder ==r'04\2023-04-24_001':
        raw_od.info['bads']= raw_od.info['bads']+['S2_D3 760', 'S2_D3 850','S3_D4 760', 'S3_D4 850','S3_D5 760', 'S3_D5 850','S4_D5 760', 'S4_D5 850','S5_D5 760', 'S5_D5 850','S6_D4 760', 'S6_D4 850','S6_D18 760', 'S6_D18 850','S3_D5 760', 'S3_D5 850','S7_D4 760', 'S7_D4 850','S7_D5 760', 'S7_D5 850','S7_D19 760', 'S7_D19 850','S8_D8 760', 'S8_D8 850','S11_D8 760', 'S11_D8 850','S11_D12 760', 'S11_D12 850','S14_D14 760', 'S14_D14 850','S14_D15 760', 'S14_D15 850','S15_D15 760', 'S15_D15 850','S15_D14 760', 'S15_D14 850','S16_D23 760','S16_D23 850']      
        ID = 4
    elif subjfolder ==r'07\2023-05-04_001':
        raw_od.info['bads']= raw_od.info['bads']+['S1_D16 760', 'S1_D16 850','S7_D19 760', 'S7_D19 850','S9_D21 760','S9_D21 850','S12_D22 760', 'S12_D22 850','S14_D15 760', 'S14_D15 850']      
        ID = 7    
    elif subjfolder ==r'08\2023-12-18_002':
            raw_od.info['bads']= raw_od.info['bads'] # no bad channels
            ID = 8
    elif subjfolder ==r'09\2023-12-21_001':
            raw_od.info['bads']= raw_od.info['bads'] +['S2_D17 760', 'S2_D17 850', 'S6_D18 760', 'S6_D18 850','S9_D21 760','S9_D21 850','S10_D8 760', 'S10_D8 850','S16_D23 760','S16_D23 850']
            ID = 9
    elif subjfolder ==r'10\2024-01-04_001':
            raw_od.info['bads']= raw_od.info['bads'] +['S3_D4 760', 'S3_D4 850', 'S4_D4 760', 'S4_D4 850','S5_D5 760', 'S5_D5 850' ,'S6_D4 760', 'S6_D4 850', 'S6_D18 760', 'S6_D18 850', 'S7_D4 760', 'S7_D4 850','S7_D5 760', 'S7_D5 850','S7_D19 760', 'S7_D19 850', 'S9_D21 760','S9_D21 850', 'S11_D8 760', 'S11_D8 850','S12_D22 760', 'S12_D22 850', 'S14_D15 760', 'S14_D15 850', 'S15_D14 760', 'S15_D14 850','S16_D13 760','S16_D13 850','S16_D14 760','S16_D14 850', 'S16_D23 760','S16_D23 850']
            ID = 10
    elif subjfolder ==r'11\2024-01-09_001':
            raw_od.info['bads']= raw_od.info['bads'] +['S9_D21 760','S9_D21 850']
            ID = 11
    elif subjfolder ==r'12\2024-01-16_003':
            raw_od.info['bads']= raw_od.info['bads'] +['S9_D9 760','S9_D9 850','S9_D21 760','S9_D21 850','S10_D8 760', 'S10_D8 850','S11_D8 760', 'S11_D8 850','S12_D8 760', 'S12_D8 850' , 'S12_D22 760', 'S12_D22 850']
            ID = 12
    elif subjfolder ==r'13\2024-01-19_001':
            raw_od.info['bads']= raw_od.info['bads']
            ID = 13
    elif subjfolder ==r'14\2024-01-19_002':
            raw_od.info['bads']= raw_od.info['bads'] +['S2_D3 760', 'S2_D3 850','S6_D3 760', 'S6_D3 850', 'S6_D18 760', 'S6_D18 850','S7_D4 760', 'S7_D4 850','S7_D5 760', 'S7_D5 850','S7_D19 760', 'S7_D19 850', 'S11_D8 760', 'S11_D8 850', 'S14_D15 760', 'S14_D15 850']
            ID = 14
    elif subjfolder ==r'15\2024-01-23_001':
            raw_od.info['bads']= raw_od.info['bads'] 
            ID = 15
    elif subjfolder ==r'16\2024-01-23_002':
            raw_od.info['bads']= raw_od.info['bads'] +['S2_D17 760', 'S2_D17 850', 'S3_D2 760', 'S3_D2 850', 'S6_D4 760', 'S6_D4 850', 'S6_D18 760', 'S6_D18 850', 'S8_D20 760', 'S8_D20 850','S9_D10 760', 'S9_D10 850','S9_D21 760','S9_D21 850','S10_D9 760', 'S10_D9 850','S10_D10 760', 'S10_D10 850','S12_D10 760', 'S12_D10 850','S12_D22 760', 'S12_D22 850', 'S13_D9 760', 'S13_D9 850', 'S13_D13 760', 'S13_D13 850',  'S14_D10 760', 'S14_D10 850',  'S14_D13 760', 'S14_D13 850', 'S14_D15 760', 'S14_D15 850', 'S15_D9 760', 'S15_D9 850', 'S15_D10 760', 'S15_D10 850', 'S15_D15 760', 'S15_D15 850', 'S16_D13 760','S16_D13 850']
            ID = 16
    elif subjfolder ==r'17\2024-01-24_003':
            raw_od.info['bads']= raw_od.info['bads'] +['S9_D7 760', 'S9_D7 850','S11_D8 760', 'S11_D8 850','S12_D8 760', 'S12_D8 850']
            ID = 17
    # 18 has missing triggers, excluded
    elif subjfolder ==r'19\2024-01-26_002':
            raw_od.info['bads']= raw_od.info['bads'] +['S16_D23 760', 'S16_D23 850']
            ID = 19
    elif subjfolder == r'20\2024-01-29_001':
            raw_od.info['bads']= raw_od.info['bads'] # no bad channels
            ID = 20
    elif subjfolder ==r'21\2024-01-30_001':
            raw_od.info['bads']= raw_od.info['bads'] +['S6_D18 760', 'S6_D18 850','S11_D8 760', 'S11_D8 850','S16_D23 760', 'S16_D23 850']
          # add 10_8?? 'S10_D8 760', 'S10_D8 850'
            ID = 21                                                                                                    
    elif subjfolder == r'22\2024-02-01_001':
            raw_od.info['bads']= raw_od.info['bads'] +['S9_D21 760', 'S9_D21 850']
            ID = 22   
    elif subjfolder ==r'23\2024-02-08_001':
            raw_od.info['bads']= raw_od.info['bads'] +['S16_D23 760', 'S16_D23 850']
            ID = 23  
                # 24 verybad -> discarded
    elif subjfolder ==r'25\2024-02-15_001':
            raw_od.info['bads']= raw_od.info['bads'] +['S7_D4 760', 'S7_D4 850','S7_D5 760', 'S7_D5 850','S7_D19 760', 'S7_D19 850','S8_D8 760', 'S8_D8 850','S8_D20 760', 'S8_D20 850','S11_D11 760', 'S11_D11 850','S11_D12 760', 'S11_D12 850','S12_D11 760', 'S12_D11 850', 'S14_D15 760', 'S14_D15 850']
            ID = 25  
    

        
    events, event_dict = mne.events_from_annotations(raw_od)
    events = events[np.argsort(events[:, 0])]
    
    cropped_blocks = []
    gap_threshold = 30  # seconds
    post_stimulus_buffer = 15  # seconds to include after the last stimulus in a block
    pre_stimulus_buffer = 5 # seconds to exclude before the start of the next block
    
    start_idx = 0
    while start_idx < len(events):
        # Start of the block is at the onset of the current event
        block_start = events[start_idx, 0] / raw_od.info['sfreq'] - pre_stimulus_buffer
        
        # Find the end of the block
        end_idx = start_idx + 1
        while end_idx < len(events) and (events[end_idx, 0] - events[end_idx - 1, 0]) / raw_od.info['sfreq'] < gap_threshold:
            end_idx += 1
    
        # End of the block is determined by the last event plus a buffer
        block_end = events[end_idx - 1, 0] / raw_od.info['sfreq'] + post_stimulus_buffer
    
        # Ensure block_end does not exceed the maximum time in the raw data
        block_end = min(block_end, raw_od.times[-1])
    
        if block_start < block_end:
            # Crop the data
            cropped_block = raw_od.copy().crop(tmin=block_start, tmax=block_end)
            cropped_blocks.append(cropped_block)
        else:
            print(f"Skipping block due to invalid time range: tmin={block_start}, tmax={block_end}")
    
        # Move to the next set of events
        start_idx = end_idx
    
    print(f"Total blocks identified: {len(cropped_blocks)}")
    
    # Concatenate the cropped blocks
    concatenated_raw = mne.concatenate_raws(cropped_blocks)

    

    # Now you can compute the SCI on the concatenated data
    sci = mne.preprocessing.nirs.scalp_coupling_index(concatenated_raw)
    # fig, ax = plt.subplots()
    # ax.hist(sci)
    # ax.set(xlabel='Scalp Coupling Index', ylabel='Count', xlim=[0, 1])
    # plt.show()
    
    # Identify bad channels based on SCI for the concatenated data
    bads_SCI = list(compress(concatenated_raw.ch_names, sci <sci_cutoff)) # 0.8 recommended, 0.89 knee point indata without already rejected subjects
    print(bads_SCI)
    print("Channels with SCI < {sci_cutoff}:", bads_SCI)
    
    # check peak power
    concatenated_raw, PPscores, times = peak_power(concatenated_raw, time_window=10)  # 60 s windows 
    # plot_timechannel_quality_metric(concatenated_raw, scores, times, threshold=0.1,
    #                             title="Peak Power Quality Evaluation")
    howmanybadPP = np.sum(PPscores < 0.1,axis = 1)
    howmanybadPP_percent = howmanybadPP/PPscores.shape[1]
    #bads_PP = list(compress(concatenated_raw.ch_names, np.mean(PPscores,1) < 0.1)) # does not make much sense to average, better have a count.. mor ethan 50% of windows under 0.1
    bads_PP = [name for name, percent in zip(concatenated_raw.ch_names, howmanybadPP_percent) if percent > 0.5]
    print(bads_PP)
    
    print("Channels with > 50% 10s windows with Peak power < 0.1:", bads_PP)
   
    ChanStd = np.std(concatenated_raw._data, axis=1)
    bads_CV = list(compress(concatenated_raw.ch_names, ChanStd > cv_cutoff))  # knee point 

    bads_CV_full = set()
    for chan in bads_CV:
        import re
        base_name = re.sub(r'\s+\d+$', '', chan)  # Remove the wavelength part
        # Add both wavelength versions to the set
        bads_CV_full.add(f"{base_name} 760")
        bads_CV_full.add(f"{base_name} 850")

    # Now update bads_CV to include pairs
    bads_CV = list(bads_CV_full)
    
    print("Channels with CV > {cv_cutoff}:", bads_CV)
    bads_PP_set = set(bads_PP)
    bads_SCI_set = set(bads_SCI)
    #inSCIbutnotPP= bads_SCI_set.difference(bads_PP_set)
    # inPPbutnotSCI = bads_PP_set.difference(bads_SCI_set)
    # print("Channels detected as bad by SCI but not by PeakPower :", inSCIbutnotPP)
    # print("Channels as bad by PeakPower but not by SCI:", inPPbutnotSCI)
   
    bads_SCI_PP_set = bads_SCI_set.union(bads_PP_set)
    
    # add also channels found by CV to set
    bads_CV_set = set(bads_CV)
    bads_SCI_PP_CV_set = bads_SCI_PP_set.union(bads_CV_set)
    bads_SCI_PP_CV = list(bads_SCI_PP_CV_set)

    # output_file_path = r'C:\Users\AICU\OneDrive - Demant\Documents - Eriksholm Research Centre\Research Projects\AudTacLoc\fNIRS results - reanalysis with more subjects\Final files_21_03_24\newMNEbadchanreject\separateConditions_MAINANALYSIS/infoBadChansSubj_'+ str(ID) +'.txt'

    # # Open the file in write mode ('w')
    # with open(output_file_path, 'w') as file:
    #     # Write the information to the file
    #     # file.write(f"Channels detected as bad by SCI but not by PeakPower: {inSCIbutnotPP}\n")
    #     # file.write(f"Channels as bad by PeakPower but not by SCI: {inPPbutnotSCI}\n")
    #     file.write(f"Data was cut into {len(cropped_blocks)} blocks and reconcatenated to base bad chan rejection on segments with stimulus events only")
    #     file.write(f"Bad channels as detected by SCI: {bads_SCI}\n")
    #     file.write(f"Bad channels as detected by peak power: {bads_PP}\n")
    #     file.write(f"Bad channels as detected by CV: {bads_CV}\n")
    #     file.write(f" Union of all-->Bad channels excluded for further analyses: {bads_SCI_PP_CV}\n")

    # use mne diff for rejection
    raw_od.info['bads']= bads_SCI_PP_CV
    
    #if plots:
        # concatenated_raw.plot(n_channels=len(concatenated_raw.ch_names),
        #         duration=500, show_scrollbars=False)
        # events, event_dict = mne.events_from_annotations(concatenated_raw)
        # fig = mne.viz.plot_events(events, event_id=event_dict,
        #                   sfreq=raw.info['sfreq'])
        
        # raw_od.plot(n_channels=len(raw_od.ch_names),
        #         duration=500, show_scrollbars=False)
    
    #############################repeat SCI on long chan only to add this info to plot later:
  
    ODcopy = get_long_channels(concatenated_raw.copy())
    sci_longchans = mne.preprocessing.nirs.scalp_coupling_index(ODcopy)
    sci_longchans = sci_longchans[::2] #keep one val per channel only
    # lsits should match chnames of ODcopy hbo only
        # same for CV 
    CV_longchans = np.std(ODcopy._data, axis=1) 
    CV_longchans = CV_longchans[::2] #keep the hbo value only    
        # and pp
    ODcopy, PPscores_longch, times = peak_power(ODcopy, time_window=10)  # 60 s windows 
    # plot_timechannel_quality_metric(concatenated_raw, scores, times, threshold=0.1,
    #                             title="Peak Power Quality Evaluation")
    howmanybadPP_longch = np.sum(PPscores_longch < 0.1,axis = 1)
    howmanybadPP_percent_longch = howmanybadPP_longch/PPscores_longch.shape[1]
    howmanybadPP_percent_longch = howmanybadPP_longch[::2]
             

    # ###### check SQI for comparison
    # # needs OD and HB signals of one channel
    # #loop over data, extract the four parameters and call SQI for that channel
    # fs = raw_od.info['sfreq']  # Sampling frequency
    
    # # Convert optical density to haemoglobin concentration
    # raw_haemo_copy = mne.preprocessing.nirs.beer_lambert_law(raw_od.copy(), ppf=0.1)
    
    # # Example processing loop through channels
    # # SQIs = []
    # # for i in range(0, raw_od._data.shape[0], 2):  # Assuming channels are paired: WL1, WL2
    # #     od1 = raw_od._data[:, i]   # Wavelength 1
    # #     od2 = raw_od._data[:, i+1] # Wavelength 2
        
    # #     hbo = raw_haemo_copy._data[:, i]   # Corresponding HbO
    # #     hbr = raw_haemo_copy._data[:, i+1] # Corresponding HbR
        
    # #     # Call SQI or any other quality metric function
    # #     SQIvalue = SQI(od2, od1, hbo, hbr, fs)
    # #     SQIs.append(SQIvalue)
      
    
    
# Continue with your existing code for SCI and peak power analysis...
 
    corrected_tddr = temporal_derivative_distribution_repair(raw_od)
    
    #HB
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(corrected_tddr, ppf = 0.1); # default option, same as in nirstoolbox (it comes from dpf =6/ partial volume correctioin = 60), however ppf = 6 is still recommended  in nirstoolbox PPF = 5 / 50;   % partial pathlength factor 
    
    
    raw_haemo_longandshort = raw_haemo.copy() 
    
    #Filtering
    raw_haemo = raw_haemo.filter(l_freq = None, h_freq = 0.25,  
                                 method="iir", iir_params =dict(order=5, ftype='butter'))
     #high-pass
    raw_haemo= raw_haemo.filter(l_freq =  0.005, h_freq = None, method="iir", iir_params =dict(order=5, ftype='butter'))#t0.05 was cutoff in andreas analysis
      

    
    if ID >18:
         raw_haemo.annotations.rename(T_Noi_rename)
    
    annotations = raw_haemo.annotations
    eventlist = annotations.description
    raw_haemo.annotations.delete(np.where(eventlist =='Delete')) # exp start # delete conditions of no interest -> helps epoch cleaning
        
    
    ####### COMMENTING FROM HERE ONWARDS TO SEE IF I CAN RUN IT
    
    """
    
    # Short channel regression
    raw_haemo, short_chs = short_channel_regression_GLM_ALL(raw_haemo)
        
   
    # Negative correlation enhancement
    raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)
     # the middle long channels are classified as bad short channels, are now in short_chs, but were not used as regressor
    # but are still in ch_names list while real short channels are not
    
    # drop bad chans entirely to avoid confusion later -> caused confusion too
   # raw_haemo= raw_haemo.drop_channels(raw_haemo.info["bads"])
  #  datacheck= raw_haemo2.get_data()
  #
    eventsall, event_dict_all = mne.events_from_annotations(raw_haemo, verbose=False)
   # events, event_dict = mne.events_from_annotations(raw_haemo,{'Control_Sil':1, 'Noise_Sil':2, 'Speech_Sil':3})
    


#%%
    if waveform:
  #        #% #########epoching for block average
  #       # hbTempRoi = raw_haemo.copy().pick([60,61,80,81,100,101])
  #       # hbTempRoi = raw_haemo.copy().pick([80,81,84,85,86,87,94,95,96,97,36,37,26,27,12,13,22,23,6,7]) #old picks when not removing middle long chans from montage
      
  #        # hbTempRoi = raw_haemo.copy().pick([80,81,84,85,86,87,94,95,96,97,36,37,26,27,12,13,22,23,6,7]) #old picks when not removing middle long chans from montage
  #        good_channels = [ch for ch in [6,7,8,9,10,11,14,15,16,17,18,19,20,21,28,29,68,69,70,71,72,73,74,75,78,79,80,81,82,83,84,85,86,87,90,91] if f'S{ch // 10}_D{ch % 10} hbo' not in raw_haemo.info['bads'] and f'S{ch // 10}_D{ch % 10} hbr' not in raw_haemo.info['bads']]
  # # with picks it needs to made sure explicitly that bads should not be picked..!
  # # no roi weighting -> set to false in glm too if it should be comparable
  #        hbTempRoi = raw_haemo.copy().pick(good_channels)

  #              l+ r STg exactly as in robs paper + extra ones inbetween, HBO and HBR for every channel also listed below as ROI lSTG + rSTG

     # epoch start and end
        tmin, tmax = -5, 20
    
         # epochs for all channels 
         # Problem: bad channel information is not used -> rejects more epochs based on threshold exceedings in bad chans..?!
        reject_criteria = fixed_cutoff # dict(hbo=1000e-6)# fixed_cutoff # going for a fixed cutoff instead#dict(hbo=1000e-6)
        epochs_allCond = mne.Epochs(raw_haemo, eventsall, event_id=event_dict_all,
                             tmin=tmin, tmax=tmax,
                             reject=reject_criteria,reject_by_annotation=True,
                             proj=True, baseline=(None, 0), preload=True,
                             detrend=1, verbose=True)
        
        # #largerthanstdsALLcond = Nstdtoforpeaktopeakreject*np.std(epochs_allCond._data) # std over whole longchan data, individual threshold per subject, but not per chan
        # reject_criteria = dict(hbo=largerthanstdsALLcond)
        # epochs_allCond = mne.Epochs(raw_haemo, eventsall, event_id=event_dict_all,
        #                       tmin=tmin, tmax=tmax,
        #                       reject=reject_criteria,reject_by_annotation=True,
        #                       proj=True, baseline=(None, 0), preload=True,
        #                       detrend=1, verbose=True)
        
       # epochs_allCond.plot_drop_log() 
        droppedepochsStats =epochs_allCond.drop_log_stats() # in percent
        
        # one copy of these with all stim together for plotting
        hbcopy = raw_haemo.copy()
        hbcopy.annotations.rename({'Speech_Sil':'All_Sil_Conditions', 'Noise_Sil':'All_Sil_Conditions', 'Control_Sil':'All_Sil_Conditions', 'A_SPIN_Sil':'All_Sil_Conditions',  'T_Sil':'All_Sil_Conditions', 'AT_SPIN_Sil':'All_Sil_Conditions'})
        eventscopy, event_dict_copy = mne.events_from_annotations(hbcopy, verbose=False)
        

        #reject_criteria = dict(hbo=largerthanstdsALLcond) # use rejection cutoff as from complete data but look particulalry at effect on conditions of interest
        reject_criteria = fixed_cutoff
        epochs_Sil = mne.Epochs(hbcopy, eventscopy, event_id=event_dict_copy,
                             tmin=tmin, tmax=tmax,
                             reject=reject_criteria,reject_by_annotation=True,
                             proj=True, baseline=(None, 0), preload=True,
                             detrend=1, verbose=True)
        
        
        # short channels
            
        reject_criteria = dict(hbo=1000e-6)
        epochs_SChs = mne.Epochs(short_chs, eventsall, event_id=event_dict_all,
                              tmin=tmin, tmax=tmax,
                              reject=reject_criteria,reject_by_annotation=True,
                              proj=True, baseline=(None, 0), preload=True,
                              detrend=1, verbose=True)
        largerthanstdsSC = Nstdtoforpeaktopeakreject*np.std(epochs_SChs._data) # two or three times std
        reject_criteria = dict(hbo=largerthanstdsSC)
        epochs_SChs = mne.Epochs(short_chs, eventsall, event_id=event_dict_all,
                              tmin=tmin, tmax=tmax,
                              reject=reject_criteria,reject_by_annotation=True,
                              proj=True, baseline=(None, 0), preload=True,
                              detrend=1, verbose=True)
       # epochs_SChs.plot_drop_log()
        
    
    if glm:
    
        isis = mne_nirs.experimental_design.longest_inter_annotation_interval(raw_haemo)
        cosine_hp =1/(2*max(isis[0]))
        print(cosine_hp)  


        #GLM
        design_matrix = make_first_level_design_matrix(raw_haemo,
        drift_model='cosine', high_pass=cosine_hp,  # Must be specified per experiment
        hrf_model='glover',
        stim_dur=stimdur)
        
        # fig, ax1 = plt.subplots(figsize=(10, 6), nrows=1, ncols=1)
        # plot_design_matrix(design_matrix, ax=ax1)
        # fig.savefig(r"C:\Users\AICU\OneDrive - Demant\Documents - Eriksholm Research Centre\Research Projects\AudTacLoc\fNIRS results - reanalysis with more subjects\Final files_MNE analysis\stim vs no stim\GLM\designmatrix_glm_stimnostim.svg", format='svg')


       

        glm_est = run_glm(raw_haemo, design_matrix, noise_model = 'ar5') #still has info with bads listed, but the data frame does include the bad channels?!?!?

        # Extract channel metrics
        #channelstats = glm_est.pick(picks= "fnirs", exclude ="bads").to_dataframe()  # better to exclude form list or to nan? try what causes more problems in group analysis and plotting functions
        channelstats = glm_est.to_dataframe()  # better to exclude form list or to nan? try what causes more problems in group analysis and plotting functions
      # no
        # not posible to exclude  channelstats = glm_est.to_dataframe(exclude = "bads") 
       
        # Iterate through the rows of the DataFrame
        for idx, row in channelstats.iterrows():
            # Check if the channel in this row is marked as bad
            if row['ch_name'] in glm_est.info['bads']:
                # Set the values to NaN
                channelstats.loc[idx, ['theta', 't', 'p_value', 'dof']] = np.nan  # Replace 'theta', 't', 'p_value', 'dof' with the actual column names of your estimates
        
        #add ROIs
        lIFG = [[8,6],[9,6],[9,7],[10,6],[10,7]]; # 8-6 and 9-6 are the same as in luke
        #lSTG = [[13,9],[13,13],[14,9],[14,13],[14,14]] # exactly as in robs paper
        lSTG = [[13,9],[13,13],[14,9],[14,10],[14,13],[14,14],[15,10],[15,9],[15,14],[14,15]] # exactly as in robs paper + extra ones inbetween
        #rSTG = [[7,5],[5,5],[3,5],[5,2],[3,2]] # as in luke except one chan that we don't have
        rSTG = [[7,5],[5,5],[3,5],[5,2],[3,2],[4,4],[4,5],[3,4]]# 6_4 and 7_4 additonal channels in our montage (posterior)

        bilatSTG = lSTG + rSTG
        lSC =[[9,9],[11,8],[11,9],[11,11],[11,12],[12,8],[12,11],[12,12]]
        rSC =[[2,3],[4,3],[6,3]]
        bilatSC= lSC + rSC
        lpostTemp = [[15,12], [15,14],[15,15],[14,14],[14,15],[14,13],[16,13],[16,14],[14,11]] # maybe not 14,11

        # Define the function to filter out bad channels 
        def filter_bad_channels(roi_def, info):
            """ "Filter out bad channels from the ROI definitions.""""
            filtered_roi_def = []
            for pair in roi_def:
                ch_name_hbo = f'S{pair[0]}_D{pair[1]} hbo'
                ch_name_hbr = f'S{pair[0]}_D{pair[1]} hbr'
                if ch_name_hbo not in info['bads'] and ch_name_hbr not in info['bads']:
                    filtered_roi_def.append(pair)
            return filtered_roi_def
        
        # Filter out bad channels from each ROI
        lIFG = filter_bad_channels(lIFG, raw_haemo.info)
        lSTG = filter_bad_channels(lSTG, raw_haemo.info)
        rSTG = filter_bad_channels(rSTG, raw_haemo.info)
        lSC = filter_bad_channels(lSC, raw_haemo.info)
        rSC = filter_bad_channels(rSC, raw_haemo.info)
        lpostTemp = filter_bad_channels(lpostTemp, raw_haemo.info)

        rois = dict(lSTG=picks_pair_to_idx(raw_haemo, lSTG, on_missing ='ignore'),
                        rSTG =picks_pair_to_idx(raw_haemo, rSTG, on_missing ='ignore'),
                        bilateralSTG =picks_pair_to_idx(raw_haemo, bilatSTG, on_missing ='ignore'),
                        Left_SC=picks_pair_to_idx(raw_haemo, lSC, on_missing ='ignore'),
                        Right_SC=picks_pair_to_idx(raw_haemo, rSC, on_missing ='ignore'),
                        Bilateral_SC=picks_pair_to_idx(raw_haemo, bilatSC, on_missing ='ignore'),
                        leftIFG=picks_pair_to_idx(raw_haemo, lIFG, on_missing ='ignore'),
                        leftPostTemp=picks_pair_to_idx(raw_haemo, lpostTemp, on_missing ='ignore'))
            
        
    
        # roi_glm = glm_est.pick(picks= "fnirs", exclude ="bads").to_dataframe_region_of_interest(rois,
        #                                               design_matrix.columns,
        #                                               demographic_info=True) # exclude = "bads" excists for picks of regression results
        #                         # this instead of filter_bad_channels before does not work..gives error, don't understand where it comes from
                                
                                
        roi_glm = glm_est.to_dataframe_region_of_interest(rois,
                                                      design_matrix.columns,
                                                      demographic_info=True, weighted=False) # exclude = "bads" excists for picks of regression results
        # weighted by inverse standard error if not weighted  = False
        
        # evoked responses in short channels
        short_ch_glm_est = run_glm(short_chs, design_matrix, noise_model = 'ar5')
       # short_ch_glm = short_ch_glm_est.to_dataframe() 
        
        # short chan rois
        allshortchans = [[1,16],[2,17],[6,18],[7,19],[8,20],[9,21],[12,22],[16,23]]
        shortchanstemppost = [[6,18],[7,19],[16,23],[12,22]]
        shortchansfront = [[1,16],[2,17],[8,20],[9,21]]

        # Filter out bad channels from each ROI
        allshortchans = filter_bad_channels(allshortchans, short_chs.info)
        shortchanstemppost = filter_bad_channels(shortchanstemppost , short_chs.info)
        shortchansfront = filter_bad_channels(shortchansfront,  short_chs.info)
        
        SC_ROI= dict(all_shortchans=picks_pair_to_idx(short_chs, allshortchans),
                     frontal =picks_pair_to_idx(short_chs, shortchansfront),
                     posterior_temporal = picks_pair_to_idx(short_chs,shortchanstemppost))

    
        shortch_roi_glm = short_ch_glm_est.to_dataframe_region_of_interest(SC_ROI,
                                                      design_matrix.columns,
                                                      demographic_info=True, weighted = False)
        
        # check if this has bad channels and how they are handled later on..!
    
        # Convert to uM for nicer plotting below.
        channelstats["theta"] = [t * 1.e6 for t in channelstats["theta"]]
        roi_glm["theta"] = [t * 1.e6 for t in roi_glm["theta"]]
        
        if len(short_chs.info['bads']) < len(short_chs.ch_names):
            shortch_roi_glm["theta"] = [t * 1.e6 for t in shortch_roi_glm["theta"]]
            shortch_roi_glm["ID"] = ID  # Assign the ID
        else:
            # Create an empty DataFrame with the same columns as shortch_roi_glm would have
            columns = ['theta', 'ID']  # Add other necessary columns here
            shortch_roi_glm = pd.DataFrame(columns=columns)
            # Optionally, you can insert the ID with NaN values for 'theta' or leave it completely empty
            shortch_roi_glm = pd.DataFrame({'theta': [np.nan], 'ID': [ID]})

         # Add the participant ID to the dataframes
        roi_glm["ID"] = channelstats["ID"] = roi_glm["ID"] = ID #
        
        
        
           #boxcar--> not solved yet to plot boxcar. and how hrf changes with boxcar length!!!
           
       
        # fig, ax = plt.subplots(constrained_layout=True)
        # s = mne_nirs.experimental_design.create_boxcar(raw, stim_dur=5)
        
        # #code of create boxcar function
        # bc = np.ones(int(round(raw_haemo.info['sfreq'] * stimdur)))
        # events, ids = mne.events_from_annotations(raw_haemo)
        # s = np.zeros((len(raw.times), len(ids)))
        # for idx, id in enumerate(ids):
        #     id_idx = [e[2] == idx + 1 for e in events]
        #     id_evt = events[id_idx]
        #     event_samples = [e[0] for e in id_evt]
        #     s[event_samples, idx] = 1.
        #     s[:, idx] = np.convolve(s[:, idx], bc)[:len(raw.times)]
            
            
        # ax.plot(raw_haemo.times, s[:, 1])
        # ax.plot(design_matrix['A_SPIN_Sil'])
        # ax.legend(["Stimulus", "Expected Response"])
        # ax.set(xlim=(225, 270), xlabel="Time (s)", ylabel="Amplitude") #xlim=(225, 270),
        
        # plt.plot(raw_haemo.times, s)
    if plots:
    
        #    #boxcar not working
        # fig, ax = plt.subplots(constrained_layout=True)
        # s = mne_nirs.experimental_design.create_boxcar(raw_haemo, stim_dur=1)
        # ax.plot(raw_haemo.times, s[:, 1])
        # ax.plot(design_matrix)
        # ax.legend(["Stimulus", "Expected Response"])
        # ax.set(xlim=(225, 270), xlabel="Time (s)", ylabel="Amplitude")
        
        
        # waves per subj /epochs
        # bad channels in red, bad epochs are deleted automatically out of allCond variable during epoching
        # Define your list of conditions
        
         
        #conditions = ['Speech_Sil','Noise_Sil', 'Control_Sil', 'A_SPIN_Sil', 'T_Sil', 'AT_SPIN_Sil', 'Speech_Noi', 'Control_Noi' ] 
        
        conditions = ['All_Sil_Conditions']
        # rename to one for these plots
        
        
        for condition in conditions:
            ch_dict = {}
            cond_epochs = epochs_Sil[condition].pick(picks="hbo")  #epochs_allCond[condition].pick(picks="hbo")
            cond_data = cond_epochs.get_data()  # This is now a 3D array: epochs x channels x time
            channels = cond_epochs.ch_names
        
            # Store the data, noting whether each epoch is good or bad
            for i_ch, ch in enumerate(channels):
                if ch not in ch_dict:
                    ch_dict[ch] = []
                for i_epoch in range(cond_data.shape[0]):
                    is_good = ch not in cond_epochs.info['bads']
                    ch_dict[ch].append((cond_data[i_epoch, i_ch, :], is_good))
        
            # Plot the data for the current condition
            size = int(np.ceil(np.sqrt(len(ch_dict))))
            fig, ax = plt.subplots(size, size, figsize=(15, 10))
            ax = ax.flatten()
        
            for i, (ch, epochs_data) in enumerate(ch_dict.items()):
                good_epoch_data = []
                for epoch_data, is_good in epochs_data:
                    if is_good:
                        good_epoch_data.append(epoch_data)
                        ax[i].plot(cond_epochs.times, epoch_data * 1e6, linewidth=0.5)  # Good epochs plotted in different colors
                    else:
                        ax[i].plot(cond_epochs.times, epoch_data * 1e6, color='red', linewidth=0.5)  # Bad epochs in red
        
                # Plot the mean of good epochs
                if good_epoch_data:
                    mean_data = np.mean(good_epoch_data, axis=0)
                    ax[i].plot(cond_epochs.times, mean_data * 1e6, color='black', linewidth=2)  # Mean plotted without adding to legend
        
               # Annotations for SCI and CV, PP --> need to make sure to use info of long channels only, as cshort channels are not available anymore in epoch object and wont be plotted
                sci_value = sci_longchans[i]  # Ensure this is updated if sci_longchans changes with conditions
                cv_value = CV_longchans[i]  # Coefficient of Variation for the channel
                pp_bad_percent= howmanybadPP_percent_longch[i] 
                
                # compare epoched chan names to ODcopy.ch_names
                
                ax[i].annotate(f"SCI: {sci_value:.2f}, CV: {cv_value:.2f}, PP: {pp_bad_percent:.0f}%", xy=(0.05, 0.85), xycoords='axes fraction', fontsize=9, ha='left', va='top', fontweight='bold')


                ax[i].set_title(f"{condition}: {ch}")
                ax[i].set_xlabel('Time')
                ax[i].set_ylabel('Concentration (uM)')
                ax[i].set_ylim(-50, 50)
          
            if fixed_cutoff:
                largerthanstdsALLcond = fixed_cutoff['hbo']
                
            ax[0].annotate(f"Epoch Rej. Thresh: {largerthanstdsALLcond * 1e6} μM", xy=(0.5, 1), xycoords='axes fraction', fontsize=10, ha='center', va='bottom', fontweight='bold')
            
          # Hide unused axes
            for i in range(len(ch_dict), len(ax)):
                ax[i].set_visible(False)
        
        
            # Save the figure with the condition name in the filename
            #fig.savefig(f'C:\\Users\\AICU\\OneDrive - Demant\\Documents - Eriksholm Research Centre\\Research Projects\\AudTacLoc\\fNIRS results - reanalysis with more subjects\\Final files_21_03_24\\newMNEbadchanreject\\separateConditions_MAINANALYSIS\\epochoverview_eachchannel\\Subj{str(ID)}_epochsperchan_waveforms_{condition}.svg', format='svg')
            fig.savefig(r'C:\Users\AICU\OneDrive - Demant\Desktop\subject_channel_epoch_plots aud tac loc\Sub' + str(ID) + 'epochsperchan_waveforms_' + condition + '.svg', format='svg')
            # move to right location later...
           
    if fixed_cutoff:
                largerthanstdsALLcond = fixed_cutoff['hbo']
 
    return raw_haemo,short_chs, raw_haemo_longandshort, epochs_allCond, channelstats, roi_glm, epochs_SChs, shortch_roi_glm, largerthanstdsALLcond, bads_SCI_PP_CV, droppedepochsStats, sci,  howmanybadPP_percent, ChanStd#, normvalue  epochs_TempRoi,
    # return raw_haemo, short_chs, epochs_allCond, epochs, epochs_TempRoi, epochs_SChs #, normvalue
     


"""

######## SINGLE SUBJECT ANALYSIS - INVESTIGATION ON BACJGROUN NOISE PRESENCE ##################

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

subjfolder = r"Subject01\2023-04-18_002"

