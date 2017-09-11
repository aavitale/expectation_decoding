# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%
import numpy as np
import os.path as op
import matplotlib as mpl
import matplotlib.pyplot as plt

import mne
from mne.filter import notch_filter, filter_data
from mne.time_frequency import tfr_morlet, tfr_multitaper, psd_multitaper
#mne.pick_types?


#%% LOAD DATA
# AUDIO-VISUAL data
#data_path = 'C:/Users/andvit/Desktop/=KENT - MNE'
data_path = 'C:/Users/andvit/Desktop/=KENT - MNE/PRNI_MNE_tutorial'
data_file = 'mne_audiovisual-raw.fif'
chan_stim = 'STI 014'
event_id = {"visual/left": 3, "visual/right": 4, "auditory/left": 1, "auditory/right": 2}

#file_name = data_path + '/' + data_file
file_name = op.join(data_path, data_file)
print(file_name)

data_raw = mne.io.read_raw_fif(file_name, preload=False)


# or ODDBALL data - - - - - - - - - - - - - - - - 
from mne.datasets.brainstorm import bst_auditory
from mne.io import read_raw_ctf
use_precompute = True
data_path = bst_auditory.data_path()
data_path = data_path + '/' + 'MEG' + '/' + 'bst_auditory'
data_file1 = 'S01_AEF_20131218_01.ds'
#data_file2 = 'S01_AEF_20131218_02.ds'
#subjects_dir = op.join(data_path, 'subjects')
chan_stim = 'UPPT001'
event_id = {"standard": 1, "deviant": 2}

# EMPTY ROOM NOISE
#erm_fname = op.join(data_path, 'MEG', 'bst_auditory',
#                    'S01_Noise_20131218_01.ds')

file_name1 = op.join(data_path, data_file1)
#file_name2 = op.join(data_path, data_file2)

data_raw = mne.io.read_raw_ctf(file_name1, preload=True)
#mne.io.concatenate_raws([data_raw, read_raw_ctf(file_name2, preload=True)])

#print(data_raw)
print(data_raw.info)


#%% PARAMETER
samp_rate = data_raw.info['sfreq']

chan_name = data_raw.ch_names
# list of bad channels
print(data_raw.info['bads'])

chan_meg_pick = mne.pick_types(data_raw.info, meg=True, exclude=[])
chan_mag_pick = mne.pick_types(data_raw.info, meg='mag', exclude=[])
#chan_mag_pick = np.setdiff1d(chan_meg_pick, chan_grad_pick)
chan_grad_pick = mne.pick_types(data_raw.info, meg='grad', exclude=[])
chan_eeg_pick = mne.pick_types(data_raw.info, meg=False, eeg=True, eog=False, stim=False, exclude=[])
chan_stim_pick = mne.pick_types(data_raw.info, meg=False, eeg=False, eog=False, stim=True, exclude=[])

n_chan_mag = len(chan_mag_pick)
#plt.plot(data_raw[0, 1:1000])
data_raw.plot(block=True)


#%% FILTERING
# reload data with preload
data_raw = mne.io.Raw(file_name, preload=True)

#POWER SPECTRUM
fmin, fmax = 1, 300
data_raw.plot_psd(tmax = 10.0, picks=chan_mag_pick, average=False, fmax=fmax)
#data_raw.plot_psd(tmax = 10.0, picks=chan_mag_pick, average=True, fmax=fmax)


# NOTCH filter
notch = np.arange(60, 181, 60)
data_raw.notch_filter(notch, picks=chan_mag_pick)
data_raw.plot_psd(tmax=10.0, picks=chan_mag_pick, average=True, fmax=fmax)


# HIGH and LOW pass filters
hpf = 0.1 #to remove slow drift
lpf = 100 #30.0
data_raw_filt = data_raw.filter(hpf, lpf, method='iir', picks=chan_mag_pick)
                #h_trans_bandwidth=0.5, filter_length='10s',phase='zero-double')
print(data_raw_filt.info)
data_raw_filt.plot()

#data_raw_orig = data_raw; data_raw = data_raw_filt


#%% DOWNSAMPLE before epoching
#print('Original sampling rate:', samp_rate, 'Hz')
#samp_rate_down = 250
#
##data_raw_down = data_raw.copy().resample(100, npad='auto')
#data_raw_down = data_raw.resample(samp_rate_down, npad='auto')
#data_raw = data_raw_down
#print('New sampling rate:', data_raw.info['sfreq'], 'Hz')


#%% READ EVENT
event = mne.find_events(data_raw_filt, stim_channel=chan_stim)
#[ timepoint,  , trigger]
d,time = data_raw[data_raw_filt.ch_names.index(chan_stim), :]
plt.plot(d[0,:1000])
#plt.plot(data_raw[data_raw.ch_names.index(chan_stim), :])

# how many triggers x event 1
len(event[event[:,2]==1])

event_id 
fig = mne.viz.plot_events(event, samp_rate, event_id=event_id);


#%% EPOCHING
t_min = -0.5 #if in pre-stim window should be NEGATIVE
t_max = 1

#baseline = (t_min, 0)
# NO BASELINE correction
baseline = (None, 0)

reject = {} #NO peak-to-peak rejection parameter
#or
#reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)  
#reject_mag = dict(grad=4000e-13, mag=4e-12)
reject_mag = dict(mag=4e-12)
#reject_eeg = 
reject_eog = dict(eog=150e-6)

epoch_mag = mne.Epochs(data_raw_filt, event, event_id, t_min, t_max, baseline=baseline, \
                       picks=chan_mag_pick, reject=reject_mag, preload=True) #{})
print(epoch_mag)
# remove bad epochs based on peak-to-peak parameter
epoch_mag.drop_bad()
epoch_mag.plot_drop_log();

#epoch_mag_matrix = epoch_mag.get_data()


#%% DOWNSAMPLE AFTER epoching
print('Original sampling rate:', samp_rate, 'Hz')
samp_rate_down = 250

#data_raw_down = data_raw.copy().resample(100, npad='auto')
#epochs_mag_down.load_data()  
epoch_mag_down = epoch_mag.resample(samp_rate_down, npad='auto', )
print('New sampling rate:', epoch_mag_down.info['sfreq'], 'Hz')
epoch_mag = epoch_mag_down
print('New sampling rate:', epoch_mag.info['sfreq'], 'Hz')


#%% EVOKED RESPONSE
evoked_mag_all = epoch_mag.average()
print(evoked_mag_all)

#evoked_meg_all.plot()
evoked_mag_all.plot(spatial_colors=True, gfp=True)

# TOPOGRAPHY plots
evoked_mag_all.plot_topomap(times=np.linspace(0.05, 0.2, 10));
evoked_mag_all.plot_topomap(times=np.linspace(0.3, 0.5, 10));
evoked_mag_all.plot_joint()

#for chan_type in ('mag', 'grad'):
#    evoked_meg_all.plot_topomap(times=np.linspace(0.05, 0.15, 10), ch_type=chan_type);
#
#    evoked_meg_all.plot_joint(ch_type=chan_type)

#or
#epoch_meg.average().pick_types(meg='grad').crop(None, 0.2).plot(spatial_colors=True)
epoch_mag.average().crop(0, 0.2).plot(spatial_colors=True)


#%% compute CONTRAST
# AUDITORY STANDARD
evoked_mag_std = epoch_mag['standard'].average()
evoked_mag_dev = epoch_mag['deviant'].average()


time_interest = [0.11, 0.175, 0.3, 0.4]

evoked_mag_std.plot_joint(time_interest) #window_title='Standard', gfp=True 
evoked_mag_dev.plot_joint(time_interest)


#evoked_mag_diff = mne.combine_evoked([evoked_mag_dev, evoked_mag_std], weights='equal')
evoked_mag_diff = mne.combine_evoked([evoked_mag_dev, -evoked_mag_std], weights='equal')
evoked_mag_diff.plot(window_title='Difference', gfp=True)
evoked_mag_diff.crop(0,0.4).plot_joint(time_interest)




# =============================================================================
# AUDITORY - - - - - - - -  -
time_interest = [0.08, 0.1, 0.12, 0.14]

epoch_meg_aud_left = epoch_meg['auditory/left']
#epoch_meg_aud_left.average().plot(spatial_colors=True)
epoch_meg_aud_left.average().plot_joint()

epoch_meg_aud_right = epoch_meg['auditory/right']
#epoch_meg_aud_right.average().plot(spatial_colors=True)
epoch_meg_aud_right.average().plot_joint()

# VISUAL - - - - - -  - -
epoch_meg['visual/left'].average().pick_types(meg='grad').crop(None, 0.2).plot_joint(times=time_interest) #plot(spatial_colors=True)
epoch_meg['visual/right'].average().pick_types(meg='grad').crop(None, 0.2).plot_joint(times=time_interest) #plot(spatial_colors=True)


# CONTRAST (weighted differenced with a balanced number of trials per condition)
evoked_meg = {k:epoch_meg[k].average() for k in event_id} 
#contrast_meg_aud = evoked_meg['auditory/left'] - evoked_meg['auditory/right']
contrast_meg_aud = mne.combine_evoked([evoked_meg['auditory/left'], evoked_meg['auditory/right']], weights='equal')
print(contrast_meg_aud)

#epoch_meg_aud_left_matrix = epoch_meg_aud_left.get_data()
#epoch_meg_aud_right_matrix = epoch_meg_aud_right.get_data()
#contrast_meg_aud = epoch_meg_aud_left_matrix - epoch_meg_aud_right_matrix
fig = contrast_meg_aud.plot_joint()


contrast_meg_vis = mne.combine_evoked([evoked_meg['visual/left'], evoked_meg['visual/right']], weights='equal')
fig = contrast_meg_vis.plot_joint()


#%% FREQUENCY 
#POWER SPECTRUM topography
epoch_mag.plot_psd_topomap(normalize=True)
epoch_mag['standard'].plot_psd_topomap(normalize=True)
epoch_mag['deviant'].plot_psd_topomap(normalize=True)

#epoch_meg.plot_psd_topomap(ch_type='mag', normalize=True)
#epoch_meg.plot_psd_topomap(ch_type='grad', normalize=True)


# define frequencies of interest (log-spaced)
low_freq = 1
high_freq = 100

freq_interest = np.logspace(*np.log10([low_freq, high_freq]), num=8)
n_cycle = freq_interest / 2.  # different number of cycle per frequency

# MORLET wavelet - - - - - - - - - - -
power_std, itc = tfr_morlet(epoch_mag['standard'], freqs=freq_interest, n_cycles=n_cycle, use_fft=True,
                        return_itc=True, decim=3, n_jobs=1)
power_dev, itc = tfr_morlet(epoch_mag['deviant'], freqs=freq_interest, n_cycles=n_cycle, use_fft=True,
                        return_itc=True, decim=3, n_jobs=1)
#power, itc = tfr_morlet(epoch_meg, freqs=freq_interest, n_cycles=n_cycle, use_fft=True,
#                        return_itc=True, decim=3, n_jobs=1)

#or MULTITAPER - - - - - - - - - - - -
power, itc = tfr_multitaper(epoch_meg, freqs=freq_interest, n_cycles=n_cycle, use_fft=True,
                        return_itc=True, decim=3, n_jobs=1)


power_std.plot_topo(tmin=-0.2, tmax=0.7, baseline=(-0.2, 0), mode='logratio', title='Average power')
power_std.plot([82], tmin=-0.2, tmax=0.7, baseline=(-0.2, 0), mode='logratio')

power_dev.plot_topo(tmin=-0.2, tmax=0.7, baseline=(-0.2, 0), mode='logratio', title='Average power')
power_dev.plot([82], tmin=-0.2, tmax=0.7, baseline=(-0.2, 0), mode='logratio')


fig, axis = plt.subplots(1, 2, figsize=(7, 4))

power_dev.plot_topomap(tmin=0.4, tmax=0.5, fmin=7, fmax=12,
                   baseline=(-0.2, 0), mode='logratio', axes=axis[0],
                   title='Alpha', vmax=0.45, show=False)
power_dev.plot_topomap(tmin=0.4, tmax=0.4, fmin=13, fmax=25,
                   baseline=(-0.2, 0), mode='logratio', axes=axis[1],
                   title='Beta', vmax=0.45, show=False)
#power_dev.plot_topomap(ch_type='grad', tmin=0, tmax=0.2, fmin=8, fmax=12,
#                   baseline=(-0.2, 0), mode='logratio', axes=axis[0],
#                   title='Alpha', vmax=0.45, show=False)

mne.viz.tight_layout()
plt.show()

# for Inter-trial COHERENCE
# https://martinos.org/mne/stable/auto_tutorials/plot_sensors_time_frequency.html


#%% 
