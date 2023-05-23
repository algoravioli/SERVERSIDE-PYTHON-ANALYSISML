#%%
import os
import warnings
import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from timeit import default_timer as timer
from sklearn import preprocessing
from scipy.signal import find_peaks, peak_prominences
from scipy.special import erfcinv
from pyAudioAnalysis import ShortTermFeatures

from AudioFunctions import *
from BellFunctions_v2 import detect_bell_and_return_timings
from BellFunctions_v2 import JohnnySTFT
from BellFunctions_v2 import fft_at_peaks
from BellFunctions_v2 import extract_bell_times

#%%
def remove_outliers(lst):
    """
    Takes a list of values, and detects and outputs a list with removed outliers based on 3 x scaled MAD.
    The outliers are listed in the second output.

    Parameters:
    lst (numpy.ndarray): a 1D numpy array of values.

    Returns:
    newList (numpy.ndarray): a 1D numpy array of values with outliers removed.
    outliers (numpy.ndarray): a 1D numpy array of outlier values that were removed.
    """

    # Compute Median Absolute Deviation (MAD)
    medList = np.median(lst)
    MAD = np.median(np.abs(lst - np.median(lst)))

    # Compute scaled MAD
    c = -1 / (np.sqrt(2) * erfcinv(3 / 2))
    scaledMAD = c * MAD
    lowerLimit = medList - 3 * scaledMAD
    upperLimit = medList + 3 * scaledMAD

    outliers = lst[(lst > upperLimit) | (lst < lowerLimit)]
    newList = lst[(lst <= upperLimit) & (lst >= lowerLimit)]

    return newList, outliers

#peak clipping functions
def clip_data(unclipped, high_clip, low_clip):
    ''' Clip unclipped between high_clip and low_clip. 
    unclipped contains a single column of unclipped data.'''
    
    # convert to np.array to access the np.where method
    np_unclipped = np.array(unclipped)
    # clip data above HIGH_CLIP or below LOW_CLIP
    cond_high_clip = (np_unclipped > HIGH_CLIP) | (np_unclipped < LOW_CLIP)
    np_clipped = np.where(cond_high_clip, np.nan, np_unclipped)
    return np_clipped.tolist()


def create_sample_data():
    ''' Create sine wave, amplitude +/-2 with random spikes. '''
    x = np.linspace(0, 2*np.pi, 1000)
    y = 2 * np.sin(x)
    df = pd.DataFrame(list(zip(x,y)), columns=['x', 'y'])
    df['rand'] = np.random.random_sample(len(x),)
    # create random positive and negative spikes
    cond_spike_high = (df['rand'] > RAND_HIGH)
    df['spike_high'] = np.where(cond_spike_high, SPIKE, 0)
    cond_spike_low = (df['rand'] < RAND_LOW)
    df['spike_low'] = np.where(cond_spike_low, -SPIKE, 0)
    df['y_spikey'] = df['y'] + df['spike_high'] + df['spike_low']
    return df

def ewma_fb(df_column, span):
    ''' Apply forwards, backwards exponential weighted moving average (EWMA) to df_column. '''
    # Forwards EWMA.
    fwd = pd.Series.ewm(df_column, span=span).mean()
    # Backwards EWMA.
    bwd = pd.Series.ewm(df_column[::-1],span=10).mean()
    # Add and take the mean of the forwards and backwards EWMA.
    stacked_ewma = np.vstack(( fwd, bwd[::-1] ))
    fb_ewma = np.mean(stacked_ewma, axis=0)
    return fb_ewma
    
    
def remove_outliers(spikey, fbewma, delta):
    ''' Remove data from df_spikey that is > delta from fbewma. '''
    np_spikey = np.array(spikey)
    np_fbewma = np.array(fbewma)
    cond_delta = (np.abs(np_spikey-np_fbewma) > delta)
    np_remove_outliers = np.where(cond_delta, np.nan, np_spikey)
    return np_remove_outliers

    
def main():
    df = create_sample_data()

    df['y_clipped'] = clip_data(df['y_spikey'].tolist(), HIGH_CLIP, LOW_CLIP)
    df['y_ewma_fb'] = ewma_fb(df['y_clipped'], SPAN)
    df['y_remove_outliers'] = remove_outliers(df['y_clipped'].tolist(), df['y_ewma_fb'].tolist(), DELTA)
    df['y_interpolated'] = df['y_remove_outliers'].interpolate()
    
    ax = df.plot(x='x', y='y_spikey', color='blue', alpha=0.5)
    ax2 = df.plot(x='x', y='y_interpolated', color='black', ax=ax)
    

path_to_audio_file = "/Users/ivantan/Documents/GitHub/SERVERSIDE-PYTHON-ANALYSISML/audio_tests/sghTLT5t2.wav"
filename = path_to_audio_file
print(filename)

 #%%
#bell parameters
f0 = np.array([1325, 1525])  # f0 of bell
f1 = np.array([3475, 3675])  # f1 of bell
threshold = 0.4  # amplitude threshold for bell at f0 and f1


# Extract audio signal and sampling frequency
x, fs = librosa.load(filename, sr=None)
filelength = x.shape[0]/fs
print("File is ",filelength, "s long")

# %%
# define variables for STFT of signal
df = 25  # frequency bin width
L = int(fs / df)  # window / NFFT size
noverlap = int(L * 0.5)  # overlap size
hannWin = 0.5 * (1 - np.cos(2 * np.pi * np.arange(L) / (L - 1)))  # hanning window

#%%
## ivan find peaks here
x_rescale = x.reshape(-1, 1)
x_rescale = preprocessing.MinMaxScaler().fit_transform(x_rescale)
x_rescale = x_rescale[:,0]

##clip x_rescale
logging.basicConfig(datefmt='%H:%M:%S',
                    stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(message)s')

# Distance away from the FBEWMA that data should be removed.
DELTA = 0.1

# clip data above this value:
HIGH_CLIP = 2.1

# clip data below this value:
LOW_CLIP = -2.1

# random values above this trigger a spike:
RAND_HIGH = 0.95

# random values below this trigger a negative spike:
RAND_LOW = 0.001

# How many samples to run the FBEWMA over.
SPAN = 10

# spike amplitude
SPIKE = 2

x_clipped = clip_data(x_rescale, HIGH_CLIP, LOW_CLIP)
x_fbewma = ewma_fb(x_clipped, SPAN)
x_trimmed = remove_outliers(x_rescale, x_fbewma, DELTA)
x_rescale = x_trimmed.interpolate()


peaklocs, _ = find_peaks(x_rescale,height=(0.2, 0.8),distance=250, prominence= (0.1,0.9))
ypeaks = x_rescale[peaklocs]
#plt.plot(x)
plt.plot(x_rescale)
plt.plot(peaklocs, ypeaks, "x")

#%%
# Compute windowed FFT of peak locations
starttime = timer()
f, t, s = fft_at_peaks(x,fs,peaklocs)
#%%
s = s[(f >= 500) & (f <= 5000), :]
f = f[(f >= 500) & (f <= 5000)]

# %%
# rescale power of s at each time step to 0 and 1
s_norm = np.abs(s) ** 2
s_norm = s_norm / np.max(s_norm, axis=0)

# %%
# convert to logical
s_logical = s_norm >= threshold  # 0 if below threshold, 1 if above threshold

# %%
# frequency vector flags to check for bell
f0check = (f >= f0[0]) & (f <= f0[1])
f1check = (f >= f1[0]) & (f <= f1[1])
fotherscheck = ~(f0check | f1check)
# %%
# return true for each time sample if above threshold at f0 and f1 range
f0_flag = np.sum(s_logical[f0check, :], axis=0)  # check for power within f0
f0_flag = f0_flag >= 1  # logical true if power within f0
f1_flag = np.sum(s_logical[f1check, :], axis=0)  # repeat for f1
f1_flag = f1_flag >= 1  # repeat for f1

# %%
# return true for each time sample if above threshold beyond f0 and f1 range
fothers_flag = np.sum(
    s_logical[fotherscheck, :], axis=0
)  # check for power outside f0 and f1
fothers_flag = fothers_flag >= 1  # true if power outside f0 and f1
# %%
# check for damped hits
s_logical_damped = s_norm >= 0.9  # threshold at f1
s_logical_damped2 = s_norm >= 0.45  # threshold for outside f0 and f1
f1_flag_damped = np.sum(
    s_logical_damped[f1check, :], axis=0
)  # check for power at f1
f1_flag_damped = f1_flag_damped >= 1  # true if power at f1
fothers_flag_damped = np.sum(
    s_logical_damped2[fotherscheck, :], axis=0
)  # check for power outside f0 and f1
fothers_flag_damped = fothers_flag_damped >= 1  # true if power outside f0 and f1
# %%
# check for power at low frequencies
s_logical_low = s_norm >= 0.16
f_flag_low = np.sum(s_logical_low[f < 1000, :], axis=0)
f_flag_low = f_flag_low >= 1
# %%
# flag as bell if f0f1_flag = true and fothers0_flag = 0 at sampling time
bell_flag = f0_flag & f1_flag & ~fothers_flag
bell_flag_damped = f1_flag_damped & ~fothers_flag_damped
bell_flag_low = f_flag_low
# %%
# Discard single sample occurrences for damped hits
bell_flag_damped_lead = np.concatenate(
    [bell_flag_damped[1:], [0]]
)  # damped trigger vector, led by 1 sample
bell_flag_damped_lag = np.concatenate(
    [[0], bell_flag_damped[:-1]]
)  # damped trigger vector, lag by 1 sample
bell_flag_damped = bell_flag_damped & (bell_flag_damped_lead | bell_flag_damped_lag)
# %%
# locate times where bell is supposedly identified
bell_flag_all = (bell_flag | bell_flag_damped) & ~bell_flag_low
# get index of t where bell_flag_all is true
tBellIndexes = np.where(bell_flag_all)[0]
tBell = t[tBellIndexes]
if len(tBell) == 0:
    # Set tBellStart to 0 and tBellEnd to last
    tBellStart = 0
    # tBellEnd = t[-1] previous value to set end time if error
    tBellEnd = len(x)/fs-0.5
else:
    collectTimes = []
    currentTime = tBell[0]
    for i in range(len(tBell)):
        if i == 0:
            continue
        else:
            previousTime = currentTime
            currentTime = tBell[i]
            if currentTime - previousTime < 10:
                continue
            else:
                collectTimes.append([previousTime, currentTime])
    if len(collectTimes) == 0:
        tBellStart = 0
        tBellEnd = len(x)/fs-0.5

    else:
        tBellStart = collectTimes[0][0]
        tBellEnd = collectTimes[0][-1]
# %%
print(tBellStart, tBellEnd)
# %%
