#%%
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from timeit import default_timer as timer
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

path_to_audio_file = "C:\\Users\\Ivan\\Documents\\GitHub\\SERVERSIDE-PYTHON-ANALYSISML\\audio_tests\\sghTLT1t1.wav"
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
# ivan find peaks here

peaklocs, _ = find_peaks(x,height=0.06)
ypeaks = x[peaklocs]
plt.plot(x)
plt.plot(peaklocs, ypeaks, "x")

#%%
# Compute windowed FFT of peak locations
starttime = timer()
f, t, s = fft_at_peaks(x,fs,peaklocs)
[tBellStart,tBellEnd] = extract_bell_times(s,f,t,f0,f1,threshold)



#%%
#perform STFT on remaining signal

if tBellStart == 0 or tBellEnd >= len(x)/fs-0.5:
    f, t, s = JohnnySTFT(x, window=hannWin, noverlap=noverlap, Fs=fs, compare=peaklocs)
    [tBellStart,tBellEnd] = extract_bell_times(s,f,t,f0,f1,threshold)

endtime = timer()
print("The FFT process took", endtime - starttime, "s.")

# librosa stft
# %%
# remove frequencies before 500 Hz and after 5 kHz
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
    tBellEnd = t[-1]
else:
    collectTimes = []
    currentTime = tBell[0]
    for i in range(len(tBell)):
        if i == 0:
            continue
        else:
            previousTime = currentTime
            currentTime = tBell[i]
            if currentTime - previousTime < 1:
                continue
            else:
                collectTimes.append([previousTime, currentTime])
    tBellStart = collectTimes[0][0]
    tBellEnd = collectTimes[0][-1]

print("_____________________________________________")
print(tBellStart, tBellEnd)

# %%
# Segregate into start and end bell times
# tBellStart = tBell[tBell < t[-1] / 2]
# tBellEnd = tBell[tBell >= t[-1] / 2]

# # remove outliers
# if len(tBellStart) > 10:
#     tBellStart, _ = remove_outliers(tBellStart)

# if len(tBellEnd) > 10:
#     tBellEnd, _ = remove_outliers(tBellEnd)

# # Take last sample of BellStart and first sample of BellEnd times
# tBellStart = tBellStart[-1]
# tBellEnd = tBellEnd[0]

# # checks and warnings if start and end times make sense
# if tBell.size == 0:
#     warnings.warn("No bells detected")
#     tBellStart = 1 / fs
#     tBellEnd = t[-1]

# if tBellEnd - tBellStart < 5:
#     if tBellStart > 0.5 * t[-1]:
#         tBellStart = 0
#         warnings.warn("No start bell detected.")
#     if tBellEnd < 0.5 * t[-1]:
#         tBellEnd = t[-1]
#         warnings.warn("No end bell detected")
#%%
print(tBellStart, tBellEnd)

# %%