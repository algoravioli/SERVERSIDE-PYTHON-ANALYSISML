# %%
import warnings
import numpy as np
from scipy.signal import stft
from scipy.io import wavfile
import librosa
from scipy.special import erfcinv


def detect_bell_chime(filename):
    # bell parameters
    f0 = np.array([1325, 1525])  # f0 of bell
    f1 = np.array([3475, 3675])  # f1 of bell
    threshold = 0.4  # amplitude threshold for bell at f0 and f1

    # Extract audio signal and sampling frequency
    x, fs = librosa.load(filename, sr=None)

    # Get STFT of signal
    df = 25  # frequency bin width
    L = int(fs / df)  # window / NFFT size
    noverlap = int(L * 0.5)  # overlap size
    hannWin = 0.5 * (1 - np.cos(2 * np.pi * np.arange(L) / (L - 1)))  # hanning window

    f, t, s = stft(x, fs, window=hannWin, nperseg=L, noverlap=noverlap)

    # remove frequencies before 500 Hz and after 5 kHz
    s = s[(f >= 500) & (f <= 5000), :]
    f = f[(f >= 500) & (f <= 5000)]

    # rescale power of s at each time step to 0 and 1
    s_norm = np.abs(s) ** 2
    s_norm = s_norm / np.max(s_norm, axis=0)

    # convert to logical
    s_logical = s_norm >= threshold  # 0 if below threshold, 1 if above threshold

    # frequency vector flags to check for bell
    f0check = (f >= f0[0]) & (f <= f0[1])
    f1check = (f >= f1[0]) & (f <= f1[1])
    fotherscheck = ~(f0check | f1check)

    # return true for each time sample if above threshold at f0 and f1 range
    f0_flag = np.sum(s_logical[f0check, :], axis=0)  # check for power within f0
    f0_flag = f0_flag >= 1  # logical true if power within f0
    f1_flag = np.sum(s_logical[f1check, :], axis=0)  # repeat for f1
    f1_flag = f1_flag >= 1  # repeat for f1

    # return true for each time sample if above threshold beyond f0 and f1 range
    fothers_flag = np.sum(
        s_logical[fotherscheck, :], axis=0
    )  # check for power outside f0 and f1
    fothers_flag = fothers_flag >= 1  # true if power outside f0 and f1

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

    # check for power at low frequencies
    s_logical_low = s_norm >= 0.16
    f_flag_low = np.sum(s_logical_low[f < 1000, :], axis=0)
    f_flag_low = f_flag_low >= 1

    # flag as bell if f0f1_flag = true and fothers0_flag = 0 at sampling time
    bell_flag = f0_flag & f1_flag & ~fothers_flag
    bell_flag_damped = f1_flag_damped & ~fothers_flag_damped
    bell_flag_low = f_flag_low

    # Discard single sample occurrences for damped hits
    bell_flag_damped_lead = np.concatenate(
        [bell_flag_damped[1:], [0]]
    )  # damped trigger vector, led by 1 sample
    bell_flag_damped_lag = np.concatenate(
        [[0], bell_flag_damped[:-1]]
    )  # damped trigger vector, lag by 1 sample
    bell_flag_damped = bell_flag_damped & (bell_flag_damped_lead | bell_flag_damped_lag)

    # locate times where bell is supposedly identified
    bell_flag_all = (bell_flag | bell_flag_damped) & ~bell_flag_low
    tBell = t[bell_flag_all]

    # Segregate into start and end bell times
    tBellStart = tBell[tBell < t[-1] / 2]
    tBellEnd = tBell[tBell >= t[-1] / 2]

    # remove outliers
    if len(tBellStart) > 10:
        tBellStart, _ = remove_outliers(tBellStart)

    if len(tBellEnd) > 10:
        tBellEnd, _ = remove_outliers(tBellEnd)

    # Take last sample of BellStart and first sample of BellEnd times
    tBellStart = tBellStart[-1]
    tBellEnd = tBellEnd[0]

    # checks and warnings if start and end times make sense
    if tBell.size == 0:
        warnings.warn("No bells detected")
        tBellStart = 1 / fs
        tBellEnd = t[-1]

    if tBellEnd - tBellStart < 5:
        if tBellStart > 0.5 * t[-1]:
            tBellStart = 0
            warnings.warn("No start bell detected.")
        if tBellEnd < 0.5 * t[-1]:
            tBellEnd = t[-1]
            warnings.warn("No end bell detected")

    return tBellStart, tBellEnd


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


path_to_audio = "bell_check.wav"
tBellStart, tBellEnd = detect_bell_chime(path_to_audio)
# %%
