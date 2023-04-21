import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from pyAudioAnalysis import ShortTermFeatures
import tensorflow as tf
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler


# Load an audio file
def GiveMeData(path_to_audio, path_to_csv):
    audioFile = path_to_audio
    x, Fs = librosa.load(audioFile, sr=None)
    superFlow_df = pd.read_csv(path_to_csv, error_bad_lines=False)
    # return superFlow_df column names
    voidedVolume = superFlow_df["Vmic"].to_numpy()
    voidedVolume_diff = np.diff(voidedVolume[::2])
    Q = superFlow_df["Qura"].to_numpy()[::2]
    length_Q = len(Q)
    F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 8820, 8820, deltas=False)
    length_F = F.shape[1]
    length_V = len(voidedVolume_diff)
    if length_V < length_Q and length_Q < length_F:
        F = F[:, 0:length_V]
        Q_trim = Q[0:length_V]
        voidedVolume_diff_trim = voidedVolume_diff[0:length_V]
        Q_column = np.reshape(Q_trim, (length_V, 1))
        V_column = np.reshape(voidedVolume_diff_trim, (length_V, 1))
        F_transposed = np.transpose(F)

    elif length_V == length_Q and length_Q < length_F:
        F = F[:, 0:length_Q]
        Q_trim = Q
        voidedVolume_diff_trim = voidedVolume_diff[0:length_Q]
        Q_column = np.reshape(Q_trim, (length_Q, 1))
        V_column = np.reshape(voidedVolume_diff_trim, (length_Q, 1))
        F_transposed = np.transpose(F)
    else:
        Q_trim = Q[0:length_F]
        voidedVolume_diff_trim = voidedVolume_diff[0:length_F]
        F_transposed = np.transpose(F)
        Q_column = np.reshape(Q_trim, (length_F, 1))
        V_column = np.reshape(voidedVolume_diff_trim, (length_F, 1))
    # print(F_transposed.shape)
    # print(Q_column.shape)
    # print(V_column.shape)
    return F_transposed, f_names, Q_column, V_column


def GiveMeFreqInfo(x, Fs):
    F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 8820, 8820, deltas=False)
    F_transposed = np.transpose(F)
    return F_transposed, f_names


def DoScalingStuff(array, fitarray):
    scaler = StandardScaler()
    scaler.fit(fitarray)
    scaled_array = scaler.transform(array)
    return scaled_array


def DeriveFitArray():
    systemDIR = os.getcwd()
    filesINFolder = os.listdir(systemDIR + "/audio_tests")
    # print(filesINFolder)
    # %%
    F1, f_names_1, Q1, V1 = GiveMeData(
        f"{systemDIR}/audio_tests/sghfebt2sf1w.wav",
        f"{systemDIR}/audio_tests/sf1post.CSV",
    )
    F2, f_names_2, Q2, V2 = GiveMeData(
        f"{systemDIR}/audio_tests/sghfebt2sf1wr2.wav",
        f"{systemDIR}/audio_tests/sf1post.CSV",
    )  # test
    F3, f_names_3, Q3, V3 = GiveMeData(
        f"{systemDIR}/audio_tests/sghfebt2sf2w.wav",
        f"{systemDIR}/audio_tests/sf2post.CSV",
    )
    F4, f_names_4, Q4, V4 = GiveMeData(
        f"{systemDIR}/audio_tests/sghfebt2sf2wr2.wav",
        f"{systemDIR}/audio_tests/sf2post.CSV",
    )
    # %%
    # V_stack F1, F3, F4
    F = np.vstack((F1, F3, F2))
    Q = np.vstack((Q1, Q3, Q2))
    V = np.vstack((V1, V3, V2))

    return F


def GetQ(array):
    outputArray = np.array([])
    for i in range(len(array) - 5):
        # look at next 10 samples
        current_array = array[i : i + 4]
        current_Q = np.sum(current_array)
        outputArray = np.append(outputArray, current_Q)
        # get every 10 number from array
    outputArray = outputArray[::5]

    return outputArray
