# %%
import os
import sys

# import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import pickle as pkl
from AudioFunctions import *
from BellFunctions import detect_bell_and_return_timings

# Example i.e How to run this script
# python inference.py --audio_file_path "./audio_tests/bell_check.wav" --output_image_path "output_img.png" --save_cut_audio_path "saved_audio/audio1.wav" --save_csv_path "output.csv"

audio_file_path = "audio.wav"
output_image_path = "image.png"
save_cut_audio_path = "audio_cut.wav"
save_csv_path = "features.csv"

BellStart, BellEnd = detect_bell_and_return_timings(audio_file_path)
# print(BellStart, BellEnd)

# Cut audio file at BellStart and BellEnd
pad = 0.5
audioFile, Fs = librosa.load(audio_file_path, sr=None)
if BellStart == None:
    audioFile = audioFile
else:
    BellStartIdx = int((BellStart + pad) * Fs)
    BellEndIdx = int((BellEnd - pad) * Fs)
    audioFile = audioFile[BellStartIdx:BellEndIdx]

# Extract features
F, f_names = GiveMeFreqInfo(audioFile, Fs)
FitArray = DeriveFitArray()
F_scaled = DoScalingStuff(F, FitArray)

# Load the model
VModel = pkl.load(open("model_saves/VModel.pkl", "rb"))

# Run inference on the audio file
y_hat_V = VModel.predict(F_scaled)

# Total cumulative volume of predicted V
TotalVolume_hat = np.cumsum(y_hat_V)

# Return Q of predicted V
y_hat_V_Q = GetQ(y_hat_V)

# Write y_hat_V_Q to csv
df = pd.DataFrame(y_hat_V_Q)
df.to_csv(save_csv_path)

# Save results to output_image_path
plt.plot(y_hat_V_Q)
plt.savefig(output_image_path)

# Save cut audio file
# Normalise cut audio file first
audioFile = audioFile / np.max(np.abs(audioFile))
sf.write(save_cut_audio_path, audioFile, Fs, subtype="PCM_24")
