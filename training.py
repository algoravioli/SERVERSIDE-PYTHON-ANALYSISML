# %%
import os

# import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from pyAudioAnalysis import ShortTermFeatures
import tensorflow as tf
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
import pickle as pkl
from AudioFunctions import *

# %%
systemDIR = os.getcwd()
filesINFolder = os.listdir(systemDIR + "/audio_tests")
print(filesINFolder)
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
    f"{systemDIR}/audio_tests/sghfebt2sf2w.wav", f"{systemDIR}/audio_tests/sf2post.CSV"
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
print(F.shape)
print(Q.shape)
print(V.shape)

Test_F = F4
Test_Q = Q4
Test_V = V4

FitArray = DeriveFitArray()

F = DoScalingStuff(F, FitArray)
Test_F = DoScalingStuff(Test_F, FitArray)


# %%
# Use F predict Q
regQ = MLPRegressor(
    solver="adam",
    activation="logistic",
    max_iter=500,
    alpha=1e-5,  # 1e-6,
    verbose=True,
    hidden_layer_sizes=(10, 10, 10),
    tol=0.0000000001,
    random_state=100,
)

# Use F predict V
regV = MLPRegressor(
    solver="adam",
    activation="logistic",
    max_iter=500,
    alpha=1e-5,  # 1e-6,
    verbose=True,
    hidden_layer_sizes=(10, 10, 10),
    tol=0.0000000001,
    random_state=100,
)

regQ.fit(F, Q)
regV.fit(F, V)

# Save Model using pickle
with open("model_saves/QModel.pkl", "wb") as f:
    pkl.dump(regQ, f)

with open("model_saves/VModel.pkl", "wb") as f:
    pkl.dump(regV, f)

# %%
y_hat = regQ.predict(Test_F)
plt.plot(Test_Q, label="Actual")
plt.plot(y_hat, "--", alpha=1, label="Predicted", linewidth=0.4)
plt.legend()
plt.figure(2)
y_hat_V = regV.predict(Test_F)
plt.plot(Test_V, label="Actual")
plt.plot(y_hat_V, "--", alpha=1, label="Predicted", linewidth=0.4)
plt.legend()

# %%
TotalVolume = np.cumsum(Test_V)
TotalVolume_hat = np.cumsum(y_hat_V)
plt.figure(3)
plt.plot(TotalVolume, label="Actual")
plt.plot(TotalVolume_hat, "--", alpha=1, label="Predicted", linewidth=1)
plt.legend()

# %%

Test_V_Q = GetQ(Test_V)
y_hat_V_Q = GetQ(y_hat_V)

plt.figure(4)
plt.plot(Test_V_Q, label="Actual")
plt.plot(y_hat_V_Q, "--", alpha=1, label="Predicted", linewidth=1)
plt.legend()

# %%
