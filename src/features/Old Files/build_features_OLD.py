# PART 5a & 5b

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
df = pd.read_pickle("../../data/interim/02_outlier_removed_chauvenet.pkl")

predictor_columns = list(df.columns[:6])

# Custom Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# ------------------------------------------------------------
# Dealing with missing values (imputation)
# ------------------------------------------------------------
for per_col in predictor_columns:
    df[per_col] = df[per_col].interpolate()

# ------------------------------------------------------------
# Calculating set duration
# ------------------------------------------------------------

# Calculating the average duration of a set. Single set.
single_duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]
single_duration.seconds

for per_set in df["set"].unique():
    
    start = df[df["set"] == per_set].index[0]
    end = df[df["set"] == per_set].index[-1]
    
    duration = end - start
    
    # This will make a new column "duration" and insert the
    # duration with respect to the current selected set.
    df.loc[(df["set"] == per_set), "duration"] = duration.seconds

mean_duration_df = df.groupby(["category"])["duration"].mean()

# ------------------------------------------------------------
# Butterworth lowpass filter (Need to learn this. The Function)
# ------------------------------------------------------------

    # Making a copy of original DF and inserting the lowpass 
    # values into it. Making an instance of the LowPassFilter.
df_lowpass = df.copy()
LowPass = LowPassFilter() 

    # Setting variables for the lowpass filter function
fs = 1000 / 200 # 5 readings per second (200ms)
# Trying 1 for now for the cutoff. 1 is the smoothest, and higher
# number means it close to the original data.
cutoff = 1.3

    # Butterworth Lowpass Filter. Single column first. -- "acce_y"
df_lowpass = LowPass.low_pass_filter(df_lowpass, "acce_y", fs, cutoff, order=5)

    # ------------------------------------------------------------
    # Making a subset top plot the of the original and the lowpass
    # ------------------------------------------------------------
subset = df_lowpass[df_lowpass["set"] == 45]

    # Styling
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acce_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acce_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
    
for per_col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, per_col, fs, cutoff, order=5)
    
    # This two lines will overwrite the original and delete 
    # the columns with "_lowpass" since those values has overwrite 
    # the original values.
    df_lowpass[per_col] = df_lowpass[per_col + "_lowpass"]
    del df_lowpass[per_col + "_lowpass"]

# ------------------------------------------------------------
# Principal component analysis (PCA) (Need to learn this. The Function)
# ------------------------------------------------------------

    # Making a copy of the DF and instance of the PCA.
df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

    # Determining the value of PCA and plot it.
pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("principal component number")
plt.ylabel("explained variance")
plt.show()

    # Applying the PCA to the dataframe.
df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

    # Making a subset again, see what category it is and plot it.
subset = df_pca[df_pca["set"] == 35]
subset[["pca_1", "pca_2", "pca_3"]].plot()

# ------------------------------------------------------------
# Sum of squares attributes (Need to learn this. The Function)
# ------------------------------------------------------------
df_squared = df_pca.copy()

    # To square a value in Python, use "**".
    # Single column. Will loop later.
acce_r = df_squared["acce_x"] ** 2 + df_squared["acce_y"] ** 2 + df_squared["acce_z"] ** 2
gyro_r = df_squared["gyro_x"] ** 2 + df_squared["gyro_y"] ** 2 + df_squared["gyro_z"] ** 2

df_squared["acce_r"] = np.sqrt(acce_r)
df_squared["gyro_r"] = np.sqrt(gyro_r)

    # Making a subset again, see what category it is and plot it.
subset = df_squared[df_squared["set"] == 22]
subset[["acce_r", "gyro_r"]].plot(subplots=True)

# ------------------------------------------------------------
# Temporal abstraction (Need to learn this. The Function)
# ------------------------------------------------------------

    # Making a copy of the DF and instance of the Numerical Abstraction.
df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

    # Adding the squared columns to the variable.
predictor_columns = predictor_columns + ["acce_r", "gyro_r"]

    # Like the cutoff variable, this is a trial and error.
size_window = int(1000 / 200)

for per_col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [per_col], size_window, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [per_col], size_window, "std")

    # Deleting the "duration" column since it's done its job for the Butterworth Lowpass Filter.
del df_temporal["duration"]
df_temporal.info()

    # making a subset base on the set. compute the values 
    # base on individual sets.
df_temporal_list = []
for per_set in df_temporal["set"].unique():
    # .copy() to avoid any warnings or error.
    subset = df_temporal[df_temporal["set"] == per_set].copy()
    for per_col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [per_col], size_window, "mean") # we could use the predictor_column here since its a list already.
        subset = NumAbs.abstract_numerical(subset, [per_col], size_window, "std")
        
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)

# ------------------------------------------------------------
# Frequency features (Need to learn this. The Function) (Fourier Transformation. T_T)
# ------------------------------------------------------------
    # Making a copy of the DF and instance of the Frequency Abstraction.
df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

    # defining the variable for frequency size and window size
frequency_size = int(1000 / 200) # ms --> 5 readings per seconds
size_window = int(2800 / 200) # ms --> 14 readings per window

    # Let's to apply the fourier transformation to one column 
    # before looping through all the columns so that we can 
    # have a better understanding to it first.
    # And just like the Temporal Abstraction, the "col" needs 
    # to be a list or we would get an error.
df_freq = FreqAbs.abstract_frequency(df_freq, ["acce_y"], size_window, frequency_size)

    # Visualizing the result using subset
subset = df_freq[df_freq["set"] == 51]
subset[["acce_y"]].plot()
subset[
    [
        "acce_y_max_freq",
        "acce_y_freq_weighted",
        "acce_y_pse",
        "acce_y_freq_1.429_Hz_ws_14",
        "acce_y_freq_2.5_Hz_ws_14",
    ]
].plot()

    # Now looping through all of the columns
df_freq_list = []
for per_set in df_freq["set"].unique():
    print(f"Applying fourier transformation to set {per_set}")
    # .copy() to avoid any warnings or error.
    subset = df_freq[df_freq["set"] == per_set].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, size_window, frequency_size)
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

# ------------------------------------------------------------
# Dealing with overlapping windows
# ------------------------------------------------------------
'''This will get a little bit long. Basically what I am are doing
here is droping all "NaN" values. This will result in lost of a 
lot of rows. After that, I am going to tackle the problem with
overlapping windows. 
iloc from pandas. one of the things that it can do --> has 
3 (command?) like this [X:X:X]. The first X is for the column,
then row, and finally the step count for each data. In this case,
it will select data every 2nd row from the last one. This will 
result to a half of the already reduced DF. But this proves over
and over again (Accordin to the Youtube video that I've watched, 
that most of the most of successful Language Models (LM) has 
used this method.) to have a better performance since it less prone
to overfitting (or in a more understandable way... This data has 
a lot of overlap, by doing a 50% reduction by selecting every 
second row, this will result to a much accurate data since the 
overlapping of data has been reduce significantly). Though, this
only works on a lot of data like this. Let's say you only have a
100 of data, doing 90% or 80% retaining would be much more
reasonable than 50% retaining - removing.'''
df_freq = df_freq.dropna()
df_freq = df_freq.iloc[::2]

# ------------------------------------------------------------
# Clustering - KMeans Clustering (Unsupervised Machine Learning) (Need to learn this and the function too.)
# ------------------------------------------------------------
    # Making a copy of the DF.
df_cluster = df_freq.copy()

    # Selecting the original 3 columns of the Accelerometer.
    # Preparation step to see the elbow and see what value
    # of K is the most appropriate. 
cluster_columns = ["acce_x", "acce_y", "acce_z"]
k_values = range(2, 10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.xlabel("k range(2 - 10)")
plt.ylabel("Sum of squared distances")
plt.show()

    # Now that we know the optimal valu of K, We can now do it
    # again, but this time only at 5. Then insert it into our
    # DF with a new column named "cluster"
kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

    # Plotting the cluster
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for per_cluster_value in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == per_cluster_value]
    ax.scatter(subset["acce_x"], subset["acce_y"], subset["acce_z"], label=per_cluster_value)
ax.set_xlabel("X-Axis")
ax.set_ylabel("Y-Axis")
ax.set_zlabel("Z-Axis")
plt.legend()
plt.show()

    # Plotting, label wise. To compare from the previous.
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for per_label in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == per_label]
    ax.scatter(subset["acce_x"], subset["acce_y"], subset["acce_z"], label=per_label)
ax.set_xlabel("X-Axis")
ax.set_ylabel("Y-Axis")
ax.set_zlabel("Z-Axis")
plt.legend()
plt.show()

# ------------------------------------------------------------
# Export dataset
# ------------------------------------------------------------
df_cluster.to_pickle("../../data/interim/03_data_features.pkl")