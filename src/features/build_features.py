# PART 5a & 5b

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

# ----------------------------------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------------------------------
df = pd.read_pickle("../../data/interim/02_outlier_removed_chauvenet.pkl")

    # Selecting and inserting the first 6 column names of the DF as list into the variable.
predictor_columns = list(df.columns[:6])

# Custom Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# ----------------------------------------------------------------------------------------------------
# Dealing with missing values (imputation)
# Insert a data between the gap of each data with respect to each data so that there would be no
# broken lines in graph and the lines is smooth - not sharp.
# ----------------------------------------------------------------------------------------------------
for per_col in predictor_columns:
    df[per_col] = df[per_col].interpolate()

# ----------------------------------------------------------------------------------------------------
# Calculating set duration
# For loop that loops between the set to accurately calculate the total duration of each set.
# ----------------------------------------------------------------------------------------------------
for per_set in df["set"].unique():
    
    start = df[df["set"] == per_set].index[0]
    end = df[df["set"] == per_set].index[-1]
    
    duration = end - start
    
    # This will make a new column "duration" and insert the
    # duration with respect to the current selected set.
    df.loc[(df["set"] == per_set), "duration"] = duration.seconds

# ----------------------------------------------------------------------------------------------------
# Butterworth Lowpass Filter (Need to learn this and the function)
# ----------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------
    # Making a copy of the original DF that will use to insert the values that the Butterworth Lowpass
    # Filter will calculate.
    # Making an instance of the Class LowPassFilter.
    # ---------------------------------------------------------------------------------------------------- 
df_lowpass = df.copy()
LowPass = LowPassFilter()

    # ----------------------------------------------------------------------------------------------------
    # Setting variables that will be use for the lowpass filter function.
    # cutoff --> Trying 1 at first. Higher number will make the  graph more smooth, and lower number leads
    # to a graph that is more like the original graph.
    # ----------------------------------------------------------------------------------------------------
fs = 1000 / 200
cutoff = 1.3
    
    # ----------------------------------------------------------------------------------------------------
    # This For Loop loop through each of the 6 columns and apply the low_pass_filter function and adds new
    # column named after the current column + "_lowpass" with its repective computed value.
    # Overall, this cleans-up the data a little bit. Hence, the lowpass filter
    # ----------------------------------------------------------------------------------------------------
for per_col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, per_col, fs, cutoff, order=5)
    
    # ----------------------------------------------------------------------------------------------------
    # This subset serves as a comparison between the original and the lowpass filter values. 
    # Then plots it after to see the difference.
    # ----------------------------------------------------------------------------------------------------
subset = df_lowpass[df_lowpass["set"] == 35]
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(df_lowpass["gyro_y"].reset_index(drop=True), label="raw data")
ax[1].plot(df_lowpass["gyro_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

    # ----------------------------------------------------------------------------------------------------
    # This for loop will overwrite the original values and delete the new created column made by the
    # function (low_pass_filter) since those values is now the same to the original.
    # ----------------------------------------------------------------------------------------------------
for per_col in predictor_columns:
    df_lowpass[per_col] = df_lowpass[per_col + "_lowpass"]
    del df_lowpass[per_col + "_lowpass"]

# ----------------------------------------------------------------------------------------------------
# Principal component analysis (PCA) (Need to learn this. The Function)
# ----------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------
    # Making a copy of the lowpass DF that will use to insert the values that the PCA will compute
    # Making an instance of the Class PrincipalComponentAnalysis.
    # ----------------------------------------------------------------------------------------------------
df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

    # ----------------------------------------------------------------------------------------------------
    # Using the "determine_pc_explained_variance" will give as an array of values based on the df_pca
    # and predictor_columns. Then Plot it afterwards to determine the the "number component" that I will be
    # selecting. It's called the "Elbow Technique" since it looks like an elbow and the value that will be
    # chosen is the value right before the line of the graph is almost stable and after when the line of the
    # graph is unstable (or changing rapidly on each step of the value.).. Basically, choose the value
    # right before the "diminishing of return".
    # ----------------------------------------------------------------------------------------------------
pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("principal component number")
plt.ylabel("explained variance")
plt.show()

    # ----------------------------------------------------------------------------------------------------
    # Using the "apply_pca", now with the "number component" that I've selected based on the plot. This
    # will result to additional 3 columns with the result based on the column selected (predictor_columns)
    # on the DF provided (df_pca)
    # ----------------------------------------------------------------------------------------------------
df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

# ----------------------------------------------------------------------------------------------------
# Sum of squares attributes (Need to learn this. The Function)
# ----------------------------------------------------------------------------------------------------

    # Making a copy of the pca DF that will be use to insert the squre-root of the accelerometer and 
    # gyroscope data.
df_squared = df_pca.copy()

    # ----------------------------------------------------------------------------------------------------
    # To square a value in Python, use "**".
    # What this 4 lines of code does is, the first 2 lines of codes square each column with respect to the
    # device (Acceleroment or Gyroscope) and adds them together. While the last 2 lines of code makes a new
    # column with "_r" in it to correspond to the annotation "squared", and the values of each row is 
    # squared using numpy's sqrt method.
    # ----------------------------------------------------------------------------------------------------
acce_r = df_squared["acce_x"] ** 2 + df_squared["acce_y"] ** 2 + df_squared["acce_z"] ** 2
gyro_r = df_squared["gyro_x"] ** 2 + df_squared["gyro_y"] ** 2 + df_squared["gyro_z"] ** 2

df_squared["acce_r"] = np.sqrt(acce_r)
df_squared["gyro_r"] = np.sqrt(gyro_r)

# ----------------------------------------------------------------------------------------------------
# Temporal abstraction (Need to learn this. The Function)
# ----------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------
    # Making a copy of the squared DF that will use to insert the values that the Temporal Abstraction will
    # compute.
    # Making an instance of the Class NumericalAbstraction.
    # ---------------------------------------------------------------------------------------------------- 
df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

    # ----------------------------------------------------------------------------------------------------
    # Adding the squared columns to the predictor_column.
    # Like the cutoff variable, this is a trial and error. Though this is already a good start since this
    # value corresponds to the sampling rate (200ms) that was initialy used. This might not be changed.
    # Deleting the column "duration" since it has done its part and checking via .info()
    # ----------------------------------------------------------------------------------------------------
predictor_columns = predictor_columns + ["acce_r", "gyro_r"]
size_window = int(1000 / 200)

del df_temporal["duration"]
df_temporal.info()

    # ----------------------------------------------------------------------------------------------------
    # First, making an empty list that will hold the subset values.
    # There's 2 For Loops here. The outer loop loops through the DF in respect to the column "set" and insert
    # it into the subset DF. The 2nd loop loops through the "predictor_columns" and then applies the 
    # "abstract_numerical" method to the subset DF in respect to the current col of the loop (per_col),
    # size_window = 5, and the method - mean and std.
    # Then, appending the final value of subset to the empty list. Finally, concatinating the list using
    # pandas and inserting it back to the "df_temporal", overwritting the original "df_temporal".
    # ----------------------------------------------------------------------------------------------------
df_temporal_list = []
for per_set in df_temporal["set"].unique():
    # .copy() to avoid any warnings or error.
    subset = df_temporal[df_temporal["set"] == per_set].copy()
    for per_col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [per_col], size_window, "mean") # we could use the predictor_column here since its a list already.
        subset = NumAbs.abstract_numerical(subset, [per_col], size_window, "std")
        
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)

# ----------------------------------------------------------------------------------------------------
# Frequency features (Need to learn this. The Function) (Fourier Transformation. T_T)
# ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    # Making a copy of the temporal DF that will use to insert the values that the Fourier Transformation
    # will compute.
    # Making an instance of the Class FourierTransformation.
    # ---------------------------------------------------------------------------------------------------- 
df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

    # ----------------------------------------------------------------------------------------------------
    # Setting up the frequency size, which is 200ms that leads to 5 readings per second.
    # As for the window size, it is based on how long it takes a participant does each "label" instance
    # on average, which is 2.8 seconds. This leads to 14 readings per window / instance.
    # ----------------------------------------------------------------------------------------------------
frequency_size = int(1000 / 200)
size_window = int(2800 / 200)

    # ----------------------------------------------------------------------------------------------------
    # This loop has a lot of similiraties to the temporal loop. I also did here the: A.) An empty list.
    # B.) looping in respect to the "set" column and inserting it to the subset. C.) The "abstract_frequency" 
    # method also requires a list for the columns. Though in here, I use the "predictor_columns". Eliminating 
    # the need to use an inner For Loop. D.) Append the subset to the empty list. E.) Finally, concatinating
    # the list and inserting it back to the "df_freq", overwritting the original values of it.
    # The difference are: A.) There's a print statement which prints the current "set" column it is, to see
    # the progress. B.) The inserted DF to the subset DF has .reset_index since the method "abstract_frequency"
    # requires an integer index. C.) The inserted concat DF to the "df_freq" is set back to the
    # index = "epoch (ms)" column so that it is easy to read.
    # ----------------------------------------------------------------------------------------------------
df_freq_list = []
for per_set in df_freq["set"].unique():
    print(f"Applying fourier transformation to set {per_set}")
    subset = df_freq[df_freq["set"] == per_set].reset_index(drop=True).copy() # Also .copy() to avoid any warning / error.
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, size_window, frequency_size)
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

# ----------------------------------------------------------------------------------------------------
# Dealing with overlapping windows
# ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    # This will get a little bit long. Basically, what I am doing here is dropping all "NaN" values. This
    # will result in a lost of a  lot of rows. After that, I am going to solve the problem with 
    # overlapping windows.
    # .iloc[] from Pandas, one of the things that it can do --> has 3 (commands?) like this [X:X:X]. The
    # first X is for the column, then row, and finally the step count for each data selection. In this case,
    # it will select the 1st data and start selecting data every 2nd row from the last one. This will 
    # result to a half of the already reduced DF. But this proves (according to the Youtube video that I'm
    # basing this to, that most of the most successful Language Model (LM) has used this method.) over and
    # over again to have a better performance since it's less prone to overfitting (or in a more understandable
    # way... This data has a lot of overlap, by doing 50% reduction by selecting every 2nd row, this will
    # result to a much accurate data since the overlapping of data has been reduced significantly.) Though,
    # this only works on a lot of data like this one. Let's say you only have a 100 rows of data, doing 90%
    # of 80% retaining would be much more reasonable than 50% retaining - removing...
    # ----------------------------------------------------------------------------------------------------
df_freq = df_freq.dropna()
df_freq = df_freq.iloc[::2]

# ----------------------------------------------------------------------------------------------------
# Clustering - KMeans Clustering (Unsupervised Machine Learning) (Need to learn this and the function too.)
# This part of the code is only to see the prediction of KMeans. An Unsupervised ML. The Part - 6 is the
# part that tackles the predictive models. As we can see, this code only tackles the columns of the 
# Accelerometer. Afterwards, compare the the values of KMean to the column "label" using 3d Plot.
# ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    # Making a copy of the frequency DF that will be use to insert the values the KMean computes.
    # Making a list that holds the column names for the Accelerometer
    # A variable with a range of 2 to 10, and an empty list.
    # ----------------------------------------------------------------------------------------------------
df_cluster = df_freq.copy()

cluster_columns = ["acce_x", "acce_y", "acce_z"]
k_values = range(2, 10)
inertias = []

    # ----------------------------------------------------------------------------------------------------
    # This loop loops through the range of the "k_values". Insert the selected columns of Accelerometer to
    # the subset. Inserting the "KMean" with its respective set of values to "kmeans" variable. Then, 
    # using the "fit_predict" function of the "kmean" to the subset. Finally, appending those values to
    # the initially empty list so that the last value of subset won't go to waste.
    # TLDR: This loop is computes for the values related to the "k_values". Then, we Plot it to see the
    # "Elbow Technique" that we also used back from PCA. Determine the "number component" value.
    # ----------------------------------------------------------------------------------------------------
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

    # ----------------------------------------------------------------------------------------------------
    # Now that we know the "number component" value, use the method again. But this time, no more For Loop
    # since that is only for determining the "number component". The computer values this time will
    # be inserted to the new column "cluster" from the "df_cluster"
    # ----------------------------------------------------------------------------------------------------
kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

    # ----------------------------------------------------------------------------------------------------
    # Finally, Plotting the data. Used "3d" since we're dealing with 3-dimensional values here.
    # The first plot is for the "cluster" column values and second plot is for the "label" column values.
    # And then compare the plot results.
    # ----------------------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------------------
# Export dataset
# ----------------------------------------------------------------------------------------------------
df_cluster.to_pickle("../../data/interim/03_data_features.pkl")