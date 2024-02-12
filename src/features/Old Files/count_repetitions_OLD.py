# PART - 7

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None

    # General Plot settings:
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# ----------------------------------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------------------------------

    # Loading the data from DF and removing all data under "rest"
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")
df = df[df["label"] != "rest"]

    # Computing the squared again since there might some useful information that we can get when
    # counting the repetitions.
acce_r = df["acce_x"] ** 2 + df["acce_y"] ** 2 + df["acce_z"] ** 2
gyro_r = df["gyro_x"] ** 2 + df["gyro_y"] ** 2 + df["gyro_z"] ** 2
df["acce_r"] = np.sqrt(acce_r)
df["gyro_r"] = np.sqrt(gyro_r)

# ----------------------------------------------------------------------------------------------------
# Split data
# ----------------------------------------------------------------------------------------------------

    # Splitting the data --> Making 5 DF, each containing a single exercise label.
df_bench = df[df["label"] == "bench"]
df_squat = df[df["label"] == "squat"]
df_row = df[df["label"] == "row"]
df_ohp = df[df["label"] == "ohp"]
df_dead = df[df["label"] == "dead"]

# ----------------------------------------------------------------------------------------------------
# Visualize data to identify patterns
# ----------------------------------------------------------------------------------------------------

    # Selecting single "set" first.
    # df for plotting.
df_plot = df_bench

df_bench[df_bench["set"] == df_bench["set"].unique()[0]]["acce_x"].plot()
df_bench[df_bench["set"] == df_bench["set"].unique()[0]]["acce_y"].plot()
df_bench[df_bench["set"] == df_bench["set"].unique()[0]]["acce_z"].plot()
df_bench[df_bench["set"] == df_bench["set"].unique()[0]]["acce_r"].plot()

df_bench[df_bench["set"] == df_bench["set"].unique()[0]]["gyro_x"].plot()
df_bench[df_bench["set"] == df_bench["set"].unique()[0]]["gyro_y"].plot()
df_bench[df_bench["set"] == df_bench["set"].unique()[0]]["gyro_z"].plot()
df_bench[df_bench["set"] == df_bench["set"].unique()[0]]["gyro_r"].plot()

# ----------------------------------------------------------------------------------------------------
# Configure LowPassFilter
# ----------------------------------------------------------------------------------------------------

    # Configuring  the sampling rate to match 200ms
    # Making an intance of the LowPassFilter class.
sampling_frequency = 1000/200
LowPass = LowPassFilter()

# ----------------------------------------------------------------------------------------------------
# Apply and tweak LowPassFilter
# ----------------------------------------------------------------------------------------------------

    # Making a set for each exercise DF and selecting the first index only of the set.unique.
set_bench = df_bench[df_bench["set"] == df_bench["set"].unique()[0]]
set_squat = df_squat[df_squat["set"] == df_squat["set"].unique()[0]]
set_row = df_row[df_row["set"] == df_row["set"].unique()[0]]
set_ohp = df_ohp[df_ohp["set"] == df_ohp["set"].unique()[0]]
set_dead = df_dead[df_dead["set"] == df_dead["set"].unique()[0]]

    # Single column just to see the comparison from original and LowPassFilter.
set_bench["acce_r"].plot()
LowPass.low_pass_filter(
    set_bench, col="acce_y", sampling_frequency=sampling_frequency,
    cutoff_frequency=0.4, order=5
)["acce_y_lowpass"].plot()

# ----------------------------------------------------------------------------------------------------
# Create funtion to count repetitions
# ----------------------------------------------------------------------------------------------------

    # Custom function so that there's no need to copy and paste the code here again and again
    # just to see the result of LowPassFilter and the Plot.
    # Function with its default values.
def count_reps(dataset, cutoff=0.4, order=10, column="acce_r"):
    # Starting the function by using the low_pass_filter.
    # This will insert a new column to the DF based on the selected
    # column + "_lowpass" in the column title.
    data = LowPass.low_pass_filter(
        dataset, col=column, sampling_frequency=sampling_frequency,
        cutoff_frequency=cutoff, order=order
    )
    # the indexes contains the result of agrelextrema. Which its data is from data DF's
    # lowpass column. Using the .value since the "data[column + "_lowpass"]" will return
    # a Pandas Series instead of an array.  "np.greater" returns the greater values.
    indexes = argrelextrema(data[column + "_lowpass"].values, np.greater)
    # The peaks stored the location of the indexes from the DF.
    peaks= data.iloc[indexes]
    
    # Plotting the data.
    fig, ax = plt.subplots()
    plt.plot(dataset[f"{column}_lowpass"])
    plt.plot(peaks[f"{column}_lowpass"], "o", color="red") # This will highlight the peaks in the graph.
    ax.set_ylabel(f"{column}_lowpass")
    exercise = dataset["label"].iloc[0].title()
    category = dataset["category"].iloc[0].title()
    plt.title(f"{category} {exercise}: {len(peaks)} Reps")
    plt.show()
    
    # Returns the length of the peaks. Heavy should have 5. While Medium should have 10.
    return len(peaks)

    # Trying the function. With custom cutoff values.
count_reps(set_bench, cutoff=0.4)
count_reps(set_squat, cutoff=0.35)
count_reps(set_row, cutoff=0.65, column="gyro_x")
count_reps(set_ohp, cutoff=0.35)
count_reps(set_dead, cutoff=0.4)

# ----------------------------------------------------------------------------------------------------
# Create benchmark dataframe
# ----------------------------------------------------------------------------------------------------

    # Inserting a new column "reps" in the df. Values will be based of category.
    # Heavy = 5, Medium = 10.
df["reps"] = df["category"].apply(lambda x : 5 if x == "heavy" else 10)
    # Making a DF for comparison purposes. ".groupby" method needs an aggretion method that's why there's
    # ".max()". Will probably still the same with ".min()". Also, resetting the index.
df_reps = df.groupby(["label", "category", "set"])["reps"].max().reset_index()
    # Adding a new column named "reps_prediction" and set the value to 0 for all:
df_reps["reps_prediction"] = 0

    # Will now use loop to use the function to loop all the exercises and make prediction. Insert those
    # prediction to the column "reps_prediction"
for per_set in df["set"].unique():
    # Making a subset base on the current "set"
    subset = df[df["set"] == per_set]
    
    # default values for the function:
    column = "acce_r"
    cutoff = 0.4
    
    # Custom values for particular exercise:
    if subset["set"].iloc[0] == "squat":
        cutoff = 0.35
    
    if subset["set"].iloc[0] == "row":
        cutoff = 0.65
        column = "gyro_r"
    
    if subset["set"].iloc[0] == "ohp":
        cutoff = 0.35
    
    # Calling the function, saving it to reps.
    # For this, since this a loop, I am going to disable the plotting of the function for the mean time.
    reps = count_reps(subset, cutoff=cutoff, column=column)
    
    # Saving it to the DF proper location base on the current "set".
    # Converting the value to Int so that it looks clean.
    df_reps.loc[df_reps["set"] == per_set, "reps_prediction"] = int(reps)

# ----------------------------------------------------------------------------------------------------
# Evaluate the results
# ----------------------------------------------------------------------------------------------------

    # Getting the average error for the DF it tested:
error_rate = mean_absolute_error(df_reps["reps"], df_reps["reps_prediction"]).round(2)

    # Making a visual, Plotting it:
df_reps.groupby(["label", "category"])["reps", "reps_prediction"].mean().plot.bar()