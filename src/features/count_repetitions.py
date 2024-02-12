# PART - 7

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

    # This suppresses warning from chained assignments.
pd.options.mode.chained_assignment = None

    # General Plot settings:
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# ----------------------------------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------
    # Loading the data from DF and removing all data under "rest" since we don't need that. We only need
    # the exercises since we want to count the repetitions of those.
    # Then, compute for the squares of the data again, since there might some useful information that we
    # can get when couting the repetitions.
    # ----------------------------------------------------------------------------------------------------
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")
df = df[df["label"] != "rest"]

acce_r = df["acce_x"] ** 2 + df["acce_y"] ** 2 + df["acce_z"] ** 2
gyro_r = df["gyro_x"] ** 2 + df["gyro_y"] ** 2 + df["gyro_z"] ** 2
df["acce_r"] = np.sqrt(acce_r)
df["gyro_r"] = np.sqrt(gyro_r)

# ----------------------------------------------------------------------------------------------------
# Split data
# ----------------------------------------------------------------------------------------------------

    # Splitting the data --> Making 5 DF, each containing a single exercise label column.
df_bench = df[df["label"] == "bench"]
df_squat = df[df["label"] == "squat"]
df_row = df[df["label"] == "row"]
df_ohp = df[df["label"] == "ohp"]
df_dead = df[df["label"] == "dead"]

# ----------------------------------------------------------------------------------------------------
# Visualize data to identify patterns
# ----------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------
    # Single plots, just visualizing the repetition of x,y,z, and r of Accelerometer and Gyroscope.
    # ----------------------------------------------------------------------------------------------------
df_bench[df_bench["set"] == df_bench["set"].unique()[0]]["acce_x"].plot()
df_bench[df_bench["set"] == df_bench["set"].unique()[0]]["acce_y"].plot()
df_bench[df_bench["set"] == df_bench["set"].unique()[0]]["acce_z"].plot()
df_bench[df_bench["set"] == df_bench["set"].unique()[0]]["acce_r"].plot()

df_bench[df_bench["set"] == df_bench["set"].unique()[0]]["gyro_x"].plot()
df_bench[df_bench["set"] == df_bench["set"].unique()[0]]["gyro_y"].plot()
df_bench[df_bench["set"] == df_bench["set"].unique()[0]]["gyro_z"].plot()
df_bench[df_bench["set"] == df_bench["set"].unique()[0]]["gyro_r"].plot()
    # ----------------------------------------------------------------------------------------------------
    # By looking at the graph, it looks like there's only 4 repetition. Maybe the sensor is not tracking 
    # it that correctly... Need to do some adjustments.
    # ----------------------------------------------------------------------------------------------------

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
    # The graph looks good. But not the final value for cutoff_frequency. At least we have a baseline.

# ----------------------------------------------------------------------------------------------------
# Create funtion to count repetitions
# ----------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------
    # Custom function so that there's no need to copy and paste the code again and again just to se the 
    # result of LowPassFilter and the Plot.
    # Setting the function with default values for some of its arguments.
    # ----------------------------------------------------------------------------------------------------
def count_reps(dataset, cutoff=0.4, order=10, column="acce_r"):
    # ----------------------------------------------------------------------------------------------------
    # Starting the fucntion by using the "low_pass_filter". This will insert a new column to the DF based
    # on the selected column + "_lowpass" in the column title.
    # ----------------------------------------------------------------------------------------------------
    data = LowPass.low_pass_filter(
        dataset, col=column, sampling_frequency=sampling_frequency,
        cutoff_frequency=cutoff, order=order
    )
    # ----------------------------------------------------------------------------------------------------
    # The indexes contains the results of "agrelextrema" function. Which its data is from data of DF's
    # lowpass column. Using the .values since the "data[column + "_lowpass"]" will return a Pandas Series
    # instead of an array. "np.greater" returns the greater values.
    # The "peaks" stored the location of the indexes from the DF.
    # ----------------------------------------------------------------------------------------------------
    indexes = argrelextrema(data[column + "_lowpass"].values, np.greater)
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

    # ----------------------------------------------------------------------------------------------------
    # Inserting a new column "reps" in the DF. Values will be based on category. Heavy = 5, Medium = 10.
    # Then, make a DF (df_reps) for comparison purposes. ".groupby" method needs an aggretion method 
    # that's why there's ".max()". Will probably still the same with ".min()". Also, resetting the index.
    # Finally, add a new column named "reps_prediction" and set the value to 0 for all
    # ----------------------------------------------------------------------------------------------------
df["reps"] = df["category"].apply(lambda x : 5 if x == "heavy" else 10)
df_reps = df.groupby(["label", "category", "set"])["reps"].max().reset_index()
df_reps["reps_prediction"] = 0

    # ----------------------------------------------------------------------------------------------------
    # Will now Loop through the DF and use the function to make prediction of the count. Then, insert
    # those values to the column "reps_prediction".
    # ----------------------------------------------------------------------------------------------------
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
    reps = count_reps(subset, cutoff=cutoff, column=column)
    
    # ----------------------------------------------------------------------------------------------------
    # Saving it to the DF's proper location, based on the current "set". Converting the values to Int so
    # that it looks clean.
    # ----------------------------------------------------------------------------------------------------
    df_reps.loc[df_reps["set"] == per_set, "reps_prediction"] = int(reps)

# ----------------------------------------------------------------------------------------------------
# Evaluate the results
# ----------------------------------------------------------------------------------------------------

    # Getting the average error for the DF it tested:
error_rate = mean_absolute_error(df_reps["reps"], df_reps["reps_prediction"]).round(2)

    # Making a visual, Plotting it:
df_reps.groupby(["label", "category"])["reps", "reps_prediction"].mean().plot.bar()

    # ----------------------------------------------------------------------------------------------------
    # END RESULT: The error_rate gets a 1.16, which is not bad. This means that the average error for the 
    # DF it tested was just 1. It is off by 1.16 value only.
    # As for the Plot, this let us see on which specific exercise it got it wrong.
    # ----------------------------------------------------------------------------------------------------