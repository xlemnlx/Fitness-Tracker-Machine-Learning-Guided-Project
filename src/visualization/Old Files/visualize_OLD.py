import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# from IPython.display import display # This was use only once.
# Didn't need it.

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# ------------------------------------------------------------
# Plot single columns
# ------------------------------------------------------------
setDF = df[df["set"] == 1]  # Selecting the "Set" column only
                            # and filtering it at 1st set == 1.
plt.plot(setDF["acce_y"])   # Plotting the "acce_y" column base
                            #on the filtered data that is in "setDF".

# This shows how many data the plot has instead of showing the 
# timestamp in the x-axis.
plt.plot(setDF["acce_y"].reset_index(drop=True)) 

# ------------------------------------------------------------
# Plot all exercises
# ------------------------------------------------------------
# This function returns all the value / string of the selected 
# column without repeatition.
labelList = df["label"].unique() 

for perLabel in labelList:
    subset = df[df["label"] == perLabel]
    # display(subset.head(2)) # This just displays the data 
    # that has been selected just to see if its working correctly.
    fig, ax = plt.subplots()
    # subset instead of setDF since its a fixed string.
    plt.plot(subset["acce_y"].reset_index(drop=True), label=perLabel) 
    plt.legend()
    plt.show()
    
for perLabel in labelList:
    subset = df[df["label"] == perLabel]
    fig, ax = plt.subplots()
    # selecting the first 100 entries only.
    plt.plot(subset[:100]["acce_y"].reset_index(drop=True), label=perLabel)
    plt.legend()
    plt.show()    

# ------------------------------------------------------------
# Adjust plot settings
# ------------------------------------------------------------
# Name of the style to be use: seaborn-v0_8-deep. Can also use
# seaborn-deep, though it will throw a warning but still works.
# In the warning it will throw the "seaborn-v0_8-deep" which
# still works and will not throw an error.
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100

# ------------------------------------------------------------
# Compare medium vs. heavy sets
# ------------------------------------------------------------
categoryDF = (
    df.query("label == 'squat'")
    .query("participant == 'A'")
    .reset_index()
    )

fig, ax = plt.subplots()
categoryDF.groupby(["category"])["acce_y"].plot()
ax.set_ylabel("acce_y")
ax.set_xlabel("samples")
plt.legend() # True by default. Just enabling it.

# ------------------------------------------------------------
# Compare participants
# ------------------------------------------------------------
participantsDF = (
    df.query("label == 'bench'")
    .sort_values("participant")
    .reset_index()
    )

# Let's try to skip the sorting of participants and see how the plot goes.
# Don't forget the reset_index, because the plot will consider if you don't reset it.
# Disregard this afterwards.
participantsDF = df.query("label == 'bench'").reset_index()

fig, ax = plt.subplots()
participantsDF.groupby(["participant"])["acce_y"].plot()
ax.set_ylabel("acce_y")
ax.set_xlabel("samples")
plt.legend()

# ------------------------------------------------------------
# Plot multiple axis
# ------------------------------------------------------------
label = "squat"
participant = "A"

allAxisDF = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index()
    )

fig, ax = plt.subplots()
# The use of double [[]] is to make sure that the data that we get is "dataframe" and not a "series" type.
allAxisDF[["acce_x", "acce_y", "acce_z"]].plot(ax=ax)
ax.set_ylabel("acce_y")
ax.set_xlabel("samples")
plt.legend()

# ------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# ------------------------------------------------------------
labelList = df["label"].unique()
participantList = df["participant"].unique()

# This two for loops is for Accelerometer.
for perLabel in labelList:
    for perParticipant in participantList:
        allAxisDF = (
            df.query(f"label == '{perLabel}'")
            .query(f"participant == '{perParticipant}'")
            .reset_index()
                    )
        
        if len(allAxisDF) > 0:
            fig, ax = plt.subplots()
            allAxisDF[["acce_x", "acce_y", "acce_z"]].plot(ax=ax)
            ax.set_ylabel("acce_y")
            ax.set_xlabel("samples")
            # .tilte makes the title first letter capitalize.
            plt.title(f"{perLabel} ({perParticipant})".title()) 
            plt.legend()

# This two for loops is for Gyroscope.
for perLabel in labelList:
    for perParticipant in participantList:
        allAxisDF = (
            df.query(f"label == '{perLabel}'")
            .query(f"participant == '{perParticipant}'")
            .reset_index()
                    )
        
        if len(allAxisDF) > 0:
            fig, ax = plt.subplots()
            allAxisDF[["gyro_x", "gyro_y", "gyro_z"]].plot(ax=ax)
            ax.set_ylabel("gyro_y")
            ax.set_xlabel("samples")
            plt.title(f"{perLabel} ({perParticipant})".title())
            plt.legend()

# ------------------------------------------------------------
# Combine plots in one figure
# ------------------------------------------------------------
label = "row"
participant = "A"
combinedPlotDF = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index(drop=True)
)

# Sharex=True makes the values at the x-axis only appear at 
# the bottom of the Plots. Use this only when you have the same 
# data range like this one.
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
# [0] and [1] Indicates where you want to plot it. 0 is the top 
# and the 1 is the bottom of the plot.
combinedPlotDF[["acce_x", "acce_y", "acce_z"]].plot(ax=ax[0]) 
combinedPlotDF[["gyro_x", "gyro_y", "gyro_z"]].plot(ax=ax[1])

# Styling
ax[0].legend(
    loc="upper center", 
    bbox_to_anchor=(0.5, 1.15), 
    ncol=3, 
    fancybox=True, shadow=True
    )
ax[1].legend(
    loc="upper center", 
    bbox_to_anchor=(0.5, 1.15), 
    ncol=3, 
    fancybox=True, 
    shadow=True
    )
ax[1].set_xlabel("samples")

# ------------------------------------------------------------
# Loop over all combinations and export for both sensors
# ------------------------------------------------------------
labelList = df["label"].unique()
participantList = df["participant"].unique()

for perLabel in labelList:
    for perParticipant in participantList:
        combinedPlotDF = (
            df.query(f"label == '{perLabel}'")
            .query(f"participant == '{perParticipant}'")
            .reset_index()
                    )
        
        if len(combinedPlotDF) > 0:
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10)) 
            combinedPlotDF[["acce_x", "acce_y", "acce_z"]].plot(ax=ax[0]) 
            combinedPlotDF[["gyro_x", "gyro_y", "gyro_z"]].plot(ax=ax[1])

            ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            ax[1].set_xlabel("samples")
            
            plt.savefig(f"../../reports/figures/{perLabel.title()} ({perParticipant}).png")
            plt.show()