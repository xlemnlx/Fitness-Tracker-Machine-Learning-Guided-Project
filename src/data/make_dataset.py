# PART - 1 & 2

import pandas as pd
from glob import glob

# ----------------------------------------------------------------------------------------------------
# Turn into function
# This covers all the individual tasks that was made from the 
# now OLD python file.
# ----------------------------------------------------------------------------------------------------
file_list = glob("../../data/raw/MetaMotion/*.csv")

def read_data_from_files(files):
    
    acce_df = pd.DataFrame()
    gyro_df = pd.DataFrame()
    
    data_path = "../../data/raw/MetaMotion"

    acce_set = 1
    gyro_set = 1

    for f in file_list:
        
        participant = f.split("-")[0].replace(data_path + "\\", "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
        
        df = pd.read_csv(f)
        
        df["participant"] = participant
        df["label"] = label
        df["category"] = category
        
        if "Accelerometer" in f:
            df["set"] = acce_set
            acce_set += 1
            acce_df = pd.concat([acce_df, df])
        
        if "Gyroscope" in f:
            df["set"] = gyro_set
            gyro_set += 1
            gyro_df = pd.concat([gyro_df, df])
    
    # changing the index from an automated incrementing number to converted epoch time.
    acce_df.index = pd.to_datetime(acce_df["epoch (ms)"], unit="ms")
    gyro_df.index = pd.to_datetime(gyro_df["epoch (ms)"], unit="ms")

    del acce_df["epoch (ms)"]
    del acce_df["time (01:00)"]
    del acce_df["elapsed (s)"]

    del gyro_df["epoch (ms)"]
    del gyro_df["time (01:00)"]
    del gyro_df["elapsed (s)"]
    
    return acce_df, gyro_df

acce_df, gyro_df = read_data_from_files(file_list)

# ----------------------------------------------------------------------------------------------------
# Merging datasets
# ----------------------------------------------------------------------------------------------------

# merge data acce_df and gyro_df, column wise.
data_merged = pd.concat([acce_df.iloc[:,:3], gyro_df], axis=1)

# Renaming of the columns
data_merged.columns = [
    "acce_x",
    "acce_y",
    "acce_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "participant",
    "label",
    "category",
    "set"
]

# ----------------------------------------------------------------------------------------------------
# Resample data (frequency conversion)
# ----------------------------------------------------------------------------------------------------

# Dictionary so that it is easy to set the apply() method to
# each columns.
sampling = {
    'acce_x': "mean",
    'acce_y': "mean",
    'acce_z': "mean",
    'gyro_x': "mean",
    'gyro_y': "mean",
    'gyro_z': "mean",
    'participant': "last",
    'label': "last",
    'category': "last",
    'set': "last"
}

# Split by day; Need to understand this since this is a list 
# comprehension. And I'm still not good at list comprehension.
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])

# Styling fix... Converting the set to int so that it doesn't
# have decimal.
data_resampled["set"] = data_resampled["set"].astype("int")

# ----------------------------------------------------------------------------------------------------
# Export dataset
# ----------------------------------------------------------------------------------------------------

# saving to Pickle file. Easier to load in python, and doesn't 
# shuffle the data when saving.
data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")


# ----------------------------------------------------------------------------------------------------
# This is not a part of the Guided Project that I've followed. I just want to see how long the 
# participants wore the sensor. 
# Make 2 new DF. One is a sub DF with the index reset. And the other one is also a sub of the sub DF.
# But the "rest" data undet the "label" column has been removed.
# ----------------------------------------------------------------------------------------------------

df_sub = data_merged.copy().reset_index()

df_filtered = df_sub[df_sub["label"] != "rest"]

def compute_total_time(dataset):
    participant_list = pd.Series(
    dataset["participant"]
    .unique()
    ).dropna().sort_values(ascending=True).tolist()
    
    for per_participant in participant_list:
        dataset_per_participant = (
            dataset[dataset["participant"] == per_participant]
            .sort_values(by="epoch (ms)")
        )
        
        time_delta = (
            dataset_per_participant["epoch (ms)"].iloc[-1] 
            - dataset_per_participant["epoch (ms)"].iloc[0]
        )
        
        days = time_delta.days
        hours = time_delta.seconds // 3600
        minutes = time_delta.seconds % 60
        
        print(f"Total time for participant {per_participant}: {days} days, {hours} hours, and {minutes} minutes.")
        
compute_total_time(df_sub)
compute_total_time(df_filtered)