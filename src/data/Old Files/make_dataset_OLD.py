import pandas as pd
from glob import glob

# ------------------------------------------------------------
# Read single CSV File
# ------------------------------------------------------------

perFileAcce = pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")
perFileGyro = pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")

# ------------------------------------------------------------
# List all data in data/raw/MetaMotion
# ------------------------------------------------------------

fileList = glob("../../data/raw/MetaMotion/*.csv")

# ------------------------------------------------------------
# Extract features from filename
# ------------------------------------------------------------

dataPath = "../../data/raw/MetaMotion"
f = fileList[0]

participant = f.split("-")[0].replace(dataPath + "\\", "")
label = f.split("-")[1]
category = f.split("-")[2].rstrip("123")

df = pd.read_csv(f)

df["participant"] = participant
df["label"] = label
df["category"] = category

# ------------------------------------------------------------
# Read all files
# ------------------------------------------------------------

acceDF = pd.DataFrame()
gyroDF = pd.DataFrame()

acceSet = 1
gyroSet = 1

for f in fileList:
    
    participant = f.split("-")[0].replace(dataPath + "\\", "")
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
    
    df = pd.read_csv(f)
    
    df["participant"] = participant
    df["label"] = label
    df["category"] = category
    
    if "Accelerometer" in f:
        df["set"] = acceSet
        acceSet += 1
        acceDF = pd.concat([acceDF, df])
    
    if "Gyroscope" in f:
        df["set"] = gyroSet
        gyroSet += 1
        gyroDF = pd.concat([gyroDF, df])

# Use of additional "Set" column, sample:
acceDF[acceDF["set"] == 1] # Basically, an Identifier. Then enter to interactive jupyter. This will show data that is on "set" 1 only. Eliminating the need to filter it out using multiple columns.
    
# ------------------------------------------------------------
# Working with datetimes
# ------------------------------------------------------------

# Sample - 1
# Handling epoch / Unix time.
pd.to_datetime(df["epoch (ms)"], unit="ms")
df["time (01:00)"] # see the 1 hour difference?

# Sample 2
dateDate = pd.to_datetime(df["time (01:00)"]) # converting it to a date-time
# ^ Why convert it to date-time format? See below code:
dateDate.dt.week # Converting it to proper datetime format lets you extract it to a week / month and other data representation.
dateDate.dt.month




# Actually doing something to the Acce & Gyro DF.
acceDF.index = pd.to_datetime(acceDF["epoch (ms)"], unit="ms") # changing the index from an automated incrementing number to converted epoch time.
gyroDF.index = pd.to_datetime(gyroDF["epoch (ms)"], unit="ms")

# Deleting redundant columns on both DF.
del acceDF["epoch (ms)"]
del acceDF["time (01:00)"]
del acceDF["elapsed (s)"]

del gyroDF["epoch (ms)"]
del gyroDF["time (01:00)"]
del gyroDF["elapsed (s)"]

# ------------------------------------------------------------
# Turn into function
# ------------------------------------------------------------
fileList = glob("../../data/raw/MetaMotion/*.csv")

def readDataFromFiles(files):
    
    acceDF = pd.DataFrame()
    gyroDF = pd.DataFrame()
    
    dataPath = "../../data/raw/MetaMotion"

    acceSet = 1
    gyroSet = 1

    for f in fileList:
        
        participant = f.split("-")[0].replace(dataPath + "\\", "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
        
        df = pd.read_csv(f)
        
        df["participant"] = participant
        df["label"] = label
        df["category"] = category
        
        if "Accelerometer" in f:
            df["set"] = acceSet
            acceSet += 1
            acceDF = pd.concat([acceDF, df])
        
        if "Gyroscope" in f:
            df["set"] = gyroSet
            gyroSet += 1
            gyroDF = pd.concat([gyroDF, df])
    
    acceDF.index = pd.to_datetime(acceDF["epoch (ms)"], unit="ms") # changing the index from an automated incrementing number to converted epoch time.
    gyroDF.index = pd.to_datetime(gyroDF["epoch (ms)"], unit="ms")

    del acceDF["epoch (ms)"]
    del acceDF["time (01:00)"]
    del acceDF["elapsed (s)"]

    del gyroDF["epoch (ms)"]
    del gyroDF["time (01:00)"]
    del gyroDF["elapsed (s)"]
    
    return acceDF, gyroDF

acceDF, gyroDF = readDataFromFiles(fileList)

# ------------------------------------------------------------
# Merging datasets
# ------------------------------------------------------------
dataMerged = pd.concat([acceDF.iloc[:,:3], gyroDF], axis=1) # merge data acceDF and gyroDF, column wise.

# Disregard this, Basically what the 1 line of code below does is to see how many data or chances that the acce and gyro sensor is activated at the same time.
#dataMerged.dropna()

dataMerged.columns = [
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

# ------------------------------------------------------------
# Resample data (frequency conversion)
# ------------------------------------------------------------

# Accelerometer:    12.500Hz
# Gyroscope:        25.000Hz
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

#dataMerged[:1000].resample(rule="200ms").apply(sampling)

# Split by day; Need to understand this since this is a list comprehension.
days = [g for n, g in dataMerged.groupby(pd.Grouper(freq="D"))]

dataResampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])

# Styling fix... Converting the set to int so that it doesn't have decimal.
dataResampled["set"] = dataResampled["set"].astype("int")

# ------------------------------------------------------------
# Export dataset
# ------------------------------------------------------------

# saving to Pickle file. Easier to load in python, and doesn't shuffle the data when saving.
dataResampled.to_pickle("../../data/interim/01_data_processed.pkl")