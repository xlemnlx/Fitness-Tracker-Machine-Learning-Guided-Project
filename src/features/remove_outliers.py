# PART 4

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy

# This covers all the individual tasks that was made from the 
# now OLD python file. Also, starting next activity, I'll be
# practicing the proper practice in naming variables. Right
# now, this is what I used: varName. Most of the videos and
# function names that I encounter in the internet is like
# this: var_name. I'll be using this starting next activity.

# ----------------------------------------------------------------------------------------------------
# Load the data, list columns, and copy functions
# ----------------------------------------------------------------------------------------------------
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

outlier_column = list(df.columns[:6])

def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """ Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()

def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.
    
    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption 
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset

# ----------------------------------------------------------------------------------------------------
# Choose method and deal with outliers
# ----------------------------------------------------------------------------------------------------
outlier_removed_df = df.copy() # copies the original DF.

for per_col in outlier_column:
    for per_label in df["label"].unique():
        dataset = mark_outliers_chauvenet(
            df[df["label"] == per_label], per_col
            )
        
        # This will dynamically replace outlier values to 
        # NaN to each column.
        dataset.loc[dataset[per_col + "_outlier"], per_col] = np.nan
        
        # Update the column in the original dataframe.
        # Very advance technique, I need to understand this.
        outlier_removed_df.loc[(outlier_removed_df["label"] == per_label), per_col] = dataset[per_col]
        
        nOutliers = len(dataset) - len(dataset[per_col].dropna())
        print(f"Remove {nOutliers} from {per_col} for {per_label}")
# ----------------------------------------------------------------------------------------------------
# Export new datarame
# ----------------------------------------------------------------------------------------------------
outlier_removed_df.to_pickle("../../data/interim/02_outlier_removed_chauvenet.pkl")
