# Data Science - Fitness Tracker Project

**NOTE:** 
> **I am in no way claiming ownership of this project. This is a guided project from [Dave Ebbelaar](https://www.youtube.com/playlist?list=PL-Y17yukoyy0sT2hoSQxn1TdV0J7-MX4K) on YouTube.  I just followed along with his video tutorial. He makes these tutorial videos since he thinks that by sharing them, this will be a great tutorial and experience for those who are just starting their data science career and have a passion for it.**

> **I have kept my unedited Python files dubbed "filename_OLD.py" so that there is something I could go back to and re-learn if needed.**

## Table of Contents:
<a id="table-of-contents"></a>

1. [About the project](#heading-1)
2. [Comparison plots for the results](#heading-2)
3. [Additional feature - Counting the repetitions](#heading-3)
4. [Comparison plots for the results](#heading-4)
5. [How the data was gathered?](#heading-5)
6. [Project Structure](#heading-6)

<a id="heading-1"></a>

## [About the project:](#table-of-contents)

Developed a machine learning model using **MLPClassifier** to automatically detect different exercise types based on sensor data. The exercises the model can predict are the following:
1. Bench Press
2. Overhead Press
3. Barbell Row
4. Dead Lift
5. Squat

The accuracy of the model is **99.61%**, with 2,576 rows of data for the training and 1,292 rows of data for the test. The dataset originally had 9,009 rows of data. But it got reduced to 3,868 rows because of data cleanup from various methods and features used to make analysis, presentation, and finally train and test the model. I've then tested the dataset with multiple machine-learning models. After getting the results from different tests, I decided to choose **MLPClassifier** from Scikit-Learn's neural_network module as the model since it performs the best of the other models that have been tested.

The following are the Python Packages used for this project:
1. Used to create, managed, and manipulate dataframes    -   **Pandas**
2. Used for computations and analysis    -   **Numpy, Scipy**
3. Used for the plots   -   **Matplotlib, Seaborn**
4. Used for training the model  -   **Scikit-Learn**

```
Here's the path location of the python file for the model:

│
├── src                
│   ├── models         
│   │   └── train_model.py <- The code for the Model.
```

<a id="heading-2"></a>

### [Comparison plots for the results:](#table-of-contents)

<figure>
    <figcaption>Overview of the data per Exercise:</figcaption>
    <img src="/reports/model/data_per_exercise.png", width="900", height="500">
</figure>

<figure>
    <figcaption>Results from various machine learning models:</figcaption>
    <img src="/reports/model/various_model_results.png", width="900", height="760">
</figure>

<figure>
    <figcaption>MLPClassifier's Confusion Matrix Plot:</figcaption>
    <img src="/reports/model/MLPClassifier_confusion_matrix.png", width="900", height="760">
</figure>

<a id="heading-3"></a>

## [Additional feature - Counting the repetitions:](#table-of-contents)

Other than the model, the project also has a feature wherein it can count the participant's repetitions of the exercise it is doing based on the list of exercises it can automatically predict. The current error value of this feature is **1.16** based on 85 rows of data that have been tested using the mean_absolute_error function from Scikit-Learn's Metrics Module, which are determined to be the peaks of each set. The function compares two sets of data: one is the original, and the other is the resampled data based on the original data, which is achieved by using Scipy's Signal Module. This project uses Pandas for the dataframe, while Numpy, Scipy, and Scikit-Learn are used to smooth out the data, return the peaks, count those peaks, make predictions, and return the error value of the predictions.

```
Here's the path location of the python file for the Count repetition feature:

│
├── src                
│   ├── features       
│   │   └── count_repetitions.py <- The code for the feature.
```

<a id="heading-4"></a>

### [Comparison plots for the results:](#table-of-contents)

<figure>
    <figcaption>Unfiltered Data:</figcaption>
    <img src="/reports/features/bench_heavy_unfiltered.png", width="918", height="240">
</figure>

<figure>
    <figcaption>Resampled Data:</figcaption>
    <img src="/reports/features/bench_heavy_lowpassfilter.png", width="918", height="240">
</figure>

<figure>
    <figcaption>Results of the predictions per exercise:</figcaption>
    <img src="/reports/features/result_bar.png", width="896", height="312">
</figure>

<a id="heading-5"></a>

## [How the data was gathered?:](#table-of-contents)

The raw data was generated using the [MetaMotion Sensor](https://mbientlab.com/metamotions/), which is a wearable wrist device. The sensor has an Accelerometer and a Gyroscope to track the movement of the participants. The Gyroscope sensor has a higher sensitivy than the Accelerometer, 25.00 Hz and 12.50 Hz, respectively. This means that the Gyroscope has twice as much data as the Accelerometer. There are five participants in this data. Their exercises have been monitored by the sensor, which saves those data as CSV files. Each exercise has two categories, which are "heavy" and "medium". Heavy sets mean the exercise is repeated five times, while medium sets are repeated ten times.

<a id="heading-6"></a>

## [Project Structure:](#table-of-contents)

This project structure is based on [Cookie Cutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) and has been edited to fit this personal project.

Here's a better look of the project structure:

```
├── README.md          <- The top-level README.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── notebooks          <- Jupyter notebooks. 
│   ├── features       <- Notebooks related to the same folder under src folder.
│   │   └── Old Files  <- Files here are the result of testing the intial codes for the project.
│   │
│   ├── models         <- Notebooks related to the same folder under src folder.
│   │   └── Old Files  <- Files here are the result of testing the intial codes for the project.
│   │
│   └── visualization  <- Notebooks related to the same folder under src folder.
│       └── Old Files  <- Files here are the result of testing the intial codes for the project.
│
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── features       <- Generated graphics and figures from the same folder at "src" to be used in reporting.
│   └── figures        <- Generated graphics and figures to be used in reporting.
│   └── model          <- Generated graphics and figures from the same folder at "src" to be used in reporting.
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module.
│   │
│   ├── data           <- Scripts to download or generate data.
│   │   └── Old Files  <- Files here are the intial codes for the project.
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling.
│   │   └── Old Files  <- Files here are the intial codes for the project.
│   │   └── build_features.py
│   │   └── count_repetitions.py
│   │   └── remove_outliers.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make.
│   │   └── Old Files  <- Files here are the intial codes for the project.
│   │   │                 predictions
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations.
│   │   └── Old Files  <- Files here are the intial codes for the project.
│       └── visualize.py
```

## Thank you for visiting!