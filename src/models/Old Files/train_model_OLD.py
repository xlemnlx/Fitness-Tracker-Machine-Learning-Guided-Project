# PART - 6

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from LearningAlgorithms import ClassificationAlgorithms


# Plot settings:
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

df = pd.read_pickle("../../data/interim/03_data_features.pkl")

# ----------------------------------------------------------------------------------------------------
# Create a training and test set
# ----------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------
    # Removing some columns since we don't need those right now. 
    # Making another DF named "x" with column "label" removed
    # Making a Pandas series named as "y" containing the "label" column values.
    # Basically, "x" holds all the numerical values, and "y" holds the exercise names
    # ----------------------------------------------------------------------------------------------------
df_train = df.drop(["participant", "category", "set"], axis=1)

x = df_train.drop(["label"], axis=1)
y = df_train["label"]

    # ----------------------------------------------------------------------------------------------------
    # Declaring 4 variables (2 DF, 2 Series). That will hold the data for the data for training and test
    # "train_test_split" randoms the data by default, that's why stratify is declared to properly arranged
    # the data in respect to the series "y". "test_size" = 0.25 means 25% of the available data will be 
    # used to test (x_test and y_test) the model, while 75% of the data will be used to train (x_train and
    # y_train) the model.
    # ----------------------------------------------------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.25, 
    random_state=42, 
    stratify=y
)

    # ----------------------------------------------------------------------------------------------------
    # Just a simple Plot to see the total data for each exercise "label", train data, and test data.
    # ----------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
df_train["label"].value_counts().plot(
    kind="bar", ax=ax, color="lightblue", label="Total"
)
y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")
y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
plt.legend()
plt.show()

# ----------------------------------------------------------------------------------------------------
# Split feature subsets
# ----------------------------------------------------------------------------------------------------

    # Splitting first by their respective features: original and the other features made by using different methods:
basic_features = ["acce_x", "acce_y", "acce_z", "gyro_x", "gyro_y", "gyro_z"]
square_features = ["acce_r", "gyro_r"]
pca_features = ["pca_1", "pca_2", "pca_3", ]
time_features = [f for f in df_train.columns if "_temp_" in f]
freq_features = [f for f in df_train.columns if (("_freq" in f) or ("_pse" in f))]
cluster_features = ["cluster"]

print(f"Basic features count: {len(basic_features)}")
print(f"Sqaured features count: {len(square_features)}")
print(f"PCA features count: {len(pca_features)}")
print(f"Time features count: {len(time_features)}")
print(f"Frequency features count: {len(freq_features)}")
print(f"Cluster features count: {len(cluster_features)}")

    # set method make sure that the data is unique (no duplicate) and then converting it back to list so that it easy to iterate.
feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(basic_features + square_features + pca_features))
feature_set_3 = list(set(feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + freq_features + cluster_features))

# ----------------------------------------------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# ----------------------------------------------------------------------------------------------------
learner = ClassificationAlgorithms()
max_features = 10

selected_features, ordered_features, ordered_scores = learner.forward_selection(
    max_features, x_train, y_train
)

    # Storing the right order of columns just to be safe since this takes a lot of time.
selected_features = ['pca_1',
 'gyro_r_freq_0.0_Hz_ws_14',
 'acce_z_freq_0.0_Hz_ws_14',
 'acce_x_freq_0.0_Hz_ws_14',
 'acce_r_freq_0.357_Hz_ws_14',
 'acce_x_freq_0.357_Hz_ws_14',
 'gyro_x',
 'acce_x_freq_1.429_Hz_ws_14',
 'gyro_y_freq_1.429_Hz_ws_14',
 'acce_x']

plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, max_features + 1, 1), ordered_scores)
plt.xlabel("Number of features")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1, max_features + 1, 1))
plt.show()

# ----------------------------------------------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# ----------------------------------------------------------------------------------------------------

    # List that will be used to loop through the grid search.
possible_feature_sets = [
    feature_set_1,
    feature_set_2,
    feature_set_3,
    feature_set_4,
    selected_features
]

feature_names = [
    "Feature Set 1",
    "Feature Set 2",
    "Feature Set 3",
    "Feature Set 4",
    "Selected Features"
]

    # Normally, iteration would be more than 1 so that you can get multiple score and get the 
    # average. Doing 1 for now to see how the code works... if it works. And a empty DF that will
    # be used to store the data.
iterations = 1
score_df = pd.DataFrame()

    # Copied code.
    # "Training neural network" finished quickly since grid_search was set to false for this time.
for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i)
    selected_train_X = x_train[possible_feature_sets[i]]
    selected_test_X = x_test[possible_feature_sets[i]]

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])

# ----------------------------------------------------------------------------------------------------
# Create a grouped bar plot to compare the results
# ----------------------------------------------------------------------------------------------------

    # Sorting the score DF from highest to lowest.
score_df.sort_values(by="accuracy", ascending=False)

    # Plotting to understand it easier.
plt.figure(figsize=(10, 10))
sns.barplot(x="model", y="accuracy", hue="feature_set", data=score_df)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.show()

# ----------------------------------------------------------------------------------------------------
# Select best model and evaluate results
# ----------------------------------------------------------------------------------------------------

    # Selecting a specific train and using Feature Set 4 since it has all the columns to test.
(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.feedforward_neural_network(
    x_train[feature_set_4], y_train, x_test[feature_set_4], gridsearch=True
)

    # Getting the accuracy:
accuracy =accuracy_score(y_test, class_test_y)

    # Confusion matrix: To get an accurate understanding where did it get right and wrong labels.
    # Storing all the column names to classes.
classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

    # create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()

# ----------------------------------------------------------------------------------------------------
# Select train and test data based on participant
# ----------------------------------------------------------------------------------------------------

    # Making a sub DF from the original DF. Dropping the set and category columns 
participant_df = df.drop(["set", "category"], axis=1)

    # Filtering the Participant B - E to train section and Participant A to Test section.
x_train = participant_df[participant_df["participant"] != "A"].drop("label", axis=1)
y_train = participant_df[participant_df["participant"] != "A"]["label"]

x_test = participant_df[participant_df["participant"] == "A"].drop("label", axis=1)
y_test = participant_df[participant_df["participant"] == "A"]["label"]

x_train = x_train.drop(["participant"], axis=1)
x_test = x_test.drop(["participant"], axis=1)

    # Plotting to see the total data for each section. Total, train, test.
fig, ax = plt.subplots(figsize=(10, 5))
df_train["label"].value_counts().plot(
    kind="bar", ax=ax, color="lightblue", label="Total"
)
y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")
y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
plt.legend()
plt.show()

# ----------------------------------------------------------------------------------------------------
# Use best model again and evaluate results
# ----------------------------------------------------------------------------------------------------

    # Copying the code from the section: Select best model and evaluate results
    # This time the test if from Participant A and we will get how accurate it is.
    # Selecting a specific train and using Feature Set 4 since it has all the columns to test.
(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.feedforward_neural_network(
    x_train[feature_set_4], y_train, x_test[feature_set_4], gridsearch=True
)

    # Getting the accuracy:
accuracy =accuracy_score(y_test, class_test_y)

    # Confusion matrix: To get an accurate understanding where did it get right and wrong labels.
    # Storing all the column names to classes.
classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

    # create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()

# ----------------------------------------------------------------------------------------------------
# Try a simpler model with the selected features
# ----------------------------------------------------------------------------------------------------

    # Copying the code from the section: Use best model again and evaluate results
    # Changing the model test to random_forest (RF). Simpler model than "feedforward_neural_network" (NN).
    # Selecting a specific train and using selected_features that was from "forward_selection" function from ClassificationAlgorithms class.
(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.random_forest(
    x_train[selected_features], y_train, x_test[selected_features], gridsearch=True
)

    # Getting the accuracy:
accuracy =accuracy_score(y_test, class_test_y)

    # Confusion matrix: To get an accurate understanding where did it get right and wrong labels.
    # Storing all the column names to classes.
classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

    # create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()
