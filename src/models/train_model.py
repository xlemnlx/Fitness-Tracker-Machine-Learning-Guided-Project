# PART - 6

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from LearningAlgorithms import ClassificationAlgorithms


# General plot settings:
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# Load the data.
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
    # Declaring 4 variables (2 DF, 2 Series). That will hold the data for training and test.
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

    # ----------------------------------------------------------------------------------------------------
    # Making 5 variables and inserting in there the column names in respect of their features. Example:
    # Basic features has the columns for Accelerometer and Gyroscope. Sqaure features has the columns for
    # for the squared value and so on.
    # ----------------------------------------------------------------------------------------------------
basic_features = ["acce_x", "acce_y", "acce_z", "gyro_x", "gyro_y", "gyro_z"]
square_features = ["acce_r", "gyro_r"]
pca_features = ["pca_1", "pca_2", "pca_3", ]
time_features = [f for f in df_train.columns if "_temp_" in f]
freq_features = [f for f in df_train.columns if (("_freq" in f) or ("_pse" in f))]
cluster_features = ["cluster"]

    # ----------------------------------------------------------------------------------------------------
    # OPTIONAL: Printing the values of each list variables. To see if it has the correct column names.
    # ----------------------------------------------------------------------------------------------------
print(f"Basic features count: {len(basic_features)}")
print(f"Sqaured features count: {len(square_features)}")
print(f"PCA features count: {len(pca_features)}")
print(f"Time features count: {len(time_features)}")
print(f"Frequency features count: {len(freq_features)}")
print(f"Cluster features count: {len(cluster_features)}")

    # ----------------------------------------------------------------------------------------------------
    # Make a 4 list variables that start with just holding the "Basic" feature list. And the number of
    # features it hold increments as list number increments. This will be use to test the model in 
    # increment and see what columns perform the best for the model to predict accurately as possible.
    # ----------------------------------------------------------------------------------------------------
feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(basic_features + square_features + pca_features))
feature_set_3 = list(set(feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + freq_features + cluster_features))

# ----------------------------------------------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# ----------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------
    # Making an instace of the ClassificationAlgorithms class.
    # Setting the max_features = 10. This will be use for the "forward_selection" function of the the class
    # and is used to train the model. It will then output which column performs the best by order and the 
    # percentage of its accuracy respectively.
    # ----------------------------------------------------------------------------------------------------
learner = ClassificationAlgorithms()
max_features = 10

    # The "forward_selection" function.
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

    # Plotting the "ordered_scores" to see the diminishing return.
plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, max_features + 1, 1), ordered_scores)
plt.xlabel("Number of features")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1, max_features + 1, 1))
plt.show()

# ----------------------------------------------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# ----------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------
    # List within a List. Remember that this lists contains the column names of the DF. Also, the 
    # "feature_set_X" holds the column in incremental way while the "selected_features" are the columns
    # that performs the best in the "forwar_selecttion" function. The Loop later will loop through this
    # to train the different models. 
    # ----------------------------------------------------------------------------------------------------
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

    # ----------------------------------------------------------------------------------------------------
    # The "iterations" holds the value on how many iteration a model will train to the current "possible_
    # feature_sets".
    # The empty DF will hold of the scores of each model at the end of the loop.
    # ----------------------------------------------------------------------------------------------------
iterations = 5
score_df = pd.DataFrame()

    # ----------------------------------------------------------------------------------------------------
    # I just copied this code from the tutorial that I'm watching. This Loop basically loops through all of
    # "possible_feature_sets" list. Each select list contains the columns for the DF and will be tested by
    # multiple models. Then those models will have an score that will then be inserted to the "score_df" at
    # the end. I'll retain the comments of the owner inside the loop.
    # ----------------------------------------------------------------------------------------------------
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
            gridsearch=True,
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

    # ----------------------------------------------------------------------------------------------------
    # Selecting the "feedforward_neural_network" since it performs the best. Will be using the "feature_
    # set_4" for the model.
    # ----------------------------------------------------------------------------------------------------
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

    # ----------------------------------------------------------------------------------------------------
    # Storing all the column names to "classes".
    # Will be using the "confusion_matrix" function to understand where did the model get it right and
    # wrong labels.
    # ----------------------------------------------------------------------------------------------------
classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

    # ----------------------------------------------------------------------------------------------------
    # Copied plot settings code for plotting a nice matrix representation (from "cm") of the data.. 
    # Kind of complex.. But the plot will help us to understand where did it get wrong easier.
    # ----------------------------------------------------------------------------------------------------
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

    # ----------------------------------------------------------------------------------------------------
    # Basically, what this section of the code does is we're not putting all participants data to be train. 
    # The data that will be used for the training are for Participants B - E. And Participant A will be 
    # used for the test. This will give us a better result since the model doesn't have the data for the 
    # Participant A and we'll see a more realistic result in predicting result of accuracy.
    #
    # Making a sub DF from the original DF. Dropping the set and category columns 
    # ----------------------------------------------------------------------------------------------------
participant_df = df.drop(["set", "category"], axis=1)

    # Filtering the Participant B - E to Train section and Participant A to Test section.
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

    # ----------------------------------------------------------------------------------------------------
    # Copying the code from the section: Select best model and evaluate results. This time, the test is 
    # from Participant A and we will see how accurate it is. Using the "feature_set_4" for column selection.
    # Overall, this will give a proper result since what we want for our model to do is for it to be able
    # to cater new people (participant) to predict automatically the exercise they are doing and not 
    # predicting existing participant exercise since it already has the data for them.
    # ----------------------------------------------------------------------------------------------------
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

    # ----------------------------------------------------------------------------------------------------
    # Storing all the column names to "classes".
    # Will be using the "confusion_matrix" function to understand where did the model get it right and
    # wrong labels.
    # ----------------------------------------------------------------------------------------------------
classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

    # ----------------------------------------------------------------------------------------------------
    # Copied plot settings code for plotting a nice matrix representation (from "cm") of the data.. 
    # Kind of complex.. But the plot will help us to understand where did it get wrong easier.
    # ----------------------------------------------------------------------------------------------------
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
    # END RESULT: The test result is still really good. This makes the model able to predict new 
    # participant (people) the right exercise it is doing which it has no data to. 
    # Accuracy of 99.61%. 
    # Predicted ohp wrong 3 times as bench press. 
    # Predicted dead lift and ohp wrong once each as row.
    # ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# Try a simpler model with the selected features
# ----------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------
    # This time, will be using "random_forest" (RN) since it says to try a simpler model. Also, instead
    # of using the "feature_set_4", will be using the "selected_features" from the "forward_selection"
    # function. At the end, will the scores and plot it to understand the scores easier.
    # ----------------------------------------------------------------------------------------------------
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

    # ----------------------------------------------------------------------------------------------------
    # Storing all the column names to "classes".
    # Will be using the "confusion_matrix" function to understand where did the model get it right and
    # wrong labels.
    # ----------------------------------------------------------------------------------------------------
classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

    # ----------------------------------------------------------------------------------------------------
    # Copied plot settings code for plotting a nice matrix representation (from "cm") of the data.. 
    # Kind of complex.. But the plot will help us to understand where did it get wrong easier.
    # ----------------------------------------------------------------------------------------------------
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
    # END RESULT: Using simpler model still leads to good result overall. It manages to predicts 98.91% 
    # correctly. Predicts ohp wrong 5 times as bench press. Predicts dead lift wrong 9 time as row.
    # ----------------------------------------------------------------------------------------------------