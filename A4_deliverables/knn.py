#-------------------------------------------------------------------------
# AUTHOR: Gabriel Alfredo Siguenza
# FILENAME: knn.py
# SPECIFICATION: reads the file weather_training.csv and classifies each
#  instance from the weather_test.csv file
# FOR: CS 5990- Assignment #4
# TIME SPENT: 1.5 hours
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

#11 classes after discretization
classes = [i for i in range(-22, 39, 6)]

#defining the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

# function to discretize the target variable
def discretize(value):
    closest = min(classes, key=lambda x: abs(x - value))
    return closest

#reading the training data
training_data = pd.read_csv('weather_training.csv')
X_training = np.array(training_data.iloc[:, 1:-1]).astype(float)  # Features
y_training = np.array(training_data.iloc[:, -1]).astype(float)    # Target
y_training = np.array([discretize(val) for val in y_training])    # Discretize

#reading the test data
test_data = pd.read_csv('weather_test.csv')
X_test = np.array(test_data.iloc[:, 1:-1]).astype(float)
y_test = np.array(test_data.iloc[:, -1]).astype(float)
y_test = np.array([discretize(val) for val in y_test])    # Discretize test labels too

#initialize the highest accuracy
highest_accuracy = 0

#loop over the hyperparameter values (k, p, and w) of KNN
for k in k_values:
    for p in p_values:
        for w in w_values:

            #fitting the knn to the data
            clf = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)
            clf = clf.fit(X_training, y_training)

            #make the KNN prediction for each test sample and start computing its accuracy
            correct = 0
            total = 0
            for x_testSample, y_testSample in zip(X_test, y_test):
                prediction = clf.predict([x_testSample])[0]
                #calculate percentage difference
                if y_testSample != 0:  # avoid division by zero
                    diff_percent = 100 * abs(prediction - y_testSample) / abs(y_testSample)
                else:
                    diff_percent = 0 if prediction == y_testSample else 100  # exact match needed if true value is 0

                if diff_percent <= 15:
                    correct += 1
                total += 1

            accuracy = correct / total

            #check if the calculated accuracy is higher than the previously one calculated
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                print(f"Highest KNN accuracy so far: {highest_accuracy:.2f} Parameters: k = {k}, p = {p}, weight = {w}")
