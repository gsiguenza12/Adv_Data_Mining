#-------------------------------------------------------------------------
# AUTHOR: Gabriel Alfredo Siguenza
# FILENAME: naive_bayes.py
# SPECIFICATION: reads the file weather_training.csv and and classifies each 
#  test instance from the file weather_test.csv, uses grid search to test naive
# bayes hyper parameter s value.
# FOR: CS 5990- Assignment #4
# TIME SPENT: 2 hrs
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

#11 classes after discretization
classes = [i for i in range(-22, 39, 6)]

s_values = [0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001]

# Function to discretize the temperatures
def discretize(value):
    return min(classes, key=lambda x: abs(x - value))

#reading the training data
training_data = pd.read_csv('weather_training.csv')

#update the training class values according to the discretization (11 values only)
X_training = training_data.drop(columns=['Formatted Date', 'Temperature (C)']).values
y_training = training_data['Temperature (C)'].apply(discretize).values

#reading the test data
test_data = pd.read_csv('weather_test.csv')

#update the test class values according to the discretization (11 values only)
X_test = test_data.drop(columns=['Formatted Date', 'Temperature (C)']).values
y_test = test_data['Temperature (C)'].apply(discretize).values

highest_accuracy = 0

#loop over the hyperparameter value (s)
for s in s_values:

    #fitting the naive_bayes to the data
    clf = GaussianNB(var_smoothing=s)
    clf = clf.fit(X_training, y_training)

    #make the naive_bayes prediction for each test sample and start computing its accuracy
    predictions = clf.predict(X_test)
    correct = 0

    for pred, real in zip(predictions, y_test):
        if real != 0:  # Avoid division by zero
            percentage_difference = 100 * abs(pred - real) / abs(real)
            if percentage_difference <= 15:
                correct += 1
        else:
            # If real value is 0, check exact match
            if pred == real:
                correct += 1

    accuracy = correct / len(y_test)

    # check if the calculated accuracy is higher than the previously one calculated
    if accuracy > highest_accuracy:
        highest_accuracy = accuracy
        print(f"Highest Naïve Bayes accuracy so far: {highest_accuracy:.2f} Parameter: s = {s}")
