# -------------------------------------------------------------------------
# AUTHOR: Gabriel Alfredo Siguenza
# FILENAME: decision_tree.py
# SPECIFICATION: This program implements a decision tree classifier using the sklearn library to predict whether a tax refund is likely based on various features.
# FOR: CS 5990 (Advanced Data Mining) - Assignment #3
# TIME SPENT: 2 hours
# -----------------------------------------------------------*/

# IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

# importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv', 'cheat_training_3.csv']
#read the test data and add this data to data_test NumPy
#--> add your Python code here
test_file = 'cheat_test.csv'

# Encoding functions
def encode_features(df):
    refund_map = {'Yes': 1, 'No': 0}
    df['Refund'] = df['Refund'].map(refund_map)

    #transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    #Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
    #be converted to a float.
    marital_status_encoded = pd.get_dummies(df['Marital Status'], prefix='', prefix_sep='')
    for col in ['Single', 'Divorced', 'Married']:
        if col not in marital_status_encoded:
            marital_status_encoded[col] = 0  # ensure all one-hot columns exist
    marital_status_encoded = marital_status_encoded[['Single', 'Divorced', 'Married']]

    df['Taxable Income'] = df['Taxable Income'].str.replace('k', '', regex=False).astype(float)

    X = pd.concat([df['Refund'], marital_status_encoded, df['Taxable Income']], axis=1).values
    return X

def encode_labels(df):
    cheat_map = {'Yes': 1, 'No': 2}
    return df['Cheat'].map(cheat_map).values

# Read and encode test data once
test_df = pd.read_csv(test_file)
X_test = encode_features(test_df)
Y_test = encode_labels(test_df)

# Process each training dataset
for ds in dataSets:
    accuracies = []

    df = pd.read_csv(ds)
    X = encode_features(df)
    Y = encode_labels(df)

    #loop your training and test tasks 10 times here
    for i in range(10):
        #fitting the decision tree to the data by using Gini index and no max_depth
        clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=None)
        clf = clf.fit(X, Y)

        # plotting the decision tree
        # Uncomment to visualize the decision tree
        tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'],
                       class_names=['Yes', 'No'], filled=True, rounded=True)
        plt.show()

        correct = 0
        total = len(Y_test)
        #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction.
        for xi, true_label in zip(X_test, Y_test):
            predicted = clf.predict([xi])[0]
            if predicted == true_label: #compare the prediction with the true label (located at data[3]) of the test instance to start calculating the model accuracy.
                correct += 1
        #find the average accuracy of this model during the 10 runs (training and test set)
        accuracy = correct / total
        accuracies.append(accuracy)
    #print the accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2
    avg_accuracy = np.mean(accuracies)
    print(f'Final accuracy when training on {ds}: {avg_accuracy:.2f}')
