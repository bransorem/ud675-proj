"""
Plots Model Complexity graphs for Support Vector Machines
We vary complexity by changing the kernel
"""

import sys
import pandas as pd
import time
import pylab as pl
from numpy import *
from sklearn.utils import shuffle
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

data_file = sys.argv[1]
start = sys.argv[2]
end = sys.argv[3]
output = sys.argv[4]

f = open("../results/svm/" + data_file + "_svm_results_" + start + "-" + end + ".txt", "w")

# Load the data
data = pd.read_csv("../data/" + data_file)
dataset = shuffle(data.values)

s = int(start)
t = int(end)
v = int(output) - 1

X = dataset[:,s:t]
y = dataset[:,v]

offset = int(0.7*len(X))
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

times = []

f.write("*********** Attributes (" + 
    start + "-" + 
    end + ") **********\n\n")

# -----------------------------------
# Learn a SVM with a linear kernel
start_time = time.time()
clf = SVR(kernel='poly', degree=1)
clf.fit(X_train, y_train)
times.append(time.time() - start_time)

train_err = mean_squared_error(y_train, clf.predict(X_train))
test_err = mean_squared_error(y_test, clf.predict(X_test))

f.write("Linear Kernel\n")
f.write("Train errors: \n")
f.write(str(train_err) + "\n")
f.write("Test errors: \n")
f.write(str(test_err) + "\n\n")

# -----------------------------------
# Learn a SVM with a polynomial kernel
start_time = time.time()
clf = SVR(kernel='poly', degree=2)
# Fit the learner to the training data
clf.fit(X_train, y_train)
times.append(time.time() - start_time)

# Find the MSE on the training and testing set
train_err = mean_squared_error(y_train, clf.predict(X_train))
test_err = mean_squared_error(y_test, clf.predict(X_test))

f.write("Poly Kernel with degree 2\n")
f.write("Train errors: \n")
f.write(str(train_err) + "\n")
f.write("Test errors: \n")
f.write(str(test_err) + "\n\n")

# -----------------------------------
# Learn a SVM with a RBF kernel with degree 2
start_time = time.time()
clf = SVR(kernel='rbf', degree=2)
# Fit the learner to the training data
clf.fit(X_train, y_train)
times.append(time.time() - start_time)

# Find the MSE on the training and testing set
train_err = mean_squared_error(y_train, clf.predict(X_train))
test_err = mean_squared_error(y_test, clf.predict(X_test))

f.write("RBF Kernel with degree 2\n")
f.write("Train errors: \n")
f.write(str(train_err) + "\n")
f.write("Test errors: \n")
f.write(str(test_err) + "\n\n")

# -----------------------------------
# Learn a SVM with a RBF kernel with degree 3
start_time = time.time()
clf = SVR(kernel='rbf', degree=3)
# Fit the learner to the training data
clf.fit(X_train, y_train)
times.append(time.time() - start_time)

# Find the MSE on the training and testing set
train_err = mean_squared_error(y_train, clf.predict(X_train))
test_err = mean_squared_error(y_test, clf.predict(X_test))

f.write("RBF Kernel with degree 3\n")
f.write("Train errors: \n")
f.write(str(train_err) + "\n")
f.write("Test errors: \n")
f.write(str(test_err) + "\n\n")

f.write("Running times: \n")
f.write(str(times))