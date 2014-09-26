"""
Plots Model Complexity graphs for boosting, Adaboost in this case
For Boosting we vary model complexity by varying the number of base learners
"""

import sys
import pandas as pd
import time
from numpy import *
import pylab as pl
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

data_file = sys.argv[1]
start = sys.argv[2]
end = sys.argv[3]
output = sys.argv[4]

f = open("../results/boosting/" + data_file + "_boosting_results_" + start + "-" + end + ".txt", "w")

# Load the data
data = pd.read_csv("../data/" + data_file)
dataset = shuffle(data.values)

s = int(start)
t = int(end)
v = int(output) - 1

X = dataset[:,s:t]
y = dataset[:,v]

times = []

offset = int(0.7*len(X))
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# We will vary the number of base learners from 2 to 300
max_learners = arange(2, 300)
train_err = zeros(len(max_learners))
test_err = zeros(len(max_learners))

for i, l in enumerate(max_learners):
    start_time = time.time()
    # Set up a Adaboost Regression Learner with l base learners
    regressor = AdaBoostRegressor(n_estimators=l)

    # Fit the learner to the training data
    regressor.fit(X_train, y_train)
    times.append(time.time() - start_time)

    # Find the MSE on the training set
    train_err[i] = mean_squared_error(y_train, regressor.predict(X_train))
    # Find the MSE on the testing set
    test_err[i] = mean_squared_error(y_test, regressor.predict(X_test))

# Plot training and test error as a function of the number of base learners
pl.figure()
pl.title('Boosting: Performance vs Number of Learners')
pl.plot(max_learners, test_err, lw=2, label = 'test error')
pl.plot(max_learners, train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('Number of Learners')
pl.ylabel('RMS Error')
# pl.show()
pl.savefig("../results/boosting/" + data_file + "_boosting_" + str(start) + "-" + str(end) + ".jpg")

f.write("*********** Attributes (" + 
    start + "-" + 
    end + ") **********\n\n")
f.write("Train errors: \n")
f.write(str(train_err) + "\n\n")
f.write("Test errors: \n")
f.write(str(test_err) + "\n\n")
f.write("Running times: \n")
f.write(str(times))