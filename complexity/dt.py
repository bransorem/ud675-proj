"""
Plots Model Complexity graphs for Decision Trees
For Decision Trees we vary complexity by changing the size of the decision tree
"""

import sys
import pandas as pd
import time
from numpy import *
import pylab as pl
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

data_file = sys.argv[1]
start = sys.argv[2]
end = sys.argv[3]
output = sys.argv[4]

f = open("../results/dt/" + data_file + "_dt_results_" + start + "-" + end + ".txt", "w")

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

# We will vary the depth of decision trees from 2 to 25
max_depth = arange(2, 25)
train_err = zeros(len(max_depth))
test_err = zeros(len(max_depth))

times = []

for i, d in enumerate(max_depth):
    start_time = time.time()
	# Setup a Decision Tree Regressor so that it learns a tree with depth d
    regressor = DecisionTreeRegressor(max_depth=d)
    
    # Fit the learner to the training data
    regressor.fit(X_train, y_train)
    times.append(time.time() - start_time)

	# Find the MSE on the training set
    train_err[i] = mean_squared_error(y_train, regressor.predict(X_train))
    # Find the MSE on the testing set
    test_err[i] = mean_squared_error(y_test, regressor.predict(X_test))


# Plot training and test error as a function of the depth of the decision tree learnt
pl.figure()
pl.title('Decision Trees: Performance vs Max Depth')
pl.plot(max_depth, test_err, lw=2, label = 'test error')
pl.plot(max_depth, train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('Max Depth')
pl.ylabel('RMS Error')
# pl.show()
pl.savefig("../results/dt/" + data_file + "_dt_" + str(start) + "-" + str(end) + ".jpg")

pl.figure()
pl.title('Running times')
pl.plot(max_depth, times, lw=2, label="times")
pl.legend()
pl.xlabel('Max Depth')
pl.ylabel('Running Times (ms)')
pl.savefig("../results/dt/" + data_file + "_dt_" + str(start) + "-" + str(end) + "_times.jpg")


f.write("*********** Attributes (" + 
    start + "-" + 
    end + ") **********\n\n")
f.write("Train errors: \n")
f.write(str(train_err) + "\n\n")
f.write("Test errors: \n")
f.write(str(test_err) + "\n\n")
f.write("Running times: \n")
f.write(str(times))