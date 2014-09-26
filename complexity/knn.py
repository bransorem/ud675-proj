"""
Plots Model Complexity graphs for kNN
For kNN we vary complexity by chaning k
"""

import sys
import pandas as pd
import time
from numpy import *
import pylab as pl
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

data_file = sys.argv[1]
start = sys.argv[2]
end = sys.argv[3]
output = sys.argv[4]

f = open("../results/knn/" + data_file + "_knn_results_" + start + "-" + end + ".txt", "w")

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

# We will change k from 1 to 30
k_range = arange(1, 30)
train_err = zeros(len(k_range))
test_err = zeros(len(k_range))

times = []

for i, k in enumerate(k_range):
    start_time = time.time()
	# Set up a KNN model that regressors over k neighbors
    neigh = KNeighborsRegressor(n_neighbors=k)
    
    # Fit the learner to the training data
    neigh.fit(X_train, y_train)
    times.append(time.time() - start_time)

	# Find the MSE on the training set
    train_err[i] = mean_squared_error(y_train, neigh.predict(X_train))
    # Find the MSE on the testing set
    test_err[i] = mean_squared_error(y_test, neigh.predict(X_test))

# Plot training and test error as a function of k
pl.figure()
pl.title('kNN: Error as a function of k')
pl.plot(k_range, test_err, lw=2, label = 'test error')
pl.plot(k_range, train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('k')
pl.ylabel('RMS Error')
# pl.show()
pl.savefig("../results/knn/" + data_file + "_knn_" + str(start) + "-" + str(end) + ".jpg")

pl.figure()
pl.title('Running times')
pl.plot(k_range, times, lw=2, label="times")
pl.legend()
pl.xlabel('k')
pl.ylabel('Running Times')
pl.savefig("../results/knn/" + data_file + "_knn_" + str(start) + "-" + str(end) + "_times.jpg")


f.write("*********** Attributes (" + 
    start + "-" + 
    end + ") **********\n\n")
f.write("Train errors: \n")
f.write(str(train_err) + "\n\n")
f.write("Test errors: \n")
f.write(str(test_err) + "\n\n")
f.write("Running times: \n")
f.write(str(times))