import os
import sys

wine = ["wine.csv", 0, 11, 12]
sonar = ["sonar.csv", 0, 60, 61]

data = [wine, sonar]

# algorithms = ["boosting", "dt", "knn"]
algorithms = ["svm"]

for i in range(len(data)):
    print "********** " + str(data[i][0]) + " **********"
    for a in range(len(algorithms)):
        print "---------- " + algorithms[a] + " ----------"
        os.system("python " + algorithms[a] + ".py " + 
            str(data[i][0]) + " " + 
            str(data[i][1]) + " " + 
            str(data[i][2]) + " " +
            str(data[i][3]))
