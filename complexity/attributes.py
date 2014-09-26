import os
import sys

wine = ["wine.csv", 0, 11, 12]

algorithms = ["boosting", "dt", "knn"]

print "********** " + wine[0] + " **********"
for s in range(wine[2]):
    for a in range(len(algorithms)):
        print "---------- " + algorithms[a] + " ----------"
        os.system("python " + algorithms[a] + ".py " + 
            str(wine[0]) + " " + 
            str(s) + " " + 
            str(s+1) + " " +
            str(wine[3]))
