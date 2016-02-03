import sys
import os



for w in range(50,900,25):
    os.system("python experiment.py LogsUsedForExper "+str(w)+" 1 salida > params"+str(w)+".txt")
    #print "python experiment.py LogsUsedForExper"+str(w)+" 1 salida"
    