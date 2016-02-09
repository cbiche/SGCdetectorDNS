import sys
import os



for w in range(50,225,25):#(50,900,25):
    os.system("python experimentNoLogProc.py LogsUsedForExper "+str(w)+" 1 Resultados_exp1/salida > params"+str(w)+".txt")
    #print "python experiment.py LogsUsedForExper"+str(w)+" 1 salida"
    