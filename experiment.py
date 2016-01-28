import numpy as np
import sys
from sklearn import preprocessing
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy import interp
import os
import scipy.stats as st

#os.system("python dataExtractionForClass.py "+threadName+" "+str(window)+" "+str(typeOfLog)+" "+directory+" 1")

#BEST PARAMETERS FOR SVM
#W500. nu = 0.001 gamma = 0.00141 F1 = 0.86956
#W250. nu = 0.008 gamma = 0.00821 F1 = 0.85106
#

if len(sys.argv) < 5:
    sys.exit('Usage: %s     input-directory    windowSize      Attack(-1)/Normal(1)    output-directory  ' % sys.argv[0])


path = str(sys.argv[1])
windowSize = int(sys.argv[2])
typeOfLog = str(sys.argv[3])
directory = str(sys.argv[4])

windowCounter = 1

for file in os.listdir(path):
    if file.endswith(".txt"):
        os.system("python dataExtractionForClass.py "+path+"/"+file+" "+str(windowSize)+" "+str(typeOfLog)+" "+directory+" 1")
        #print "python dataExtractionForClass.py "+path+"/"+file+" "+str(windowSize)+" "+str(typeOfLog)+" "+directory+" 1"


os.system("python dataExtractionForSynth.py D:/Docs/Scripts/SGCdetectorDNS/raw_Attacks/SyntheticAttack.txt " +str(windowSize))
#print "python dataExtractionForSynth.py D:/Docs/Scripts/SGCdetectorDNS/raw_Attacks/SyntheticAttack.txt " +str(windowSize)

path = directory+str(windowSize)
pathAttacks = "SynthAttack"+str(windowSize)


dataset = None
flag = 0
for file in os.listdir(path):
    if file.endswith(".out"):
        raw_data = file
        print file
    	if flag == 0:
            dataset = np.loadtxt(path+"/"+raw_data, delimiter=",")
            flag = 1
    	else:
    		dataset2 = np.loadtxt(path+"/"+raw_data, delimiter=",")	
    		dataset = np.concatenate((dataset,dataset2),axis=0)	
    	
datasetAbnormal = None
flag = 0
for file in os.listdir(pathAttacks):
    if file.endswith(".out"):
        print file
        if flag==0:
			datasetAbnormal = np.loadtxt(pathAttacks+"/"+file,delimiter = ",")
            
			flag = 1
        else:
			dataset3 = np.loadtxt(pathAttacks+"/"+file,delimiter = ",")
			datasetAbnormal = np.concatenate((datasetAbnormal,dataset3),axis=0)
        

X = dataset[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20]]
y = dataset[:,21]


scaler = preprocessing.StandardScaler().fit(X)
SX = scaler.transform(X) #print  SX #SX the X data catually scaled

X_train_test, X_val, y_train_test, y_val = train_test_split(SX, y, test_size=.3,
                                                    random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X_train_test,y_train_test,test_size=0.3, random_state=0)


AX = datasetAbnormal[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20]]
Ay = datasetAbnormal[:,21]
SAX = scaler.transform(AX)

AX_test, AX_val, Ay_test,Ay_val = train_test_split(AX,Ay,test_size=0.5,random_state=0)


f1Max = -1111111
bestNu= -1
bestGamma = -1
for _nu in [0.5,0.1,0.01,0.001,0.002,0.0009,0.0008]:
    for _gamma in np.arange(0.00001,0.01,0.0001):    

        #clf = svm.OneClassSVM(nu=0.001, kernel="rbf", gamma=0.0001) #hasta ahora mejor resultado
        clf = svm.OneClassSVM(nu=_nu, kernel="rbf", gamma=_gamma)
        #clf = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.012) #W 500 real attack
        clf.fit(X_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        y_pred_outliers = clf.predict(AX_test) ## predicting for abnormal behaviour
        n_error_train = y_pred_train[y_pred_train == -1].size
        #print n_error_train
        n_error_test = y_pred_test[y_pred_test == -1].size

        n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size ## counting errors abnormal behaviour

        
        tp = len(SAX)-n_error_outliers
        precision = float(tp)/(tp+n_error_test)
        recall = float(tp)/(tp+n_error_outliers)


        f1 = 2*((precision*recall)/(precision+recall))
        if f1Max <f1:
            f1Max = f1
            bestNu = str(clf.get_params()["nu"])
            bestGamma = str(clf.get_params()["gamma"])
        #print str(precision)+","+str(recall)+","+str(f1)+","+str(clf.get_params()["nu"])+","+str(clf.get_params()["gamma"])
    

print "Best params:", bestNu,bestGamma
print "for f1 = ", f1Max
#exit()
clf = svm.OneClassSVM(nu=float(bestNu), kernel="rbf", gamma=float(bestGamma))
clf.fit(X_train)
y_pred_val = clf.predict(X_val)
Ay_pred_val = clf.predict(AX_val)


n_error_val = y_pred_val[y_pred_val == -1].size

n_error_outliers = Ay_pred_val[Ay_pred_val == 1].size ## counting errors abnormal behaviour

print "FP_Test = ", str(n_error_val),"/",len(X_val)
print "FN_Attack = ",str(n_error_outliers),"/",len(AX_val)

tp = len(AX_val)-n_error_outliers
precision = float(tp)/(tp+n_error_val)
recall = float(tp)/(tp+n_error_outliers)


f1 = 2*((precision*recall)/(precision+recall))

print "precision",precision
print "recall", recall
print "*Final F1",f1

