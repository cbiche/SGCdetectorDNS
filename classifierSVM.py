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

#BEST PARAMETERS FOR SVM
#W500. nu = 0.001 gamma = 0.00141 F1 = 0.86956
#W250. nu = 0.008 gamma = 0.00821 F1 = 0.85106
#

if len(sys.argv) < 3:
    sys.exit('Usage: %s    path-of-training-files    path-of-abnormal-files' % sys.argv[0])

path = str(sys.argv[1])
pathAttacks = str(sys.argv[2])

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
    	
#print len(dataset)
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
        

#X = dataset[:,1:20]
#X = dataset[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20]]

#X = dataset[:,[3,4,6,12,13,14,15,16,17,18,19,20]] #84/9216 (normal) 12/41 (abnormal)
#X = dataset[:,[3,4,5,6,12,13,14,15,16,17,18,19,20]] #86/9216 (normal) 8/41 (abnormal)
#X = dataset[:,[3,4,5,6,12,13,14]] #86/9216 (normal) 3/41 (abnormal)
X = dataset[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]

#X = dataset[:,[1,2]]#,14,15,16,17,18,20]]

y = dataset[:,21]

#print X
scaler = preprocessing.StandardScaler().fit(X)
#print scaler #scaler contains info about the scaling parameters

#print scaler.mean_ #the mean of the scaler
#print scaler.scale_ # the scaling of the scaler

SX = scaler.transform(X) #print  SX #SX the X data catually scaled

X_train, X_test, y_train, y_test = train_test_split(SX, y, test_size=.4,
                                                    random_state=0)
# Now scaling abnormal behaviour
#AX = datasetAbnormal[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20]]
AX = datasetAbnormal[:,[3,4,5,6,12,13,14]]
AX = datasetAbnormal[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
#AX = datasetAbnormal[:,[1,2]]#,14,15,16,17,18,20]]
Ay = datasetAbnormal[:,21]

#print AX
SAX = scaler.transform(AX)
#print len(SAX)
###

#_nu = 0.0008
#_gamma = 0.00821

_nu = 0.001
_gamma = 0.00141



#clf = svm.OneClassSVM(nu=0.001, kernel="rbf", gamma=0.0001) #hasta ahora mejor resultado
clf = svm.OneClassSVM(nu=_nu, kernel="rbf", gamma=_gamma)
#clf = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.012) #W 500 real attack
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(SAX) ## predicting for abnormal behaviour
n_error_train = y_pred_train[y_pred_train == -1].size
#print n_error_train
n_error_test = y_pred_test[y_pred_test == -1].size

n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size ## counting errors abnormal behaviour

#print y_pred_outliers

#print "FP_Training = ", str(n_error_train) ,"/", len(X_train)
print "FP_Test = ", str(n_error_test),"/",len(X_test)
print "FN_Attack = ",str(n_error_outliers),"/",len(SAX)

tp = len(SAX)-n_error_outliers
precision = float(tp)/(tp+n_error_test)
recall = float(tp)/(tp+n_error_outliers)


f1 = 2*((precision*recall)/(precision+recall))

print "Precision = ",str(precision)
print "recall = ",str(recall)
print "F1 = +",str(f1)

#exit()


# ### at this point we should concatenate abnormal and normal behaviour
# X_test_and_abnormal = np.concatenate((X_test,SAX),axis=0)
# y_test_and_abnormal = np.concatenate((y_test,Ay),axis=0)


# # ##
# y_score = clf.decision_function(X_test_and_abnormal)

# #print y_test_and_abnormal

# fpr,tpr,_ = roc_curve(y_test_and_abnormal,y_score,1)
# roc_auc = auc(fpr, tpr)


# plt.figure()
# plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.legend(loc="lower right")
# plt.show()


