import numpy as np
import sys
from sklearn import preprocessing
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy import interp
import os

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
    	

datasetAbnormal = None
flag = 0
for file in os.listdir(pathAttacks):
    if file.endswith(".out"):
        print file
        if flag==0:
			datasetAbnormal = np.loadtxt(pathAttacks+"/"+file,delimiter = ",")
			flag = 1
        else:
			dataset2 = np.loadtxt(pathAttacks+"/"+file,delimiter = ",")
			datasetAbnormal = np.concatenate((datasetAbnormal,dataset2),axis=0)
        

#X = dataset[:,1:20]
X = dataset[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20]]
y = dataset[:,21]

#print X
scaler = preprocessing.StandardScaler().fit(X)
#print scaler #scaler contains info about the scaling parameters

#print scaler.mean_ #the mean of the scaler
#print scaler.scale_ # the scaling of the scaler

SX = scaler.transform(X) #print  SX #SX the X data catually scaled

X_train, X_test, y_train, y_test = train_test_split(SX, y, test_size=.3,
                                                    random_state=0)

# Now scaling abnormal behaviour
AX = datasetAbnormal[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20]]
Ay = datasetAbnormal[:,21]

SAX = scaler.transform(AX)
print len(SAX)
###


clf = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.012)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(SAX) ## predicting for abnormal behaviour
n_error_train = y_pred_train[y_pred_train == -1].size
#print n_error_train
n_error_test = y_pred_test[y_pred_test == -1].size

n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size ## counting errors abnormal behaviour

#print len(X_test)

#print "Errors in training set....", n_error_train
#print "Errors in test set.....", n_error_test

#print "FP_Training = ", str(n_error_train/float(len(X_train))*100) , "%"
print "FP_Test = ", str(n_error_test/float(len(X_test))*100),"%"
print "FN_Attack = ",str(n_error_outliers/float(len(SAX))*100),"%"
### at this point we should concatenate abnormal and normal behaviour
X_test_and_abnormal = np.concatenate((X_test,SAX),axis=0)
y_test_and_abnormal = np.concatenate((y_test,Ay),axis=0)

##
y_score = clf.decision_function(X_test_and_abnormal)


print y_test_and_abnormal

fpr,tpr,_ = roc_curve(y_test_and_abnormal,y_score)


roc_auc = auc(fpr, tpr)

print fpr
print tpr
##########

#y_score = clf.fit(X_train, y_train).decision_function(X_test)

# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(0):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# print fpr
# print tpr
# ##############################################################################
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


