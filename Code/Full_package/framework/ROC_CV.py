import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from numpy import genfromtxt
from sklearn import preprocessing
import numpy as np 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import grid_search
from sklearn import datasets, linear_model, cross_validation, grid_search
import numpy as np
from numpy import genfromtxt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import grid_search
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
#from preprocessing_data import preprocess
import csv
import  scipy.stats as stats
from sklearn.cross_validation import cross_val_score

# http://stackoverflow.com/questions/5957470/matlab-style-find-function-in-python
def indices(a, func):# ~ Find in matlab  
    return [i for (i, val) in enumerate(a) if func(val)]

def unique(a):# removing duplicate rows
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1) 
    return a[ui]

def performance (cm): 
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]
    SPC= 1.0*TN/(TN+FP) 
    SNC= 1.0*TP/(TP+FN) 
    print 'Specificity = ' ,np.around(SPC ,decimals=3)
    print 'Sensitivity = ', np.around(SNC ,decimals=3) 

train = genfromtxt('Autism group1to5 _Autuism_Healthy_roi.csv', delimiter=',') # no activation 
train = train[1: , :] # *******check the file starts from feature values not group ID 
print train.shape
train= unique(train)
print train.shape
np.random.shuffle(train)
print train.shape

# getting equal number of samples from each label 
inds_1 = indices(train [:,train.shape[1]-1], lambda x: x == 1)
inds_0 = indices(train [:,train.shape[1]-1], lambda x: x == 0)
D=np.minimum(len(inds_1),len(inds_0))

indx=np.concatenate((inds_1[0:D],inds_0[0:D]),axis=0)
train= train[indx,:] # extracting a subset with equal number of samples in each class 
np.random.shuffle(train)

y_train = train[:,train.shape[1]-1]
x_train = train[:,:train.shape[1]-1]

# imputing the missing values with feature mean 
import  scipy.stats as stats
col_mean = stats.nanmean(x_train,axis=0)
inds = np.where(np.isnan(x_train))
x_train[inds]=np.take(col_mean,inds[1])
preprocessed_data= np.column_stack((x_train,y_train))


#np.savetxt("preprocessed_data_ROI10_sub_noscale.csv", preprocessed_data, delimiter=",") 
#x_train = train[:,[255,17,119,206,97,249,354,198,56,146,225,185,0,136,25,298,71,248,168,128,16,7,247,96,127,272,149,246,118,129,191,70,252,109,199,	24,121,344,250,57,135,207,142,148,120,238,293,158,153,143]]
x_train = train[:,[255,17,119,206,97,249,354,198,56,146,225,185,0,136,25,298,71,248,168,128,16,7,247,96,127,272,149,246,118,129]]


############## Standarize the features
scaler=preprocessing.StandardScaler().fit(x_train)

x_train= scaler.transform (x_train)
final_data = np.column_stack((x_train,y_train))
#print x_train.shape


###############################################################################
# Data IO and generation
X=x_train
y = y_train#iris.target
n_samples, n_features = X.shape
cm=np.zeros((2,2))
############################# SVM #############################################
# Classification and ROC analysis
# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(y, n_folds=10)
classifier = svm.SVC(kernel='linear', probability=True)   
#ROC (X,y,classifier ,cv )           

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []
mean_tpr_all = np.zeros(100)
mean_auc_all = 0.0
iter = 20
#for j in range(1,iter):
for i, (train, test) in enumerate(cv):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    ypred = classifier.predict(X[test])
    cm_fold = confusion_matrix(y[test], ypred)
    cm=cm+cm_fold
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
'''
    # outer loop 
    mean_tpr_all = mean_tpr_all+mean_tpr
    mean_auc_all = mean_auc_all+ mean_auc
mean_tpr = mean_tpr_all/iter
mean_auc = mean_auc_all/iter
'''
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC-SVM (area = %0.2f)' % mean_auc, lw=2)
print ('-------SVM -------')
performance(cm) 
print 'Mean_Accuracy -SVM = ' ,np.around(mean_auc,decimals=3)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
#ROC (X,y,classifier ,cv )  
#====================== LR ====
 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(class_weight='auto')
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    ypred = classifier.predict(X[test])
    cm_fold = confusion_matrix(y[test], ypred)
    cm=cm+cm_fold
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
print ('-------LR -------')
performance(cm) 
mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
print 'Mean_Accuracy -LR = ' ,np.around(mean_auc,decimals=3)
plt.plot(mean_fpr, mean_tpr, 'b--',
         label='Mean ROC-LR (area = %0.2f)' % mean_auc, lw=2)
plt.legend(loc="lower right") 
#==================== GB ================================
 
from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=300, subsample=1.0, min_samples_split=2, min_samples_leaf=1,max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False) 
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    ypred = classifier.predict(X[test])
    cm_fold = confusion_matrix(y[test], ypred)
    cm=cm+cm_fold
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
print ('-------GB -------')
performance(cm)     
mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
print 'Mean_Accuracy -GB = ' ,np.around(mean_auc,decimals=3)
plt.plot(mean_fpr, mean_tpr, 'm--',
         label='Mean ROC-GB (area = %0.2f)' % mean_auc, lw=2)
plt.legend(loc="lower right") 
#================KNN ======================= 

x_train = x_train[:,0:9]
############## Standarize the features
scaler=preprocessing.StandardScaler().fit(x_train)
x_train= scaler.transform (x_train)
X=x_train
y = y_train#iris.target
n_samples, n_features = X.shape
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=10)
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    ypred = classifier.predict(X[test])
    cm_fold = confusion_matrix(y[test], ypred)
    cm=cm+cm_fold
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)

print ('-------KNN -------')
performance(cm) 
mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
print 'Mean_Accuracy -KNN = ' ,np.around(mean_auc,decimals=3)
plt.plot(mean_fpr, mean_tpr, 'r--',
         label='Mean ROC-KNN (area = %0.2f)' % mean_auc, lw=2)
plt.legend(loc="lower right") 
#================ RF ===============================
 
from sklearn.ensemble import  RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=15, max_depth=None,min_samples_split=1, random_state=0)
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    ypred = classifier.predict(X[test])
    cm_fold = confusion_matrix(y[test], ypred)
    cm=cm+cm_fold
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)

print ('-------random Forest -------')
performance(cm)    
mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
print 'Mean_Accuracy -RF = ' ,np.around(mean_auc,decimals=3)
plt.plot(mean_fpr, mean_tpr, 'g--',
         label='Mean ROC-RF (area = %0.2f)' % mean_auc, lw=2)
plt.legend(loc="lower right")
#==================== NB =============================
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    ypred = classifier.predict(X[test])
    cm_fold = confusion_matrix(y[test], ypred)
    cm=cm+cm_fold
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
print ('-------Naive bayes -------')
performance(cm)   
mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
print 'Mean_Accuracy-NB = ' ,np.around(mean_auc,decimals=3)
plt.plot(mean_fpr, mean_tpr, 'c--',
         label='Mean ROC-NB (area = %0.2f)' % mean_auc, lw=2)
plt.legend(loc="lower right")
#======================= DT ========================

x_train = x_train[:,0:14]
############## Standarize the features
scaler=preprocessing.StandardScaler().fit(x_train)
x_train= scaler.transform (x_train)
X=x_train
y = y_train#iris.target
n_samples, n_features = X.shape

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=None, min_samples_split=10,random_state=0)
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    ypred = classifier.predict(X[test])
    cm_fold = confusion_matrix(y[test], ypred)
    cm=cm+cm_fold
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    
#print cm 
print ('-------Decsion tree -------')
performance (cm)

 
mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'y--',
         label='Mean ROC-DT (area = %0.2f)' % mean_auc, lw=2)
plt.legend(loc="lower right")
print 'Mean_Accuracy-DT = ' ,np.around(mean_auc,decimals=3) 
plt.show()
