# *******this version is for Binary classificstion with subsampling - Autism vs. Healthy ******
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
from preprocessing_data import preprocess
import csv
import  scipy.stats as stats
from sklearn.cross_validation import cross_val_score
np.set_printoptions(precision=4)
cm_list=[] 
classifier_list=[]
plt.close('all')

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
def cross_val_eval(predicted , ytrain):
        Error_rate = np.absolute(np.subtract(ytrain, predicted))
        result= np.column_stack ((ytrain,predicted))
        Acu_rate=np.column_stack ((result,Error_rate))
        temp_indx=indices(Acu_rate [:,Acu_rate.shape[1]-1], lambda x: x == 2)
        Acu_rate[temp_indx] = 0 
        #np.savetxt("yetest_ypred_multiclass.csv", Acu_rate, delimiter=",") 
        Accu =1-( np.sum(Acu_rate[:,Acu_rate.shape[1]-1])/Acu_rate.shape[0])
        return Accu 

def evaluate(clf, xtrain, xtest, ytrain, ytest, classifier_name):
        ypredtrain = clf.predict(xtrain)
        ypred = clf.predict(xtest)
        ypredSoft= clf.predict_proba(xtest)
        ypredSoft = ypredSoft[:, 1];
        print 'Confusion Matrix:' 
        cm = confusion_matrix(ytest, ypred)
        #print cm
        Error_rate = np.absolute(np.subtract(ytest, ypred))
        result= np.column_stack ((ytest,ypred))
        Acu_rate=np.column_stack ((result,Error_rate))
        temp_indx=indices(Acu_rate [:,Acu_rate.shape[1]-1], lambda x: x == 2)
        Acu_rate[temp_indx] = 0 
        #np.savetxt("yetest_ypred_multiclass.csv", Acu_rate, delimiter=",") 
        Accu =1-( np.sum(Acu_rate[:,Acu_rate.shape[1]-1])/Acu_rate.shape[0])
        
        TN = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]
        TP = cm[1][1]
        #print 'Precision = ', metrics.precision_score(ytest, ypred)
        SPC= 1.0*TN/(TN+FP) 
        print 'Specificity = ' ,np.around(SPC ,decimals=3)
        print 'Sensitivity = ', np.around(metrics.recall_score(ytest, ypred),decimals=3) # Same as Recall 
        #print 'F1 Score = ', f1_score(ytest, ypred)
        fpr, tpr, thresholds = metrics.roc_curve(ytest, ypredSoft)
        roc_auc = metrics.auc(fpr, tpr)
        print 'AUC = ', np.around(metrics.auc(fpr, tpr),decimals=3)
        print 'Accuracy on train data:%f'% np.around(metrics.accuracy_score(ytrain,ypredtrain),decimals=3)
        print 'Accuracy on test data:%f'% np.around(metrics.accuracy_score(ytest,ypred),decimals=3)
        
        #-----------Plotting the ROC Curve
        plt.plot(fpr, tpr, label=classifier_name + '(area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        #plot_cm(cm,classifier_name)
        cm_list.append(cm)
        classifier_list.append(classifier_name)
        return Accu
def evaluate1(clf, xtrain, xtest, ytrain, ytest, classifier_name):
        #ypredtrain = clf.predict(xtrain)
        ypred = np.round(clf.predict(xtest))
        print ypred
        #ypredSoft= clf.predict_proba(xtest)
        #ypredSoft = ypredSoft[:, 1];
        print 'Confusion Matrix:' 
        cm = confusion_matrix(ytest, ypred)
        #print cm
        Error_rate = np.absolute(np.subtract(ytest, ypred))
        result= np.column_stack ((ytest,ypred))
        Acu_rate=np.column_stack ((result,Error_rate))
        temp_indx=indices(Acu_rate [:,Acu_rate.shape[1]-1], lambda x: x == 2)
        Acu_rate[temp_indx] = 0 
        #np.savetxt("yetest_ypred_multiclass.csv", Acu_rate, delimiter=",") 
        Accu =1-( np.sum(Acu_rate[:,Acu_rate.shape[1]-1])/Acu_rate.shape[0])
        
        TN = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]
        TP = cm[1][1]
        #print 'Precision = ', metrics.precision_score(ytest, ypred)
        SPC= 1.0*TN/(TN+FP) 
        print 'Specificity = ' ,np.around(SPC ,decimals=3)
        print 'Sensitivity = ', np.around(metrics.recall_score(ytest, ypred),decimals=3) # Same as Recall 
        #print 'F1 Score = ', f1_score(ytest, ypred)
        fpr, tpr, thresholds = metrics.roc_curve(ytest, ypred)
        roc_auc = metrics.auc(fpr, tpr)
        print 'AUC = ', np.around(metrics.auc(fpr, tpr),decimals=3)
        #print 'Accuracy on train data:%f'% np.around(metrics.accuracy_score(ytrain,ypredtrain),decimals=3)
        print 'Accuracy on test data:%f'% np.around(metrics.accuracy_score(ytest,ypred),decimals=3)
        
'''
#from ..utils.fixes import bincount

def _balance_weights(y):

    """Compute sample weights such that the class distribution of y becomes
       balanced.
    Parameters
    ----------
    y : array-like
        Labels for the samples.
    Returns
    -------
    weights : array-like
        The sample weights.
    """

    y = np.asarray(y)
    y = np.searchsorted(np.unique(y), y)
    bins = np.bincount(y)

    weights = 1. / bins.take(y)
    weights *= bins.min()

    return weights
'''
def train_test_split(x_train, y_train,split_ratio):
# Splitting data to train and test #########
# "give Xtrain and y_train and test split size as input 
# return seperate teain and test splits for both X and Y "
    label_1 = indices(y_train , lambda x: x == 1)
    label_0 = indices(y_train , lambda x: x == 0)
    print len(label_1)
    print len(label_0)
    # ------ extrating test data split 
    test_split_ratio= split_ratio#0.2 # in put of the function 
    test_range=np.int8(len(label_1)*test_split_ratio)
    test_indx_1=label_1[0:test_range]
    test_indx_0=label_0[0:test_range]
    test_split_x=np.concatenate((x_train[test_indx_1,:],x_train[test_indx_0,:]),axis=0)
    test_split_y=np.concatenate((y_train[test_indx_1],y_train[test_indx_0]),axis=0)
    # ------ extrating train data split 
    train_indx_1=label_1[test_range:]
    train_indx_0=label_0[test_range:]
    train_split_x=np.concatenate((x_train[train_indx_1,:],x_train[train_indx_0,:]),axis=0)
    train_split_y=np.concatenate((y_train[train_indx_1],y_train[train_indx_0]),axis=0)
    test = np.column_stack((test_split_x,test_split_y))
    train=np.column_stack((train_split_x, train_split_y))
    np.random.shuffle(train)
    np.random.shuffle(test)
    train_split_y = train[:,train.shape[1]-1]
    train_split_x = train[:,:train.shape[1]-1]
    test_split_y = test[:,test.shape[1]-1]
    test_split_x = test[:,:test.shape[1]-1]
    return test_split_x, test_split_y , train_split_x , train_split_y


############### Reading the Data ###############
#--------------Importing the files
#Autism _T_group1to5_ROI_ver2
train = genfromtxt('Autism group1to5 _Autuism_Healthy_roi.csv', delimiter=',') # no activation 
train = train[1: , 2:]
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
print inds
x_train[inds]=np.take(col_mean,inds[1])
preprocessed_data= np.column_stack((x_train,y_train))

############## Standarize the features
scaler=preprocessing.StandardScaler().fit(x_train)

x_train= scaler.transform (x_train)
final_data = np.column_stack((x_train,y_train))
#np.savetxt("preprocessed_data_ROI+10_sub_scaled.csv", final_data, delimiter=",") 
print x_train.shape

####*********************************#####
####     PCA  #################
print x_train.shape
from sklearn import decomposition
pca = decomposition.PCA(n_components=40)
pca.fit(x_train)
x_train = pca.transform(x_train)
print x_train.shape
####*********************************#####

test_split_x, test_split_y , train_split_x , train_split_y = train_test_split(x_train, y_train,0.2)
x_train = train_split_x
y_train = train_split_y
x_test = test_split_x
y_test = test_split_y


'''

###### L1 ######## 
from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.1)
clf.fit(x_train,y_train)
W = clf.coef_
#print('Coefficients: \n', W)
print("Residual sum of squares of L1: %.2f"
      % np.mean((clf.predict(x_test) - y_test) ** 2))
New_inds = indices(W , lambda x: x != 0)  ### CAN BE USED IN PIPELINE SETTING
#print New_inds
#print ('L1 predictions: \n')
#print np.round(clf.predict(x_test))
#print y_test

evaluate1(clf, x_train, x_test, y_train, y_test, 'Lasso')

###### L2, Ridge regression ###### 
print ('Ridge regression')
clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
clf.fit(x_train,y_train) 
print('Alpha:', clf.alpha_)
#print x_test.shape , y_test.shape
W2 = clf.coef_
#print('Coefficients: \n', W2)
#New_inds2 = indices(W2, lambda x: x != 0)
#print New_inds2
print np.round(clf.predict(x_test))
#np.mean((clf.predict(x_test) - y_test) ** 2)
print("Residual sum of squares of L2: %.2f" % np.mean((clf.predict(x_test) - y_test) ** 2))
evaluate1(clf, x_train, x_test, y_train, y_test, 'Ridge')

##### Pipeline features from L1
x_train = x_train[: , New_inds]
x_test = x_test[: , New_inds]
'''
#============== SVM ==============
from sklearn.svm import SVC
from sklearn import metrics

clf = SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.001, kernel='rbf', max_iter=-1, probability=True,
  random_state=None, shrinking=True, tol=0.001, verbose=False) 
print "\n\n============== SVM ==================="
Accuracy =0
Accu_cross=0
for i in range(10):
    predicted = cross_validation.cross_val_predict(clf, x_train, y_train , cv=10)
    Accu_cross=cross_val_eval(predicted , y_train)
    #print Accu_cross
    scores = cross_val_score(clf, x_train, y_train , cv=10)
    score_mean=scores.mean()
    Accuracy = Accuracy + score_mean
    
print 'SVM Cross Validation Accuracy1: %f'% Accu_cross
#print 'SVM Cross Validation Accuracy2: %f'% np.around(Accuracy/10, decimals=3)

clf.fit(x_train, y_train)
print "RF evaluation on seperate Train and Test dataset:"
Acuuracy1 = evaluate(clf, x_train, x_test, y_train,y_test,'SVM')
print 'Accuracy on test data:%f'% Acuuracy1
Accu_cross=0

#============== Logistic ===================
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(class_weight='auto')
Accuracy=0
print "\n\n============== Logistic Regression ==================="
for i in range(10):
    scores = cross_val_score(clf, x_train, y_train , cv=10)
    predicted = cross_validation.cross_val_predict(clf, x_train, y_train , cv=10)
    Accu_cross=cross_val_eval(predicted , y_train)
    score_mean=scores.mean()
    Accuracy = Accuracy + score_mean

print 'LR Cross Validation Accuracy: %f'%Accu_cross

clf.fit(x_train, y_train)
print "LR evaluation on seperate Train and Test dataset:"
ypred = clf.predict(x_test)
Accuracy= evaluate(clf, x_train, x_test, y_train, y_test, 'LR')
print 'Accuracy on test data:%f'% Accuracy
Accuracy=0
Accu_cross=0
#=================== RF data set ================

#============== Random Forest ===================
from sklearn.ensemble import  RandomForestClassifier
clf = RandomForestClassifier(n_estimators=15, max_depth=None,min_samples_split=1, random_state=0)
Accuracy=0
print "\n\n============== Random Forest ==============="
for i in range(10):
    scores = cross_val_score(clf, x_train, y_train , cv=10)
    predicted = cross_validation.cross_val_predict(clf, x_train, y_train , cv=10)
    Accu_cross=cross_val_eval(predicted , y_train)
    #print Accu_cross
    score_mean=scores.mean()
    Accuracy = Accuracy + score_mean
print 'RF Cross Validation Accuracy: %f'% Accu_cross

clf.fit(x_train, y_train)
print "RF evaluation on seperate Train and Test dataset:"
ypred = clf.predict(x_test)
Acuuracy =evaluate(clf, x_train, x_test, y_train, y_test, 'RF')
#print 'Accuracy on train data:%f'% Acuuracy2
print 'Accuracy on test data:%f'% Acuuracy
Accu_cross=0

#============== kNN ===================
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=10)
Accuracy=0
print "\n\n============== kNN ==================="
for i in range(10):
    scores = cross_val_score(clf, x_train, y_train , cv=10)
    score_mean=scores.mean()
    predicted = cross_validation.cross_val_predict(clf, x_train, y_train , cv=10)
    Accu_cross=cross_val_eval(predicted , y_train)
    Accuracy = Accuracy + score_mean
print 'KNN Cross Validation Accuracy: %f'%Accu_cross

clf.fit(x_train, y_train)
print "KNN evaluation on seperate Train and Test dataset:"
Acuuracy =evaluate(clf, x_train, x_test, y_train, y_test, 'KNN')
#Acuuracy2 = evaluate(clf, x_train, x_train, y_train,y_train,'SVM')
#print 'Accuracy on train data:%f'% Acuuracy2
print 'Accuracy on test data:%f'% Acuuracy
Accu_cross=0


#=============== Gradient Boosting Classifier ====================
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=300, subsample=1.0, min_samples_split=2, min_samples_leaf=1,max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False) 
for i in range(10):
    scores = cross_val_score(clf, x_train, y_train , cv=10)
    predicted = cross_validation.cross_val_predict(clf, x_train, y_train , cv=10)
    Accu_cross=cross_val_eval(predicted , y_train)
    score_mean=scores.mean()
    Accuracy = Accuracy + score_mean
print "\n\n========== Gradient Boosting Classifier ============"
print 'GB Cross Validation Accuracy: %f'%Accu_cross

clf.fit(x_train, y_train)
print "GB evaluation on seperate Train and Test dataset:"
Acuuracy =evaluate(clf, x_train, x_test, y_train, y_test, 'GB')
#Acuuracy2 = evaluate(clf, x_train, x_train, y_train,y_train,'SVM')
#print 'Accuracy on train data:%f'% Acuuracy2
print 'Accuracy on test data:%f'% Acuuracy
Accu_cross=0
Accuracy=0
#=============== Ada Boost Classifier ====================
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()
for i in range(10):
    scores = cross_val_score(clf, x_train, y_train , cv=10)
    predicted = cross_validation.cross_val_predict(clf, x_train, y_train , cv=10)
    Accu_cross=cross_val_eval(predicted , y_train)
    score_mean=scores.mean()
    Accuracy = Accuracy + score_mean
print "\n\n========== Ada Boost Classifier ============"
print 'AdaBoost Cross Validation Accuracy: %f'%Accu_cross

clf.fit(x_train, y_train)
print "ET evaluation on seperate Train and Test dataset:"
Acuuracy =evaluate(clf, x_train, x_test, y_train, y_test, 'AdaBoost')
#Acuuracy2 = evaluate(clf, x_train, x_train, y_train,y_train,'SVM')
#print 'Accuracy on train data:%f'% Acuuracy2
print 'Accuracy on test data:%f'% Acuuracy
Accu_cross=0
Accuracy=0
#=============== Extra Trees Classifier ====================

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=1, random_state=0)
for i in range(10):
    scores = cross_val_score(clf, x_train, y_train , cv=10)
    predicted = cross_validation.cross_val_predict(clf, x_train, y_train , cv=10)
    Accu_cross=cross_val_eval(predicted , y_train)
    score_mean=scores.mean()
    Accuracy = Accuracy + score_mean
print "\n\n========== Extra Trees Classifier ============"
print 'ET Cross Validation Accuracy: %f'%Accu_cross
#print "ET evaluation on seperate Train and Test dataset:"

clf.fit(x_train, y_train)
Acuuracy =evaluate(clf, x_train, x_test, y_train, y_test, 'Extra Tree')
#Acuuracy2 = evaluate(clf, x_train, x_train, y_train,y_train,'SVM')
#print 'Accuracy on train data:%f'% Acuuracy2
print 'Accuracy on test data:%f'% Acuuracy

Accu_cross=0
Accuracy=0
#=============== Decision Tree Classifier ====================
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=10,random_state=0)
print "\n\n========== Decision Tree Classifier  ============"
for i in range(10):
    scores = cross_val_score(clf, x_train, y_train , cv=10)
    predicted = cross_validation.cross_val_predict(clf, x_train, y_train , cv=10)
    Accu_cross=cross_val_eval(predicted , y_train)
    score_mean=scores.mean()
    Accuracy = Accuracy + score_mean
print 'DT Cross Validation Accuracy:%f'%Accu_cross

print " DT evaluation on seperate Train and Test dataset:"
clf.fit(x_train, y_train)
Acuuracy =evaluate(clf, x_train, x_test, y_train, y_test, 'DecisionTree')
#Acuuracy2 = evaluate(clf, x_train, x_train, y_train,y_train,'SVM')
#print 'Accuracy on train data:%f'% Acuuracy2
print 'Accuracy on test data:%f'% Acuuracy

