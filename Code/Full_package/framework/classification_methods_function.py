'''
calls the following classification methods and evaluates the accuracy.
0 - SVM 
1 - Random Forest
2 - kNN
3 - Logistic Regression 
4 - Naive Bayes 
5 - Gradient Boosting Classifier
6 - Ada Boost Classifier
7 - Extra Trees Classifier
8 - Decision Tree Classifier 

Main function: classification_methods()
INPUT:  x_train, x_test, y_train, y_test 
OUTPUT: acc_train, acc_test (accuracy array for training and testing data)
Each ith element of the array is the accuracy of method i.
'''

import numpy as np 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, cross_validation
from sklearn.preprocessing import Imputer
from sklearn import metrics
    


def evaluate(clf, xtrain, xtest, ytrain, ytest, classifier_name):
    np.set_printoptions(precision=4)
    cm_list=[]
    classifier_list=[]

    ypredtrain = clf.predict(xtrain)
    ypred = clf.predict(xtest)
    ypredSoft= clf.predict_proba(xtest)
    ypredSoft = ypredSoft[:, 1];
    print 'Confusion Matrix:'
    cm = confusion_matrix(ytest, ypred)
    print cm
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]
    SPC= 1.0*TN/(TN+FP)
    print_flag = 1

    if print_flag == 1:
        print 'Specificity = ' ,np.around(SPC ,decimals=3)
        print 'Sensitivity = ', np.around(metrics.recall_score(ytest, ypred),decimals=3) # Same as Recall 
        fpr, tpr, thresholds = metrics.roc_curve(ytest, ypredSoft)
        roc_auc = metrics.auc(fpr, tpr)
        print 'AUC = ', np.around(metrics.auc(fpr, tpr),decimals=3)
        acc_train = np.around(metrics.accuracy_score(ytrain,ypredtrain),decimals=3)
        acc_test = np.around(metrics.accuracy_score(ytest,ypred),decimals=3)
        #print 'Accuracy on train data:%f'% acc_train
        print 'Accuracy on test data:%f'% acc_test
    
    #-----------Plotting the ROC Curve
    plot_flag = 0 #if multiple iterations, turn plot_flag to 0, or close window after every iteration
    if plot_flag == 1:
        plt.plot(fpr, tpr, label=classifier_name + '(area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        if (classifier_name == 'DT'):
            plt.show()

    #plot_cm(cm,classifier_name)
    cm_list.append(cm)
    classifier_list.append(classifier_name)

    #return acc_train, acc_test
    return acc_test
#from ..utils.fixes import bincount

def classification_methods(x_train, x_test, y_train, y_test):
    from sklearn.cross_validation import cross_val_score
    iter=10 # number of doing k-fold cross val , used as an input in range of the classifiers for loop   
    acc_train = np.zeros((9,1))
    acc_test = np.zeros((9,1))

    #------------ SVM
    from sklearn.svm import SVC
    from sklearn import metrics

    clf = SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.001, kernel='rbf', max_iter=-1, probability=True,random_state=None, shrinking=True, tol=0.001, verbose=False) 
    print "\n\n============== SVM ==================="
    Accuracy =0
    for i in range(iter):
        scores = cross_val_score(clf, x_train, y_train , cv=5)
        score_mean=scores.mean()
        Accuracy = Accuracy + score_mean
    Accu_cross_val=Accuracy/iter
    print 'SVM Cross Validation Accuracy: %f'% np.around(Accuracy/iter, decimals=3)

    clf.fit(x_train, y_train)
    print "RF evaluation on seperate Train and Test dataset:"
    #acc_train[0], acc_test[0] = evaluate(clf, x_train, x_test, y_train, y_test, 'SVM')
    acc_train[0] = Accu_cross_val
    acc_test[0] = evaluate(clf, x_train, x_test, y_train, y_test, 'SVM')

    #============== Random Forest ===================
    from sklearn.ensemble import  RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=30, max_depth=None,min_samples_split=1, random_state=0)
    Accuracy=0
    print "\n\n============== Random Forest ==============="
    for i in range(iter):
        scores = cross_val_score(clf, x_train, y_train , cv=10)
        score_mean=scores.mean()
        Accuracy = Accuracy + score_mean
    Accu_cross_val=Accuracy/iter
    print 'RF Cross Validation Accuracy: %f'% np.around(Accuracy/iter, decimals=3)

    clf.fit(x_train, y_train)
    print "RF evaluation on seperate Train and Test dataset:"
    #acc_train[1], acc_test[1] = evaluate(clf, x_train, x_test, y_train, y_test, 'RF')
    acc_train[1]= Accu_cross_val
    acc_test[1] = evaluate(clf, x_train, x_test, y_train, y_test, 'RF')
    #============== kNN ===================
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=10)
    Accuracy=0
    print "\n\n============== kNN ==================="
    for i in range(iter):
        scores = cross_val_score(clf, x_train, y_train , cv=10)
        score_mean=scores.mean()
        Accuracy = Accuracy + score_mean
    Accu_cross_val=Accuracy/iter
    print 'KNN Cross Validation Accuracy: %f'%np.around(Accuracy/iter, decimals=3)

    clf.fit(x_train, y_train)
    print "KNN evaluation on seperate Train and Test dataset:"
    acc_train[2] = Accu_cross_val
    acc_test[2] = evaluate(clf, x_train, x_test, y_train, y_test, 'kNN')
    #============== Logistic ===================
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(class_weight='auto')
    Accuracy=0
    print "\n\n============== Logistic Regression ==================="
    for i in range(iter):
        scores = cross_val_score(clf, x_train, y_train , cv=10)
        score_mean=scores.mean()
        Accuracy = Accuracy + score_mean
    Accu_cross_val=Accuracy/iter
    print 'LR Cross Validation Accuracy: %f'%np.around(Accuracy/iter, decimals=3)
    clf.fit(x_train, y_train)
    print "LR evaluation on seperate Train and Test dataset:"
    acc_train[3] = Accu_cross_val
    acc_test[3] = evaluate(clf, x_train, x_test, y_train, y_test, 'LG')
    Accuracy=0
    #=============== Naive Bayes ====================
    print "\n\n============== Naive Bayes ==================="
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    for i in range(iter):
        scores = cross_val_score(clf, x_train, y_train , cv=10)
        score_mean=scores.mean()
        Accuracy = Accuracy + score_mean
    Accu_cross_val=Accuracy/iter
    print 'NB Cross Validation Accuracy: %f'%np.around(Accuracy/iter, decimals=3)
    Accuracy=0
    print "NB evaluation on seperate Train and Test dataset:"
    clf.fit(x_train, y_train)
    acc_train[4] = Accu_cross_val
    acc_test[4] = evaluate(clf, x_train, x_test, y_train, y_test, 'NB')

    #=============== Gradient Boosting Classifier ====================
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=300, subsample=1.0, min_samples_split=2, min_samples_leaf=1,max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False) 
    for i in range(iter):
        scores = cross_val_score(clf, x_train, y_train , cv=10)
        score_mean=scores.mean()
        Accuracy = Accuracy + score_mean
    Accu_cross_val=Accuracy/iter
    print "\n\n========== Gradient Boosting Classifier ============"
    print 'GB Cross Validation Accuracy: %f'%np.around(Accuracy/iter, decimals=3)
    clf.fit(x_train, y_train)
    print "GB evaluation on seperate Train and Test dataset:"
    acc_train[5] = Accu_cross_val
    acc_test[5] = evaluate(clf, x_train, x_test, y_train, y_test, 'GBC')
    Accuracy=0
    #=============== Ada Boost Classifier ====================
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier()
    for i in range(iter):
        scores = cross_val_score(clf, x_train, y_train , cv=10)
        score_mean=scores.mean()
        Accuracy = Accuracy + score_mean
    Accu_cross_val=Accuracy/iter
    print "\n\n========== Ada Boost Classifier ============"
    print 'AdaBoost Cross Validation Accuracy: %f'%np.around(Accuracy/iter, decimals=3)
    clf.fit(x_train, y_train)
    print "ET evaluation on seperate Train and Test dataset:"
    acc_train[6] = Accu_cross_val
    acc_test[6] = evaluate(clf, x_train, x_test, y_train, y_test, 'ABC')

    Accuracy=0
    #=============== Extra Trees Classifier ====================

    from sklearn.cross_validation import cross_val_score
    from sklearn.ensemble import ExtraTreesClassifier
    clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=1, random_state=0)
    for i in range(iter):
        scores = cross_val_score(clf, x_train, y_train , cv=10)
        score_mean=scores.mean()
        Accuracy = Accuracy + score_mean
    Accu_cross_val=Accuracy/iter
    print "\n\n========== Extra Trees Classifier ============"
    print 'ET Cross Validation Accuracy: %f'%np.around(Accuracy/iter, decimals=3)
    print "ET evaluation on seperate Train and Test dataset:"
    clf.fit(x_train, y_train)
    acc_train[7] = Accu_cross_val
    acc_test[7] = evaluate(clf, x_train, x_test, y_train, y_test, 'ET')
    Accuracy=0
    #=============== Decision Tree Classifier ====================
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth=None, min_samples_split=10,random_state=0)
    print "\n\n========== Decision Tree Classifier  ============"
    for i in range(iter):
        scores = cross_val_score(clf, x_train, y_train , cv=10)
        score_mean=scores.mean()
        Accuracy = Accuracy + score_mean
    Accu_cross_val=Accuracy/iter
    print 'DT Cross Validation Accuracy:%f'%np.around(Accuracy/iter, decimals=3)
    print " DT evaluation on seperate Train and Test dataset:"
    clf.fit(x_train, y_train)
    acc_train[8] = Accu_cross_val
    acc_test[8] = evaluate(clf, x_train, x_test, y_train, y_test, 'DT')
    	
    return acc_train, acc_test
