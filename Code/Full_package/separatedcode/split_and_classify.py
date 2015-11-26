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

from classification_methods_function import classification_methods

# http://stackoverflow.com/questions/5957470/matlab-style-find-function-in-python
def indices(a, func):# ~ Find in matlab  
    return [i for (i, val) in enumerate(a) if func(val)]

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
    #print test_split_x.shape
    #print test_split_y.shape
    # ------ extrating train data split 
    train_indx_1=label_1[test_range:]
    train_indx_0=label_0[test_range:]
    train_split_x=np.concatenate((x_train[train_indx_1,:],x_train[train_indx_0,:]),axis=0)
    train_split_y=np.concatenate((y_train[train_indx_1],y_train[train_indx_0]),axis=0)
    test = np.column_stack((test_split_x,test_split_y))
    train=np.column_stack((train_split_x, train_split_y))
    np.random.shuffle(train)
    np.random.shuffle(test)
    #print train 
    #print test
    train_split_y = train[:,train.shape[1]-1]
    train_split_x = train[:,:train.shape[1]-1]
    test_split_y = test[:,test.shape[1]-1]
    test_split_x = test[:,:test.shape[1]-1]
    return test_split_x, test_split_y , train_split_x , train_split_y

def split_and_classify(x_train,y_train):
                                  
    np.set_printoptions(precision=4)
    cm_list=[]
    classifier_list=[]
    plt.close('all')

    test_split_x, test_split_y , train_split_x , train_split_y = train_test_split(x_train, y_train,0.2)
    print test_split_x.shape
    print test_split_y.shape
    print train_split_x.shape
    print train_split_y.shape

    x_train = train_split_x
    y_train = train_split_y
    x_test = test_split_x
    y_test = test_split_y

    acc_train, acc_test = classification_methods(x_train, x_test, y_train, y_test)

    x_train_sub = x_train[:,[121,215,142,255,71,250,0,249,214,199,120,70,196,113,10,148,246,118,184,56,247,216,128,57,243,344,140,86,248,272,244,187,42,374,198,21,188,114,156,218,345,136,14,252,263,171,123,359,185,174]]
    x_test_sub =x_test[:,[121,215,142,255,71,250,0,249,214,199,120,70,196,113,10,148,246,118,184,56,247,216,128,57,243,344,140,86,248,272,244,187,42,374,198,21,188,114,156,218,345,136,14,252,263,171,123,359,185,174]]
    acc_train_sub, acc_test_sub = classification_methods(x_train_sub, x_test_sub, y_train, y_test)


    return acc_train, acc_test, acc_train_sub, acc_test_sub


                                  
                                  
                                  
                                  
                                  
