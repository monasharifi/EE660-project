'''
Splits data into training and testing data, for all features, and for mRMR top features, and calls classification_methods()

Main function: split_and_classify()

INPUT: x_train, y_train, split_ratio
x_train: all features
y_train: all labels
split_ratio: testing ratio
mRMR: mRMR indices 

OUTPUT: acc_train, acc_test, acc_train_sub, acc_test_sub
acc_train: accuracy array for training data. 
acc_test: accuracy array for testing data
acc_train_sub: accuracy array for training data using only top features selected by mRMR
acc_test_sub: accuracy array for testing data using only top features selected by mRMR
	Each ith element of the array is the accuracy of method i. See enumeration if classification_methods_function

'''
import numpy as np

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

def split_and_classify(x_train,y_train,mRMR):
    mrmr_splits = np.arange(5,51,5)
    acc_train_mat = np.zeros((9,len(mrmr_splits)+1));
    acc_test_mat = np.zeros((9,len(mrmr_splits)+1));
    cnt = 0;
    x_test, y_test, x_train, y_train = train_test_split(x_train, y_train,0.2)
	#classification for all features
    acc_train, acc_test = classification_methods(x_train, x_test, y_train, y_test)
    acc_train = acc_train.reshape((9))
    acc_test = acc_test.reshape((9))
    acc_train_mat[:,cnt] = acc_train;
    acc_test_mat[:,cnt] = acc_test;

    acc_train = np.zeros((9))
    acc_test = np.zeros((9))
    for i in mrmr_splits: 
		cnt = cnt+1
		x_train_sub = x_train[:,mRMR[:i]] #indices from mRMR
		x_test_sub =x_test[:,mRMR[:i]]	
	    #classification for top features selected by mRMR
		acc_train, acc_test = classification_methods(x_train_sub, x_test_sub, y_train, y_test)
		acc_train = acc_train.reshape((9))
		acc_train - acc_test.reshape((9))

		acc_train_mat[:,cnt] = acc_train;
		acc_test_mat[:,cnt] = acc_test;



    return acc_train_mat, acc_test_mat

                                  
                                  
                                  
                                  
                                  
