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

def unique(a):# removing duplicate rows
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1) 
    return a[ui]
############### Reading the Data ###############
#--------------Importing the files
#Autism _T_group1to5_ROI_ver2
train = genfromtxt('Autism-Non_Autism_ROI_2more.csv', delimiter=',') # no activation 
train = train[1:,:]
print train.shape
train= unique(train)
print train.shape
np.random.shuffle(train)
#print train.shape
y_train = train[:,train.shape[1]-1]
x_train = train[:,:train.shape[1]-1]

# imputing the missing values with feature mean 
import  scipy.stats as stats
col_mean = stats.nanmean(x_train,axis=0)
inds = np.where(np.isnan(x_train))
#print inds
x_train[inds]=np.take(col_mean,inds[1])
np.savetxt("Autim_non_Autism_2more.csv", x_train, delimiter=",") 


