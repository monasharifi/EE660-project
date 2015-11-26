'''
START HERE 
-loads a dataset from csv file
-preprocess the data: inputes missing values & shuffles data
-calls split_and_classify()
'''

#### variables to modify######
# mRMR indices
#d_indices :Indices_Scaled _Data_Top 50
# csvfile is the dataset
csvfile= 'Autism_T_10ROI_Autism_vs_Healthy.csv'
first10 = 1; # 1 if data has first 10
#indices for ROI only
mRMR_indices = [69,15,127,206,126,248,190,55,24,246,70,78,117,254,23,0,95,297,56,205,245,16,85,148,184,247,119,224,198,151,118, 62,197, 189,6,13,18,183,86,147,128,53,264,173,104, 54,141,145, 9, 272]

if first10 == 1:
	mRMR_indices = [ 0,2, 3 ,4, 1,8, 264, 5, 65, 7, 258, 105, 6, 130, 256, 106, 79, 9, 259, 136, 138,200, 127, 145, 261, 302,257,33, 255, 66,208,225,129,28, 151,17, 16, 262, 23, 34, 194, 207, 361, 96, 242, 64,149, 234,137, 14]

#csvfile='Autism_T_10ROI_Autism_vs_Healthy.csv'
############################

from numpy import genfromtxt
import numpy as np
from sklearn import preprocessing
import  scipy.stats as stats
from split_and_classify import split_and_classify
import csv
	

np.set_printoptions(precision=4)
cm_list=[] 
classifier_list=[]

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

def _balance_weights(y):
    y = np.asarray(y)
    y = np.searchsorted(np.unique(y), y)
    bins = np.bincount(y)
    weights = 1. / bins.take(y)
    weights *= bins.min()
    return weights


def printAverage(class_index):
#np.savetxt("preprocessed_data_aut_healthy.csv", preprocessed_data, delimiter=",")
    filename_train = 'train_'+ str(class_index) 
    filename_test = 'test_'+ str(class_index) 
    np.savetxt(filename_train, train_arr,  delimiter=",")
    np.savetxt(filename_test, train_arr,  delimiter=",")
    print "\nAverage for all ROI"
    print 'Accuracy on train data:%f'% train_arr[class_index,0]
    print 'Accuracy on test data:%f'% test_arr[class_index,0]
    print "\nAverage for mRMR ROI"
    for k in np.arange(1,11):
		print 'Accuracy on train: K = %d'% (k*5), ': %f'% train_arr[class_index,k]
		print 'Accuracy on test: K = %d'% (k*5), ': %f'% test_arr[class_index,k]

 # print 'Accuracy on train_TopK data:%f'% train_arr_sub[class_index,1]
   # print 'Accuracy on test_TopK data:%f'% test_arr_sub[class_index,1]

############### Reading the Data ###############
#--------------Importing the files
train = genfromtxt(csvfile, delimiter=',')
train = train[1: , :] #<-change this line according to csv file***********************
train= unique(train)
## removing NAN from dataset 
#train=train[~np.isnan(train).any(axis=1)] 
## saving the file without any missing values 
#np.savetxt("train_healthy_autism_roi_10.csv", train, delimiter=",") 

np.random.shuffle(train)

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

#separate first 10 before scaling
if first10 == 1:
	first10_arr = x_train[:, 0:10]
	x_train = x_train[:, 11:]

#preprocessed_data= np.column_stack((x_train,y_train))
#np.savetxt("preprocessed_data_aut_healthy.csv", preprocessed_data, delimiter=",")

############## Standarize the features
scaler=preprocessing.StandardScaler().fit(x_train)
x_train= scaler.transform (x_train)

if first10 == 1:
	x_train =  np.column_stack((first10_arr,x_train))

###do m number of iterations and obrain the accuracy average for each classification method
m=5;
train_sum= 0
test_sum = 0

for i in range(m):
    train_arr, test_arr = split_and_classify(x_train,y_train,mRMR_indices)
    train_sum= train_sum+ train_arr
    test_sum = test_sum +test_arr
    
train_arr= train_sum/m
test_arr= test_sum/m

#get headers of csv file

f = open(csvfile, 'rU')
reader = csv.DictReader(f)
colnames = reader.fieldnames
f.close()


print "\n\n============== SVM ==================="
class_index= 0
printAverage(class_index)

print "\n\n============== Random Forest ==============="
class_index+=1
printAverage(class_index)

print "\n\n============== kNN ==================="
class_index+=1
printAverage(class_index)

print "\n\n============== Logistic Regression ==================="
class_index+=1
printAverage(class_index)

print "\n\n============== Naive Bayes ==================="
class_index+=1
printAverage(class_index)

print "\n\n========== Gradient Boosting Classifier ============"
class_index+=1
printAverage(class_index)

print "\n\n========== Ada Boost Classifier ============"
class_index+=1
printAverage(class_index)

print "\n\n========== Extra Trees Classifier ============"
class_index+=1
printAverage(class_index)


print "\n\n========== Decision Tree Classifier  ============"
class_index+=1
printAverage(class_index)


print "\n\n========== mRMR Selected Features  ============"
for i in mRMR_indices:
    print  '# %d'% i, ': %s'% colnames[i] #add 1 because 1st column header is the group ID. Make it consistent with file used to get mRMR indices

filename_train = 'train_RESULTS_'+ csvfile
filename_test = 'test_RESULTS_' + csvfile
np.savetxt(filename_train, train_arr,  delimiter=",")
np.savetxt(filename_test, test_arr,  delimiter=",")

