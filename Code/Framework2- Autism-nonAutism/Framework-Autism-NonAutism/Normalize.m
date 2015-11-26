clear all
dataset=load('Autim_non_Autism_2more.csv');
target = dataset(:,end);
Xtrain = dataset(:,1:end-1);
% ---preprocessing train data
% ===standardization with mean 0 and unit variance
X=Xtrain-repmat(mean(Xtrain),size(Xtrain,1),1);
Xtrain_std=X./repmat(std(X),size(Xtrain,1),1);
csvwrite('Xtrain_2more.csv',Xtrain_std)

% extract the baseline and concentration matrices and calculate the
% differencce 
% Baseline = dataset(:,1:128);
% Concentration = dataset(:,129:256);
% Activation = dataset(: , 257:end-1);
% Base_concen = Baseline-Concentration;
% Data_exp3=horzcat(Base_concen,Base_concen);
% csvwrite('Data_exp3.csv',Data_exp3)