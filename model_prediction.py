import pandas as pd
import numpy as np
import seaborn as sns # used for plot interactive graph.
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph
plt.style.use('ggplot') # nice plots
import traceback
import re
import string
from pandas import Series
from numpy import nan
from numpy import isnan
from sklearn.impute import SimpleImputer
import imblearn
from imblearn.under_sampling import NearMiss
from matplotlib import pyplot
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

import imblearn
from imblearn.under_sampling import NearMiss
from matplotlib import pyplot
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RandomizedSearchCV

from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import l1_min_c
from sklearn.model_selection import cross_val_score
from sklearn import metrics # for the check the error and accuracy of the model

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import pickle
import joblib

from data_transformation import *
# acct=pd.read_excel("./raw_data/account_attributes (1).xlsx")
# usage=pd.read_excel("./raw_data/account_usage (1).xlsx")
# iv = pd.read_csv('./iv/iv.csv')
# cutoffs = pd.read_csv("./ks/KS Table train_cutoffs.csv")


def data_prep_iv_import(usage,acct):
    
    master_data=data_prep_main(usage,acct)
    
    print(master_data)
    #print(iv)

    return master_data

def iv_column_filter(iv):
    feats=iv.loc[iv.iv >= 0.05,['feature']]
    feats = feats['feature'].to_list()
    feats = [i for i in feats if "_lt" not in i]
    feats = [i for i in feats if "SMB" not in i]

    return feats

def remove_dep(master_data, feats):
    #X_train, X_test, y_train, y_test = train_test_split(master_data,master_data.loc[:,['convert']],test_size = test_size,random_state=random_state)
    master_data_nodep=master_data.loc[:,~master_data.columns.isin(['Acct id','acct_id','convert'])]
    master_data_nodep=master_data_nodep.loc[:,feats]

    #x_test=X_test.loc[:,~X_test.columns.isin(['Acct id','acct_id','convert'])]
    #x_test=x_test.loc[:,feats]

    return master_data_nodep

def missing_value_impute(master_data_nodep):
    imputer = KNNImputer(n_neighbors=1, weights="uniform")
    master_data_nodep_nm=imputer.fit_transform(master_data_nodep)
    #y_train_nm=imputer.fit_transform(y_train)

    #x_test_nm = imputer.fit_transform(x_test)
    #y_test_nm=imputer.fit_transform(y_test)

    return master_data_nodep_nm

def scaling_data(master_data_nodep_nm):
    scaler = MinMaxScaler()
    master_data_nodep_nm=scaler.fit_transform(master_data_nodep_nm)
    #x_test_nm=scaler.fit_transform(x_test_nm)

    return master_data_nodep_nm

def write_predictions(master_data_nodep_nm, model_name,cutoffs,master_data):
    clf_fnl = joblib.load(model_name)

    pred_master = clf_fnl.predict_proba(master_data_nodep_nm)
    #pred_test = clf_fnl.predict_proba(x_test_nm)

    
    first_cutoffs = cutoffs.iloc[0,0]
    second_cutoffs = cutoffs.iloc[0,1]
    
    
    master_data['score_convert_lr'] = pred_master[:,1]
    #aster_data['score_convert_lr'] =pred_test[:,1]

    master_data['prediction_category'] = np.where(master_data['score_convert_lr']>=first_cutoffs,2,
                                              np.where(master_data['score_convert_lr']>=second_cutoffs,1,
                                                      0))


    return master_data


def model_prediction_main(usage,acct,iv,cutoffs):
    master_data=data_prep_iv_import(usage,acct)
    feats = iv_column_filter(iv)
    master_data_no_dep = remove_dep(master_data, feats)
    master_data_no_dep_nm = missing_value_impute(master_data_no_dep)
    master_data_no_dep_nm = scaling_data(master_data_no_dep_nm)
    master_data_pred = write_predictions(master_data_no_dep_nm, 'logistic_lasso.pkl',cutoffs,master_data)

    return master_data_pred

print('SUCCESS')

# if __name__=="__main__":
#     master_data_pred = model_prediction_main(usage,acct,iv,cutoffs)
#     print(master_data_pred)



    #print(master_data.head(5))
    #print(master_data.columns)