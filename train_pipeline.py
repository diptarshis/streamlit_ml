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

import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import pywt

from datetime import datetime
from datetime import date
import holidays
import pandas as pd
from datetime import timedelta
from holidays import country_holidays
import seaborn as sns
import matplotlib.pyplot as plt

from exception import *

import sys

from data_transformation import *

import pandas as pd, numpy as np, os, re, math, time


######INFORMATION VALUE
# to check monotonicity of a series
def is_monotonic(temp_series):
    return all(temp_series[i] <= temp_series[i + 1] for i in range(len(temp_series) - 1)) or all(temp_series[i] >= temp_series[i + 1] for i in range(len(temp_series) - 1))

def prepare_bins(bin_data, c_i, target_col, max_bins):
    force_bin = True
    binned = False
    remarks = np.nan
    # ----------------- Monotonic binning -----------------
    for n_bins in range(max_bins, 2, -1):
        try:
            bin_data[c_i + "_bins"] = pd.qcut(bin_data[c_i], n_bins, duplicates="drop")
            monotonic_series = bin_data.groupby(c_i + "_bins")[target_col].mean().reset_index(drop=True)
            if is_monotonic(monotonic_series):
                force_bin = False
                binned = True
                remarks = "binned monotonically"
                break
        except:
            pass
    # ----------------- Force binning -----------------
    # creating 2 bins forcefully because 2 bins will always be monotonic
    if force_bin or (c_i + "_bins" in bin_data and bin_data[c_i + "_bins"].nunique() < 2):
        _min=bin_data[c_i].min()
        _mean=bin_data[c_i].mean()
        _max=bin_data[c_i].max()
        bin_data[c_i + "_bins"] = pd.cut(bin_data[c_i], [_min, _mean, _max], include_lowest=True)
        if bin_data[c_i + "_bins"].nunique() == 2:
            binned = True
            remarks = "binned forcefully"
    
    if binned:
        return c_i + "_bins", remarks, bin_data[[c_i, c_i+"_bins", target_col]].copy()
    else:
        remarks = "couldn't bin"
        return c_i, remarks, bin_data[[c_i, target_col]].copy()

# calculate WOE and IV for every group/bin/class for a provided feature
def iv_woe_4iter(binned_data, target_col, class_col):
    if "_bins" in class_col:
        binned_data[class_col] = binned_data[class_col].cat.add_categories(['Missing'])
        binned_data[class_col] = binned_data[class_col].fillna("Missing")
        temp_groupby = binned_data.groupby(class_col).agg({class_col.replace("_bins", ""):["min", "max"],
                                                           target_col: ["count", "sum", "mean"]}).reset_index()
    else:
        binned_data[class_col] = binned_data[class_col].fillna("Missing")
        temp_groupby = binned_data.groupby(class_col).agg({class_col:["first", "first"],
                                                           target_col: ["count", "sum", "mean"]}).reset_index()
    
    temp_groupby.columns = ["sample_class", "min_value", "max_value", "sample_count", "event_count", "event_rate"]
    temp_groupby["non_event_count"] = temp_groupby["sample_count"] - temp_groupby["event_count"]
    temp_groupby["non_event_rate"] = 1 - temp_groupby["event_rate"]
    temp_groupby = temp_groupby[["sample_class", "min_value", "max_value", "sample_count",
                                 "non_event_count", "non_event_rate", "event_count", "event_rate"]]
    
    if "_bins" not in class_col and "Missing" in temp_groupby["min_value"]:
        temp_groupby["min_value"] = temp_groupby["min_value"].replace({"Missing": np.nan})
        temp_groupby["max_value"] = temp_groupby["max_value"].replace({"Missing": np.nan})
    temp_groupby["feature"] = class_col
    if "_bins" in class_col:
        temp_groupby["sample_class_label"]=temp_groupby["sample_class"].replace({"Missing": np.nan}).astype('category').cat.codes.replace({-1: np.nan})
    else:
        temp_groupby["sample_class_label"]=np.nan
    temp_groupby = temp_groupby[["feature", "sample_class", "sample_class_label", "sample_count", "min_value", "max_value",
                                 "non_event_count", "non_event_rate", "event_count", "event_rate"]]
    
    """
    **********get distribution of good and bad
    """
    temp_groupby['distbn_non_event'] = temp_groupby["non_event_count"]/temp_groupby["non_event_count"].sum()
    temp_groupby['distbn_event'] = temp_groupby["event_count"]/temp_groupby["event_count"].sum()

    temp_groupby['woe'] = np.log(temp_groupby['distbn_non_event'] / temp_groupby['distbn_event'])
    temp_groupby['iv'] = (temp_groupby['distbn_non_event'] - temp_groupby['distbn_event']) * temp_groupby['woe']
    
    temp_groupby["woe"] = temp_groupby["woe"].replace([np.inf,-np.inf],0)
    temp_groupby["iv"] = temp_groupby["iv"].replace([np.inf,-np.inf],0)
    
    return temp_groupby

"""
- iterate over all features.
- calculate WOE & IV for there classes.
- append to one DataFrame woe_iv.
"""
def var_iter(data, target_col, max_bins):
    woe_iv = pd.DataFrame()
    remarks_list = []
    for c_i in data.columns:
        if c_i not in [target_col]:
            # check if binning is required. if yes, then prepare bins and calculate woe and iv.
            """
            ----logic---
            binning is done only when feature is continuous and non-binary.
            Note: Make sure dtype of continuous columns in dataframe is not object.
            """
            c_i_start_time=time.time()
            if np.issubdtype(data[c_i], np.number) and data[c_i].nunique() > 2:
                class_col, remarks, binned_data = prepare_bins(data[[c_i, target_col]].copy(), c_i, target_col, max_bins)
                agg_data = iv_woe_4iter(binned_data.copy(), target_col, class_col)
                remarks_list.append({"feature": c_i, "remarks": remarks})
            else:
                agg_data = iv_woe_4iter(data[[c_i, target_col]].copy(), target_col, c_i)
                remarks_list.append({"feature": c_i, "remarks": "categorical"})
            # print("---{} seconds. c_i: {}----".format(round(time.time() - c_i_start_time, 2), c_i))
            woe_iv = woe_iv._append(agg_data)
    return woe_iv, pd.DataFrame(remarks_list)

# after getting woe and iv for all classes of features calculate aggregated IV values for features.
def get_iv_woe(data, target_col, max_bins):
    func_start_time = time.time()
    woe_iv, binning_remarks = var_iter(data, target_col, max_bins)
    print("------------------IV and WOE calculated for individual groups.------------------")
    print("Total time elapsed: {} minutes".format(round((time.time() - func_start_time) / 60, 3)))
    
    woe_iv["feature"] = woe_iv["feature"].replace("_bins", "", regex=True)    
    woe_iv = woe_iv[["feature", "sample_class", "sample_class_label", "sample_count", "min_value", "max_value",
                     "non_event_count", "non_event_rate", "event_count", "event_rate", 'distbn_non_event',
                     'distbn_event', 'woe', 'iv']]
    
    iv = woe_iv.groupby("feature")[["iv"]].agg(["sum", "count"]).reset_index()
    print("------------------Aggregated IV values for features calculated.------------------")
    print("Total time elapsed: {} minutes".format(round((time.time() - func_start_time) / 60, 3)))
    
    iv.columns = ["feature", "iv", "number_of_classes"]
    null_percent_data=pd.DataFrame(data.isnull().mean()).reset_index()
    null_percent_data.columns=["feature", "feature_null_percent"]
    iv=iv.merge(null_percent_data, on="feature", how="left")
    print("------------------Null percent calculated in features.------------------")
    print("Total time elapsed: {} minutes".format(round((time.time() - func_start_time) / 60, 3)))
    iv = iv.merge(binning_remarks, on="feature", how="left")
    woe_iv = woe_iv.merge(iv[["feature", "iv", "remarks"]].rename(columns={"iv": "iv_sum"}), on="feature", how="left")
    print("------------------Binning remarks added and process is complete.------------------")
    print("Total time elapsed: {} minutes".format(round((time.time() - func_start_time) / 60, 3)))
    
    iv.sort_values(['iv'],ascending=False).to_csv('./iv/iv.csv',index=False)
    woe_iv.to_csv('./iv/woe_iv.csv',index=False)

    return iv, woe_iv.replace({"Missing": np.nan})




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


def train_test_split_(master_data, feats, test_size = 0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(master_data,master_data.loc[:,['convert']],test_size = test_size,random_state=random_state)
    x_train=X_train.loc[:,~X_train.columns.isin(['Acct id','acct_id','convert'])]
    x_train=x_train.loc[:,feats]

    x_test=X_test.loc[:,~X_test.columns.isin(['Acct id','acct_id','convert'])]
    x_test=x_test.loc[:,feats]

    return (x_train, x_test, y_train, y_test, X_train, X_test)

def missing_value_impute(x_train, y_train, x_test, y_test):
    imputer = KNNImputer(n_neighbors=1, weights="uniform")
    x_train_nm=imputer.fit_transform(x_train)
    y_train_nm=imputer.fit_transform(y_train)

    x_test_nm = imputer.fit_transform(x_test)
    y_test_nm=imputer.fit_transform(y_test)

    return (x_train_nm, y_train_nm, x_test_nm, y_test_nm)

def scaling_data(x_train_nm, x_test_nm):
    scaler = MinMaxScaler()
    x_train_nm=scaler.fit_transform(x_train_nm)
    x_test_nm=scaler.fit_transform(x_test_nm)

    return x_train_nm, x_test_nm

def lr_model_hyperparameter_tuning(x_train_nm, y_train_nm, x_test_nm, y_test_nm):
    cs = l1_min_c(x_train_nm, y_train_nm, loss='log') * np.logspace(0, 7, 16)
    cs=cs[:8]

    auc_jar = pd.DataFrame()
    coefs_ = []
    intercept_ = []
    for c in cs:
        print(c)
        clf = LogisticRegression(penalty='l1', solver='liblinear',
                                              tol=1e-6, max_iter=int(1e3),
                                              warm_start=True,l1_ratio=0.2
                                 
                                )
        clf.set_params(C=c)
        clf.fit(x_train_nm,y_train_nm)
        scores_auc = cross_val_score(clf,x_train_nm,y_train_nm,cv=3,scoring='roc_auc',n_jobs=-1)
        scores_recall = cross_val_score(clf,x_train_nm,y_train_nm,cv=3,scoring='recall_macro',n_jobs=-1)
        pred = clf.predict_proba(x_test_nm)
        scores_auc_test=metrics.roc_auc_score(y_test_nm, pred[:,1])
        dict={'Tune':[c],
             'CV_AUC':scores_auc.mean(),
             'Test_AUC':scores_auc_test,
             'CV_Recall':scores_recall.mean(),
             #'CV_Test_Recall':scores_recall_test.mean()
             }
        df_auc = pd.DataFrame(dict)
        auc_jar = auc_jar._append(df_auc)
        coefs_.append(clf.coef_.ravel().copy())
        intercept_.append(clf.intercept_.ravel().copy())

    return auc_jar, coefs_, intercept_

def lr_model_training(auc_jar,x_train,x_train_nm,y_train_nm,x_test_nm, y_test_nm, model_name):
        
    ind = np.where(auc_jar["CV_AUC"]==auc_jar["CV_AUC"].max())    

    auc_jar_df = pd.DataFrame(auc_jar.iloc[ind[0][0],:]).reset_index().rename({'index':'Metric',0:'Value'},axis=1)
    
    clf_fnl = LogisticRegression(penalty='elasticnet', solver='saga',
                                          tol=1e-6, max_iter=int(1e4),
                                          warm_start=True,
                                 l1_ratio = 0.03
                             #intercept_scaling=10000.
                            )

    ###0.01
    
    clf_fnl.set_params(C=auc_jar.iloc[ind[0][0],:]["Tune"])
    #clf_fnl.set_params(C=0.0019199348952386507)
    clf_fnl.fit(x_train_nm, y_train_nm)

    cf_ = clf_fnl.coef_.ravel().copy()
    inter_ = clf_fnl.intercept_.ravel().copy()

    Variable_coeffs=pd.DataFrame(cf_)
    Intercept_coeffs = pd.DataFrame(inter_)
    #Variable_coeffs
    Var_coeffs = pd.DataFrame(cf_,x_train.columns).reset_index()
    Intercept_coeffs = pd.DataFrame(inter_,['Intercept']).reset_index()
    
    Var_coeffs.columns = ['Variables','Coefficients']
    Intercept_coeffs.columns = ['Variables','Coefficients']
    
    Var_coeffs = Var_coeffs.loc[Var_coeffs['Coefficients']!=0,:]
    Intercept_coeffs = Intercept_coeffs.loc[Intercept_coeffs['Coefficients']!=0,:]
    
    
    Var_coeffs.sort_values('Coefficients',ascending=False,inplace=True)
    Var_coeffs=Var_coeffs._append(Intercept_coeffs)

    Var_coeffs['ORE'] = np.exp(Var_coeffs['Coefficients'])

    Var_coeffs.to_csv('./iv/ELR Variable Coefficients.csv',index=False)

    pred_train = clf_fnl.predict_proba(x_train_nm)

    pred_test = clf_fnl.predict_proba(x_test_nm)

    print(auc_jar_df)

    auc_jar_df.to_csv('./ks/auc_jar_df_cv.csv',index=False)
    
    joblib.dump(clf_fnl, model_name)

    return pred_train, pred_test, clf_fnl, auc_jar_df, Var_coeffs


#Calculating KS
def ks(data=None,target=None, prob=None):
    data['target0'] = 1 - data[target]
    data['bucket'] = pd.qcut(data[prob], 10)
    grouped = data.groupby('bucket', as_index = False)
    kstable = pd.DataFrame()
    kstable['min_prob'] = grouped.min()[prob]
    kstable['max_prob'] = grouped.max()[prob]
    kstable['events']   = grouped.sum()[target]
    kstable['nonevents'] = grouped.sum()['target0']
    kstable = kstable.sort_values(by="min_prob", ascending=False).reset_index(drop = True)
    kstable['event_distribution'] = (kstable.events / data[target].sum()).apply('{0:.2%}'.format)
    kstable['nonevent_distribution'] = (kstable.nonevents / data['target0'].sum()).apply('{0:.2%}'.format)
    kstable['cum_eventdist']=(kstable.events / data[target].sum()).cumsum()
    kstable['cum_noneventdist']=(kstable.nonevents / data['target0'].sum()).cumsum()
    kstable['event_rate']=kstable['events']/(kstable['events']+kstable['nonevents'])
    #kstable['cum_eventrate'] = kstable['event_rate'].cumsum()
    kstable['KS'] = np.round(kstable['cum_eventdist']-kstable['cum_noneventdist'], 3) * 100

    #Formating
    kstable['cum_eventdist']= kstable['cum_eventdist'].apply('{0:.2%}'.format)
    kstable['cum_noneventdist']= kstable['cum_noneventdist'].apply('{0:.2%}'.format)
    kstable.index = range(1,11)
    kstable.index.rename('Decile', inplace=True)
    pd.set_option('display.max_columns', 9)
    #print(kstable)
    
    #Display KS
    #from colorama import Fore
    #print(Fore.RED + "KS is " + str(max(kstable['KS']))+"%"+ " at decile " + str((kstable.index[kstable['KS']==max(kstable['KS'])][0])))
    return(kstable)

def ks_table_out_cutoff(y_train_nm,name,pred):
    df_act_pred = pd.DataFrame(data={'Actual':y_train_nm.ravel(), 'Prob':pred[:,1]})
    #df_act_pred = pd.DataFrame(data={'Actual':y_train_nm.ravel(), 'Prob':pred[:,1]})
    #KS Test
    mydf = ks(data=df_act_pred,target="Actual", prob="Prob")
    print("KS - ",max(mydf['KS']),"-",mydf.index[mydf['KS']==max(mydf['KS'])])

    first_cutoff = mydf.loc[mydf.KS==mydf['KS'].max(),['min_prob']].iloc[0,0]
    
    print(f'''First Cutoff - {mydf.loc[mydf.KS==mydf['KS'].max(),['min_prob']].iloc[0,0]}''')

    
    second_cutoff = mydf.iloc[(mydf.index[mydf['KS']==max(mydf['KS'])]+5),:]['min_prob'].to_list()[0]
    
    print(f'''Second Cutoff - {second_cutoff}''')

    cutoffs = pd.DataFrame({'first_cutoff':[first_cutoff],
                 'second_cutoff':[second_cutoff]})
    #mydf.loc[mydf.index[mydf['KS']==max(mydf['KS'])],['min_prob]]
    print(mydf)
    mydf.to_csv(f'''./ks/{name}''')
    cutoffs.to_csv(f'''./ks/{name.replace(".csv","")}_cutoffs.csv''',index=False)
    
    return mydf, cutoffs


def write_predictions(x_train_nm, x_test_nm, model_name,X_train, X_test,cutoffs):
    clf_fnl = joblib.load(model_name)

    pred_train = clf_fnl.predict_proba(x_train_nm)
    pred_test = clf_fnl.predict_proba(x_test_nm)

    
    first_cutoffs = cutoffs.iloc[0,0]
    second_cutoffs = cutoffs.iloc[0,1]
    
    
    X_train['score_convert_lr'] = pred_train[:,1]
    X_test['score_convert_lr'] =pred_test[:,1]

    X_train['prediction_category'] = np.where(X_train['score_convert_lr']>=first_cutoffs,2,
                                              np.where(X_train['score_convert_lr']>=second_cutoffs,1,
                                                      0))
    X_test['prediction_category'] = np.where(X_test['score_convert_lr']>=first_cutoffs,2,
                                          np.where(X_test['score_convert_lr']>=second_cutoffs,1,
                                                  0))

    X_train.to_csv('./score/train_sample_score_lr.csv',index=False)
    X_test.to_csv('./score/test_sample_score_lr.csv',index=False)

    return X_train, X_test



if __name__=="__main__":
    try:
        acct=pd.read_excel("./raw_data/account_attributes (1).xlsx")
        usage=pd.read_excel("./raw_data/account_usage (1).xlsx")
        master_data=data_prep_iv_import(usage,acct)
        iv, woe_iv = get_iv_woe(master_data.loc[:,~master_data.columns.isin(['Acct id'])].copy(), target_col="convert", max_bins=20)

        feats = iv_column_filter(iv)
        x_train, x_test, y_train, y_test, X_train, X_test = train_test_split_(master_data, feats, test_size = 0.2, random_state=42)

        x_train_nm, y_train_nm, x_test_nm, y_test_nm = missing_value_impute(x_train, y_train, x_test, y_test)
        x_train_nm, x_test_nm = scaling_data(x_train_nm, x_test_nm)

        auc_jar, _, _ = lr_model_hyperparameter_tuning(x_train_nm, y_train_nm, x_test_nm, y_test_nm)

        pred_train, pred_test, clf_fnl, auc_jar_df, Var_coeffs = lr_model_training(auc_jar,x_train,x_train_nm,y_train_nm,x_test_nm, y_test_nm, 'logistic_lasso.pkl')

        mydf_train,cutoffs=ks_table_out_cutoff(y_train_nm,name = 'KS Table train.csv', pred = pred_train,)
        mydf_test,_=ks_table_out_cutoff(y_test_nm,name = 'KS Table test.csv', pred = pred_test,)

        train_sample_score_lr,test_sample_score_lr = write_predictions(x_train_nm, x_test_nm, 'logistic_lasso.pkl',X_train, X_test,cutoffs)
    except Exception as e:
        raise CustomException(e, sys)