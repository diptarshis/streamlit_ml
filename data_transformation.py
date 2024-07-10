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

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def avg_features(usage):
    Avg_features=usage.groupby(['Acct id'])['Number of link clicks'].agg(['min', 'max','mean','std','median']).reset_index()
    return Avg_features


#L3, L6, Avg
def auto_reg_avg(usage):
    usg=usage.copy()
    usg['RN'] = usg.sort_values(['Date time'], ascending=[False]) \
                 .groupby(['Acct id']) \
                 .cumcount() + 1
    usage_l3=usg.loc[usg.RN<=3,:]
    usage_l3=avg_features(usage_l3)
    usage_l3.columns = [str(i+"_l3")  if "Acct" not in i else str(i) for i in usage_l3.columns]
    
    usage_l6=usg.loc[usg.RN<=6,:]
    usage_l6=avg_features(usage_l6)
    usage_l6.columns = [str(i+"_l6")  if "Acct" not in i else str(i) for i in usage_l6.columns]

    auto_reg_avg_out=pd.merge(pd.merge(usg.loc[:,['Acct id']].drop_duplicates(),usage_l3,left_on = ['Acct id'],right_on = ['Acct id'],how='left'),usage_l6,left_on = ['Acct id'],right_on = ['Acct id'],how='left')

    return auto_reg_avg_out

def percentile_stats_function(usage):
    usg=usage.copy()
    percentile_stats = usage.agg(
    percentile_1 = ("Number of link clicks", lambda x: x.quantile(0.01)),
    percentile_5 = ("Number of link clicks", lambda x: x.quantile(0.05)),
    percentile_10 = ("Number of link clicks", lambda x: x.quantile(0.1)),
    percentile_25 = ("Number of link clicks", lambda x: x.quantile(0.25)),
    percentile_50=("Number of link clicks", lambda x: x.quantile(0.5)),
    percentile_75=("Number of link clicks", lambda x: x.quantile(0.75)),
    percentile_95=("Number of link clicks", lambda x: x.quantile(0.95)),
    percentile_99=("Number of link clicks", lambda x: x.quantile(0.99)),
        ).reset_index()

    for i in range(percentile_stats.shape[0]):
        print(int(percentile_stats.iloc[i][1]))
        print(str(percentile_stats.iloc[i][0]))
        usg[str(percentile_stats.iloc[i][0])+'_flag_gt'] = np.where(usg['Number of link clicks'] > int(percentile_stats.iloc[i][1]),1,0)
        usg[str(percentile_stats.iloc[i][0])+'_flag_lt'] = np.where(usg['Number of link clicks'] < int(percentile_stats.iloc[i][1]),1,0)
        #print((percentile_stats.iloc[i]))

    #.apply(lambda x: x['ID_2'].sum()/len(x))
    nr_flag = usg.groupby(['Acct id'])[[i for i in usg.columns if "_flag" in i]].apply(lambda x: x.sum()/len(x))

    return nr_flag

def trendline(data, order=2):
    index = [i+1 for i in range(len(data))]
    coeffs = np.polyfit(index, list(data), order)
    slope = coeffs[-2]
    return float(slope)


def dwt(x):
    a1,d2,d1 = pywt.wavedec(x,'db1',level=2,mode = 'periodic')
    #A1 - Absolute Coefficient
    #D2 - Detailed Coefficient second order
    #D1 - Detailed Coefficient first order
    mean_a1 = (a1).mean()
    median_a1 = np.median(a1)
    std_a1 = np.std(a1)
    max_a1 = np.max(a1)
    
    mean_d2 = (d2).mean()
    median_d2 = np.median(d2)
    std_d2 = np.std(d2)
    max_d2 = np.max(d2)
    
    mean_d1 = (d1).mean()
    median_d1 = np.median(d1)
    std_d1 = np.std(d1)
    max_d1 = np.max(d1)


    return (mean_a1, median_a1, std_a1, max_a1, mean_d2, median_d2, std_d2, max_d2, mean_d1, median_d1, std_d1, max_d1)


def calculate_dwt(usage):
    dwt_op = usage.groupby(['Acct id'])['Number of link clicks'].apply(dwt).reset_index()
    dwt_op['mean_a1_dwt']=dwt_op['Number of link clicks'].apply(lambda x: x[0])
    dwt_op['median_a1_dwt']=dwt_op['Number of link clicks'].apply(lambda x: x[1])
    dwt_op['std_a1_dwt']=dwt_op['Number of link clicks'].apply(lambda x: x[2])
    dwt_op['max_a1_dwt']=dwt_op['Number of link clicks'].apply(lambda x: x[3])
    
    dwt_op['mean_a2_dwt']=dwt_op['Number of link clicks'].apply(lambda x: x[4])
    dwt_op['median_a2_dwt']=dwt_op['Number of link clicks'].apply(lambda x: x[5])
    dwt_op['std_a2_dwt']=dwt_op['Number of link clicks'].apply(lambda x: x[6])
    dwt_op['max_a2_dwt']=dwt_op['Number of link clicks'].apply(lambda x: x[7])
    
    dwt_op['mean_a2_dwt']=dwt_op['Number of link clicks'].apply(lambda x: x[8])
    dwt_op['median_a2_dwt']=dwt_op['Number of link clicks'].apply(lambda x: x[9])
    dwt_op['std_a2_dwt']=dwt_op['Number of link clicks'].apply(lambda x: x[10])
    dwt_op['max_a2_dwt']=dwt_op['Number of link clicks'].apply(lambda x: x[11])
    
    dwt_op=dwt_op.drop(['Number of link clicks'],axis=1)

    return dwt_op

def ohe_acct(acct):
    acct_=acct.copy()
    
    acct_['activate']=np.where(acct_['Activate chat bot']=='Y',1,0)
    
    acct_fnl=pd.concat([acct_.loc[:,['Acct id']],pd.get_dummies(acct_['Acct type'])*1,acct_.loc[:,['activate','Converted to paid customer']]],axis=1)

    acct_fnl=acct_fnl.rename(columns = {'Converted to paid customer':'convert'})
    
    return acct_fnl

def holiday_matrix(usage,country = 'US'):
    start_date = datetime.strptime(usage.loc[:,['Date time']].min()[0], '%Y-%m-%d').date() - timedelta(365)
    end_date = datetime.strptime(usage.loc[:,['Date time']].max()[0], '%Y-%m-%d').date() + timedelta(365)
    dt_ = [i for i in range(start_date.year,end_date.year+1,1)]
    us_holidays = country_holidays(country, years=dt_)

    hol_names = list(us_holidays.values())
    holiday_dates = [i.strftime('%Y-%m-%d') for i in list(us_holidays.keys())]
    holiday_df = pd.DataFrame(list(zip(holiday_dates,hol_names)),columns=['Date','Name'])

    df_date_cross=pd.merge(usage.loc[:,['Date time']].drop_duplicates(),holiday_df,how = 'cross')
    df_date_cross['diff_days'] = (abs(pd.to_datetime(df_date_cross['Date']) - pd.to_datetime(df_date_cross['Date time']))).dt.days

    df_date_cross1=df_date_cross.loc[df_date_cross.diff_days <= 7,:].sort_values(['Date time'])
    df_date_cross1['Name'] = df_date_cross1['Name'].replace(regex=True, to_replace=r'[^A-Za-z\s]+',value=r'').replace(regex=True, to_replace=r'[\s]+',value=r'_')

    ohe = pd.get_dummies(df_date_cross1['Name'])*1
    
    tst_wide=pd.concat([df_date_cross1.loc[:,['Date time']],ohe],axis=1).groupby(['Date time']).sum().reset_index()

    return tst_wide


def me_model_estimates(usage,holiday_mat_df):
    master_re=pd.merge(usage,holiday_mat_df,left_on = ['Date time'],right_on = ['Date time'],how = 'left').fillna(0)
    accts = master_re['Acct id'].unique().tolist()
    x_var = [i for i in master_re.columns if i not in ['Acct id','Date time','Number of link clicks']]
    y_var = ['Number of link clicks']

    jar_coefs = pd.DataFrame()
    for i in accts:
        print(i)
        X_re = master_re.loc[master_re['Acct id']==str(i),x_var].to_numpy()
        Y_re = master_re.loc[master_re['Acct id']==str(i),y_var].to_numpy()
        lin_model = LinearRegression(fit_intercept=True)
        lin_model = lin_model.fit(X_re, Y_re)
        r2 = r2_score(Y_re, lin_model.predict(X_re))
        coefficients = pd.concat([pd.DataFrame(master_re.loc[master_re['Acct id']==str(i),x_var].columns),pd.DataFrame(np.transpose(lin_model.coef_))], axis = 1)
        coefficients.columns = ['Coefficient','Value']
        coefficients['r2'] = r2
        coefficients['acct_id'] = str(i)
        jar_coefs = jar_coefs._append(coefficients,ignore_index=True)

    
    jar_coefs_ = jar_coefs.pivot(index='acct_id',columns='Coefficient',values='Value')
    jc = jar_coefs_.reset_index(drop=False)
    jc['acct_id'] = jar_coefs_.index
    predictability=jar_coefs.groupby(['acct_id'])['r2'].mean().reset_index()
    fnl_holiday_out=pd.merge(predictability,jc,left_on = ['acct_id'],right_on = ['acct_id'],how = 'left')
    return fnl_holiday_out


def data_prep_main(usage,acct):
    Avg_features = avg_features(usage)

    Avg_features_auto=auto_reg_avg(usage)

    nr_flag=percentile_stats_function(usage)

    trend_=pd.DataFrame(usage.groupby(['Acct id'])['Number of link clicks'].apply(trendline)).reset_index()
    trend_.columns = ['Acct id','trend_slope']

    dwt_op=calculate_dwt(usage)

    acct_fnl = ohe_acct(acct)

    holiday_mat_df=holiday_matrix(usage,country = 'US')

    fnl_holiday_out=me_model_estimates(usage,holiday_mat_df)

    master_data=pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(Avg_features, Avg_features_auto, left_on = ['Acct id'],right_on = ['Acct id'],how='left'),
            nr_flag,left_on = ['Acct id'],right_on = ['Acct id']),trend_,left_on = ['Acct id'],right_on = ['Acct id'],how='left'),
            dwt_op,left_on = ['Acct id'],right_on = ['Acct id'],how = 'left'), 
            fnl_holiday_out,left_on = ['Acct id'],right_on = ['acct_id'],how = 'left'),
            acct_fnl,left_on = ['Acct id'],right_on = ['Acct id'],how = 'left')
         

    return master_data




