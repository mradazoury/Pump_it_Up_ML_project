import pandas as pd
import numpy as np
import math as m
import matplotlib as plt
import seaborn as sns
import sklearn as skl
import warnings
import statsmodels.api as sm

from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict, ShuffleSplit, validation_curve, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer, RobustScaler, LabelEncoder, scale, MinMaxScaler, PolynomialFeatures
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.feature_selection import RFE,SelectFromModel


from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.datasets import make_classification

from xgboost import XGBClassifier 

from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math as m
import requests

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter('ignore')


import datetime
now = datetime.datetime.now()

"""
Copies `status_group` from train_labels to train_data

Usage:

train_data = addLabelToTrainData(train_data, train_labels)
"""
def addLabelToTrainData(train_data, train_labels):
    labels = train_labels[['status_group']]
    train_data = train_data.join(labels)
    assert ('status_group' in train_data.columns)
    print("`status_group` added to train_data \n")
    return train_data

"""
Drop id and recorded from train[or]test datset

Usage:
train_data = prepareCols(train_data)
test_data = prepareCols(test_data)
"""
def prepareCols(data):
    toDrop = ['id', 'recorded_by', 'scheme_name', 'wpt_name', 'waterpoint_type_group']
    data = data.drop(columns=toDrop)
    assert (not bool(set(toDrop) & set(data.columns)))
    print("{} removed from dataset \n".format(toDrop))
    return data

"""
installer - shortlist of the 5 higher and category other, note that nulls will be included in this criteria.

Usage:
dataset = shortlist_installer(dataset)
"""
def shortlist_installer(dataset):
    def installer_replace(x):
        if x in list(['DWE', 'Government','RWE','Commu','DANIDA']):
            return x
        else:
            return 'other'

    dataset.installer = dataset.installer.map(installer_replace)
    assert(set(dataset.installer.unique()) == {'Commu', 'DANIDA', 'DWE', 'Government', 'RWE', 'other'})
    print("`installer` shortlisted to {'Commu', 'DANIDA', 'DWE', 'Government', 'RWE', 'other'} only \n")
    return dataset

"""
funder - shortlist of the 5 higher and category other, note that nulls will be included in this criteria.

Usage:
dataset = shortlist_funder(dataset)
"""
def shortlist_funder(dataset):
    def funder_replace(x):
        if x in list(['Government Of Tanzania','Danida','Hesawa','Rwssp','World Bank','Kkkt','World Vision','Unicef','Tasaf','District Council']):
            return x
        else:
            return 'other'

    dataset.funder = dataset.funder.map(funder_replace)
    assert(set(dataset.funder.unique()) == {'Government Of Tanzania','Danida','Hesawa','Rwssp','World Bank','Kkkt','World Vision','Unicef','Tasaf','District Council', 'other'})
    print("`funder` shortlisted to {'Government Of Tanzania','Danida','Hesawa','Rwssp','World Bank','Kkkt','World Vision','Unicef','Tasaf','District Council', 'other'} only \n")
    return dataset


"""
lga - shortlist of the 5 higher and category other, note that nulls will be included in this criteria.

Usage:
dataset = shortlist_lga(dataset)
"""
def shortlist_lga(dataset):
    def lga_replace(x):
        if x in list(['Njombe','Arusha Rural','Moshi Rural','Bariadi','Rungwe','Kilosa','Kasulu','Mbozi','Meru','Bagamoyo']):
            return x
        else:
            return 'other'

    dataset.lga = dataset.lga.map(lga_replace)
    assert(set(dataset.lga.unique()) == {'Njombe','Arusha Rural','Moshi Rural','Bariadi','Rungwe','Kilosa','Kasulu','Mbozi','Meru','Bagamoyo', 'other'})
    print("`lga` shortlisted to {'Njombe','Arusha Rural','Moshi Rural','Bariadi','Rungwe','Kilosa','Kasulu','Mbozi','Meru','Bagamoyo', 'other'} only \n")
    return dataset

"""
extraction_type - shortlist of the 8 higher and category other, note that nulls will be included in this criteria.

Usage:
dataset = shortlist_extraction_type(dataset)
"""
def shortlist_extraction_type(dataset):
    def extraction_type_replace(x):
        if x in list(['gravity','nira/tanira','submersible','swn 80','mono','india mark ii','afridev','ksb']):
            return x
        else:
            return 'other'
        
    dataset.extraction_type = dataset.extraction_type.map(extraction_type_replace)

    assert(set(dataset.extraction_type.unique()) == {'gravity','nira/tanira','submersible','swn 80','mono','india mark ii','afridev','ksb','other'})
    print("`extraction_type` shortlisted to {'gravity','nira/tanira','submersible','swn 80','mono','india mark ii','afridev','ksb', 'other'} only \n")
    return dataset
    

"""
scheme_management - shortlist of the 8 higher and category other, note that nulls will be included in this criteria.

Usage:
dataset = shortlist_scheme_management(dataset)
"""
def shortlist_scheme_management(dataset):
    def scheme_management_replace(x):
        if x in list(['VWC','WUG','Water authority','WUA','Water Board','Parastatal','Private operator','Company']):
            return x
        else:
            return 'other'

    dataset.scheme_management = dataset.scheme_management.map(scheme_management_replace)
    
    assert(set(dataset.scheme_management.unique()) == {'VWC','WUG','Water authority','WUA','Water Board','Parastatal','Private operator','Company', 'other'})
    print("`scheme_management` shortlisted to {'VWC','WUG','Water authority','WUA','Water Board','Parastatal','Private operator','Company', 'other'} only \n")
    return dataset

"""
scheme_management - shortlist of the 15 higher and category other, note that nulls will be included in this criteria.

Usage:
dataset = shortlist_region_code(dataset)
"""
def shortlist_region_code(dataset):
    def region_code_replace(x):
        if x in list([11,17,12,3,5,18,19,2,16,10,4,1,13,14,20]):
            return x
        else:
            return 0

    dataset.region_code = dataset.region_code.map(region_code_replace)
    
    assert(set(dataset.region_code.unique()) == {11,17,12,3,5,18,19,2,16,10,4,1,13,14,20, 0})
    print("`region_code` shortlisted to {11,17,12,3,5,18,19,2,16,10,4,1,13,14,20, 'other'} only \n")
    return dataset


"""
construction_year - converts it to years elapsed (AKA age) -- (zeroes ignored)

Usage:
dataset = convert_construction_year(dataset)
"""
import datetime
now = datetime.datetime.now()
def convert_construction_year(dataset):
    def year_convert(x):
        if x != 0:
            return now.year - x
        else:
            return x

    dataset['age'] = dataset.construction_year.map(year_convert)
    dataset = dataset.drop(columns=["construction_year"])
    assert(max(dataset.age) < 100)
    assert (not bool(set(["construction_year"]) & set(dataset.columns)))
    print("`construction_year` converted to `age`, which is elapsed years (zeroes ignored) \n")
    return dataset


"""
date_recorded - converts it to days elapsed

Usage:
dataset = convert_date_recorded(dataset)
"""
def convert_date_recorded(dataset):
    def day_convert(x):
        if x != 0:
            return (now - pd.to_datetime(x)).days
        else:
            return x

    dataset["days_since_recoreded"] = dataset.date_recorded.map(day_convert)
    dataset = dataset.drop(columns=["date_recorded"])
    assert (not bool(set(["date_recorded"]) & set(dataset.columns)))
    print("`date_recorded` converted to `days_since_recoreded`, which is elapsed days (zeroes ignored) \n")
    return dataset

"""
bin_feature - bins a feature with certain ranges

Usage:
dataset = bin_feature(dataset, feature, bins)
"""
def bin_feature(dataset, feature, bins):
    dataset[feature] = pd.cut(dataset[feature] , bins)
    print("`{}` has been binned to {} categories:".format(feature, bins))
    print(dataset[feature].unique().categories)
    print("\n")
    return dataset

"""
Compare the similarity between a series of columns

Usage:
grouping_col = train_data[['basin',
       'subvillage', 'region', 'region_code', 'district_code', 'lga', 'ward','extraction_type',
       'extraction_type_group', 'extraction_type_class', 'management',
       'management_group', 'payment', 'payment_type', 'water_quality',
       'quality_group', 'quantity', 'quantity_group', 'source', 'source_type',
       'source_class', 'waterpoint_type', 'waterpoint_type_group']]

       *All categorical variables with similarities
"""
def cramers_corrected_stat(confusion_matrix):
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

def show_similars(cols, threshold=0.90):
    for i1, col1 in enumerate(cols):
        for i2, col2 in enumerate(cols):
            if (i1<i2):
                cm12 = pd.crosstab(train_data[col1], train_data[col2]).values # contingency table
                cv12 = cramers_corrected_stat(cm12) # Cramer V statistic
                if (cv12 > threshold):
                    print((col1, col2), int(cv12*100))

"""
Plot feature importance for Gradiaent Booster and XGB

Usage:
plot_features(xgb_model, (10,14))

"""

def plot_features(booster, figsize):
    fig, ax = plt.pyplot.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

"""
Impute amount TSH of regions 'Dodoma','Kagera','Mbeya','Tabora' with the mean of the whole pop

Usage:
train_data = amount_tsh_impute_regions(train_data)
"""
def amount_tsh_impute_regions(dataset):
    mean = dataset.amount_tsh.mean()
    dataset = flag_impute(dataset,'population')
    def impute_am(row, mean):
        if float(row['amount_tsh']) == 0 and row['region'] in ['Dodoma', 'Kagera', 'Mbeya', 'Tabora']:
                row['amount_tsh'] = mean
        return row
    dataset = dataset.apply(lambda row: impute_am(row, mean), axis=1)
    print("amount_tsh imputed with mean for regions: ['Dodoma','Kagera','Mbeya','Tabora']")
    return dataset

"""
Impute latitude by the mean of the region

Usage:

train_data = impute_lat(train_data)
"""
def impute_lat(dataset):
    dataset['latitude'] = dataset['latitude'].replace({-0.00000002:np.nan})
    numeric_dtypes = ['int16', 'int32', 'int64', 
                      'float16', 'float32', 'float64']
    lat_mean = dataset['latitude'].mean()
    for i in range(0, len(dataset)): 
        if m.isnan(dataset.latitude[i]) == True:
            for j in ("subvillage", "ward", "lga", "district_code", "region", "basin"):
                if m.isnan(dataset.latitude[dataset[j] == dataset[j].iloc[i]].mean()) == False:
                    dataset.latitude.iloc[i] = dataset.latitude[dataset[j] == dataset[j].iloc[i]].mean()
                    break
                elif j == "basin":
                    dataset.latitude.iloc[i] = lat_mean
    print("latitude imputed with mean")
    return dataset

def fix_latitude(dataset):
    for i in range(0, len(dataset)):
        if dataset.latitude[i] == -0.00000002:
            dataset.latitude[i] = dataset.latitude[dataset['region']==dataset.region[i]].mean()
    print("latitude imputed with mean")
    return dataset

"""
Impute Longitude by the mean of the region

Usage:
train_data = impute_long(train_data)
"""
def impute_long(dataset):
    dataset = flag_impute(dataset,'longitude')
    dataset['longitude'] = dataset['longitude'].replace({0:np.nan})
    numeric_dtypes = ['int16', 'int32', 'int64', 
                      'float16', 'float32', 'float64']
    long_mean = dataset['longitude'].mean()
    for i in range(0, len(dataset)): 
        if m.isnan(dataset.longitude[i]) == True:
            for j in ("subvillage", "ward", "lga", "district_code", "region", "basin"):
                if m.isnan(dataset.longitude[dataset[j] == dataset[j].iloc[i]].mean()) == False:
                    dataset.longitude.iloc[i] = dataset.longitude[dataset[j] == dataset[j].iloc[i]].mean()
                    break
                elif j == "basin":
                    dataset.longitude.iloc[i] = long_mean
    print("longitude imputed with mean")
    return dataset

"""
Impute population by the mean of the population

Usage:
train_data = impute_pop(train_data)
"""
def impute_pop(dataset):
    dataset = flag_impute(dataset,'population')
    dataset['population'] = dataset['population'].replace({0:np.nan})
    numeric_dtypes = ['int16', 'int32', 'int64', 
                      'float16', 'float32', 'float64']
    mean = dataset['population'].mean()
    for i in range(0, len(dataset)): 
        if m.isnan(dataset.population[i]) == True:
            for j in ("subvillage", "ward", "lga", "district_code", "region", "basin"):
                if m.isnan(dataset.population[dataset[j] == dataset[j].iloc[i]].mean()) == False:
                    dataset.population.iloc[i] = dataset.population[dataset[j] == dataset[j].iloc[i]].mean()
                    break
                elif j == "basin":
                    dataset.population.iloc[i] = mean
    print("population imputed with mean")
    return dataset

"""
Impute construction year with the max of median, or mean, or date recorded in (years)

Usage:
train_data = impute_construction_year(train_data)
"""
def impute_construction_year(dataset):
    median = np.median(dataset['age'][dataset['age'] != 0])
    mean = np.mean(dataset['age'][dataset['age'] != 0])

    def impute_age(row, mean, median):
        if row['age'] == 0 :
                row['age'] = max(mean, median, row['days_since_recoreded'] / 365 )
        return row

    dataset = dataset.apply(lambda row: impute_age(row, mean, median), axis=1)
    assert(len(dataset[dataset['age'] == 0]) == 0)
    print("age imputed mean")
    return dataset

def fix_longitude(dataset):
    for i in range(0, len(dataset)):
        if dataset.longitude[i] == 0:
            dataset.longitude[i] = dataset.longitude[dataset['region']==dataset.region[i]].mean()
    return dataset

"""
#Add a column from density which is the result from dividing population from region density (external source)

Usage:
train_data = density(train_data)
"""
def density(dataset):
    tanz_pop = pd.read_csv("other_datasets/Tanzania_pop.csv", delimiter=';')
    dataset['region_pop'] = dataset['region'].map(tanz_pop.set_index('Region')['population'])
    dataset.density = dataset['population'] / dataset['region_pop']
    del dataset['region_pop']
    return dataset

"""
Usage:

feature_skewness(test_temp)
feature_skewness(train_temp)

test_temp = fix_skewness(test_temp)
train_temp = fix_skewness(train_temp)
"""
def feature_skewness(df):
    numeric_dtypes = ['int16', 'int32', 'int64', 
                      'float16', 'float32', 'float64']
    numeric_features = []
    for i in df.columns:
        if df[i].dtype in numeric_dtypes: 
            numeric_features.append(i)

    feature_skew = df[numeric_features].apply(
        lambda x: skew(x)).sort_values(ascending=False)
    skews = pd.DataFrame({'skew':feature_skew})
    return feature_skew, numeric_features

def fix_skewness(df):
    feature_skew, numeric_features = feature_skewness(df)
    high_skew = feature_skew[feature_skew > 0.75]
    skew_index = high_skew.index
    
    for i in skew_index:
        df[i] = boxcox1p(df[i], boxcox_normmax(df[i]+1))

    skew_features = df[numeric_features].apply(
        lambda x: skew(x)).sort_values(ascending=False)
    skews = pd.DataFrame({'skew':skew_features})
    return df

"""
#Add a column from tupil_teacher_ratio gotteen from teacher_data

Usage:
train_data = adding_PTR(train_data)
"""

def adding_PTR(train_data):
    teacher_ratio = pd.read_csv("other_datasets/teacher_data.csv", delimiter=';')

    teacher_ratio_ward = teacher_ratio.groupby(['WARD']).mean()
    teacher_ratio_ward = teacher_ratio_ward.reset_index()
    teacher_ratio_region = teacher_ratio.groupby(['REGION']).mean()
    teacher_ratio_region = teacher_ratio_region.reset_index()

    train_data['region'] = train_data['region'].str.lower()
    teacher_ratio_region['REGION'] = teacher_ratio_region['REGION'].str.lower()

    train_data.insert(2, 'PTR', train_data['ward'].map(teacher_ratio_ward.set_index('WARD')['PTR']))
    for i in range(0, len(train_data)): 
        if m.isnan(train_data.PTR[i]) == True: 
            train_data.PTR[i] =teacher_ratio_region.PTR[ teacher_ratio_region.REGION == train_data.region[i]]
    return train_data

"""
Outputs a csv file with the predictions ready for submission 
Usage:
submission(model)
"""
def submission(model):

         predictions = model.predict(test_set)

         data = {'ID': test_id, 'status_group': predictions}

         submit = pd.DataFrame(data=data)

         vals_to_replace = {1:'functional',2:'non functional',3:'functional needs repair'}

         submit.status_group = submit.status_group.replace(vals_to_replace)        

         submit.to_csv('predictions/pump_predictions.csv', index=False)

"""
Calculating distance between lat long and lat long of capital
Usage:
train_data = distance_capital(train_data) 
"""
def distance_capital(dataset):
    tanz_capital= 6.1630, 35.7516
    def haversine(coord1, coord2):
        R = 6372800  # Earth radius in meters
        lat1, lon1 = coord1
        lat2, lon2 = coord2

        phi1, phi2 = m.radians(lat1), m.radians(lat2) 
        dphi       = m.radians(lat2 - lat1)
        dlambda    = m.radians(lon2 - lon1)

        a = m.sin(dphi/2)**2 + \
            m.cos(phi1)*m.cos(phi2)*m.sin(dlambda/2)**2

        return 2*R*m.atan2(m.sqrt(a), m.sqrt(1 - a))
    for i in range(0, len(dataset)): 
        x = dataset.latitude[i], dataset.longitude[i]
        dataset['distance'] = haversine(tanz_capital, x)
    print("added distance to capital")
    return dataset

def flag_impute(df,column):
    ## This function will add a flagg_column column that flags the 0 before they are imputed
    flag = 'Flag_' + column
    df[flag ] = 0
    df[flag][df[column] == 0] = 1
    return df


