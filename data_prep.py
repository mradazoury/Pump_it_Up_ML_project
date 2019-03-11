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
def dropCols(data):
    toDrop = ['recorded_by', 'scheme_name', 'wpt_name', 'waterpoint_type_group']
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
    assert(max(dataset.age) < 100)
    print("`construction_year` converted to `age`, which is elapsed years (zeroes ignored) \n")
    return dataset

"""
age imputed with mean of rows with same extraction_type

Usage:
dataset = impute_age(dataset)
"""
def impute_age(dataset):
    dataset = flag_impute(dataset,'age')
    mean = dataset['age'].mean()    
    def impute_age(row):
        if row['age'] == 0:
            row['age'] = mean
        return row
    dataset = dataset.apply(impute_age, axis=1)
    assert(len(dataset[dataset['age'] == 0]) == 0)
    print("`age` imputed with mean of rows with same extraction_type  \n")
    return dataset


"""
date_recorded - converts it to days elapsed

Usage:
dataset = convert_date_recorded(dataset)
"""
def convert_date_recorded(dataset):
    dataset = flag_impute(dataset, 'date_recorded')
    dataset["days_since_recoreded"] = 0
    def day_convert(row):
        row["days_since_recoreded"] = (now - pd.to_datetime(row["date_recorded"])).days
        return row
    dataset = dataset.apply(day_convert, axis=1)
    print("`date_recorded` converted to `days_since_recoreded`, which is elapsed days (zeroes ignored) \n")
    return dataset

"""
bin_feature - bins a feature with certain ranges

Usage:
dataset = bin_feature(dataset, feature, bins)
"""
def bin_feature(dataset, feature, bins):
    dataset[feature] = pd.qcut(dataset[feature] , bins)
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
    dataset = flag_impute(dataset,'amount_tsh')
    def impute_am(row, mean):
        if float(row['amount_tsh']) == 0 and row['region'] in ['Dodoma', 'Kagera', 'Mbeya', 'Tabora']:
                row['amount_tsh'] = mean
        return row
    dataset = dataset.apply(lambda row: impute_am(row, mean), axis=1)
    print("amount_tsh imputed with mean for regions: ['Dodoma','Kagera','Mbeya','Tabora']")
    return dataset

"""
Impute column by the forst available mean from subvillage, ward, lga, region, dataset mean

Usage:

train_data = impute_column(train_data, "latitude")
"""
def impute_column(dataset, column):
    subvillage = pd.DataFrame(dataset.groupby('subvillage')[column].mean())
    ward = pd.DataFrame(dataset.groupby('ward')[column].mean())
    lga = pd.DataFrame(dataset.groupby('lga')[column].mean())
    region = pd.DataFrame(dataset.groupby('region')[column].mean())


    lat_mean = dataset[column].mean()


    dataset[column] = dataset[column].replace({-0.00000002:np.nan})

    def firstNonNan(listfloats):
        for item in listfloats:
            if m.isnan(item) == False:
                return item

    def impute(row):
        if m.isnan(row[column]) == True:
            subvillage_lat = subvillage.loc[row['subvillage']][column]
            ward_lat = ward.loc[row['ward']][column]
            lga_lat = lga.loc[row['lga']][column]
            region_lat = region.loc[row['region']][column]

            row[column] = firstNonNan([subvillage_lat , ward_lat , lga_lat, region_lat, lat_mean])
            assert(m.isnan(row[column]) == False)
        return row
    dataset = dataset.apply(impute, axis=1)
    dataset = flag_impute(dataset, column)
    print("{} imputed with mean".format(column))
    return dataset



"""
#Add a column from density which is the result from dividing population from region density (external source)

Usage:
train_data = density(train_data)
"""
def density(dataset):
    dataset['density'] = 0
    tanz_pop = pd.read_csv("other_datasets/Tanzania_pop.csv", delimiter=';')
    dataset['region_pop'] = dataset['region'].map(tanz_pop.set_index('Region')['population'])
    dataset['density'] = dataset['population'] / dataset['region_pop']
    del dataset['region_pop']
    print("added density")
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
def submission(model, test_set):

         predictions = model.predict(test_set)

         data = pd.concat([testIDs[['id']],predictions], axis=1)

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
    dataset['distance'] = 0 
    def haversine(coord1, coord2):
        R = 6371  # Earth radius in kms
        lat1, lon1 = coord1
        lat2, lon2 = coord2

        phi1, phi2 = m.radians(lat1), m.radians(lat2) 
        dphi       = m.radians(lat2 - lat1)
        dlambda    = m.radians(lon2 - lon1)

        a = m.sin(dphi/2)**2 + \
            m.cos(phi1)*m.cos(phi2)*m.sin(dlambda/2)**2

        return 2 * R * m.atan2(m.sqrt(a), m.sqrt(1 - a))

    def addDistance(row):
        if row['latitude'] != 0 and row['longitude'] != 0:
            x = row['latitude'], row['longitude']
            row['distance'] = haversine(tanz_capital, x)
        return row
    dataset = dataset.apply(addDistance, axis=1)
    print("added distance to capital")
    return dataset

def flag_impute(df,column):
    ## This function will add a flagg_column column that flags the 0 before they are imputed
    flag = 'Flag_' + column
    df[flag ] = 0
    df[flag][df[column] == 0] = 1
    return df

"""
Remove outliers for Logistic regression. Must include the target variable inside the dataframe.
Usage:
train_data = remove_outliers(train_data) 
"""
def remove_outliers(df):
    X = df.drop(['status_group'], axis=1)
    y = df.status_group.reset_index(drop=True)
    ols = sm.OLS(endog = y, exog = X)
    fit = ols.fit()
    test = fit.outlier_test()['bonf(p)']
    outliers = list(test[test<1e-3].index) 
    df.drop(df.index[outliers])
    return df

"""
Standarizw all numerical columns to the same scale
Usage:
train_data = scaler(train_data) 
"""
def scaler(dataset):
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)
    dataset = pd.DataFrame(dataset)
    return dataset

"""
Standarizw all numerical columns to the same scale
Usage:
train_data = scaler(train_data) 
"""
def label_encoder(df):
    def numerical_features(df):
        columns = df.columns
        return df._get_numeric_data().columns

    def categorical_features(df):
        numerical_columns = numerical_features(df)
        return(list(set(df.columns) - set(numerical_columns)))
    
    categorical = categorical_features(df)
    # Creating the label encoder object
    le =  LabelEncoder()
    
    # Iterating over the "object" variables to transform the categories into numbers 
    for col in categorical:
        df[col] = le.fit_transform(df[col].astype(str))
    return df
"""
ATTENTION: This might not work when applying on training set, no sure on how to fix it. 

adding a column with cluster numbers
Usage:
train_data = clustering(train_data) 
"""

def clustering(train):

    def cluster(X, labels_true, eps=3, min_samples=30, verbose=False):
        db = DBSCAN(eps=0.1, min_samples=30).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        if verbose is True:
            print('Estimated number of clusters: %d' % n_clusters_)
            print('Estimated number of noise points: %d' % n_noise_)
            print("Homogeneity: %0.3f" %
                  metrics.homogeneity_score(labels_true, labels))
            print("Completeness: %0.3f" %
                  metrics.completeness_score(labels_true, labels))
            print("Adjusted Mutual Information: %0.3f"
                  % metrics.adjusted_mutual_info_score(labels_true, labels))
        return db

    variables = ['longitude', 'latitude']
    db = cluster(train.loc[:, variables], train_labels)
    
    train['cluster'] = db.labels_
    train['cluster'] = train['cluster'].astype('category',copy=False)
    
    return train 
