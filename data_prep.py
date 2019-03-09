import pandas as pd
import numpy as np
import math as m

import datetime
now = datetime.datetime.now()

"""
Copies `status_group` from train_labels to train_data

Usage:

train_data = addLabelToTrainData(train_data, train_labels)
"""
def addLabelToTrainData(train_data, train_labels):
    labels = train_labels.copy().drop(columns='id')
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
    toDrop = ['id', 'recorded_by', 'scheme_name', 'ward', 'wpt_name', 'subvillage', 'waterpoint_type_group']
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
            return 'other'

    dataset.region_code = dataset.region_code.map(region_code_replace)
    
    assert(set(dataset.region_code.unique()) == {11,17,12,3,5,18,19,2,16,10,4,1,13,14,20, 'other'})
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


def reverse_geocode(latlng):
    result = {}
    url = 'https://maps.googleapis.com/maps/api/geocode/json?latlng={}'
    request = url.format(latlng)
    data = requests.get(request).json()
    if len(data['results']) > 0:
        result = data['results'][0]
    return result

def numerical_features(df):
    columns = df.columns
    return df._get_numeric_data().columns

def categorical_features(df):
    numerical_columns = numerical_features(df)
    return(list(set(df.columns) - set(numerical_columns)))

def onehot_encode(df):
    numericals = df.get(numerical_features(df))
    new_df = numericals.copy()
    for categorical_column in categorical_features(df):
        new_df = pd.concat([new_df, 
                            pd.get_dummies(df[categorical_column], 
                                           prefix=categorical_column)], 
                           axis=1)
    return new_df

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
    for i in range(0, len(dataset)):
        if dataset.amount_tsh[i] == 0:
            if dataset.region[i] in ['Dodoma','Kagera','Mbeya','Tabora']:
                dataset.amount_tsh[i] = dataset.amount_tsh.mean()
    return dataset

"""
#Impute latitude by the mean of the region

Usage:

train_data = impute_lat(train_data)
"""
def impute_lat(dataset):
    dataset['latitude'] = dataset['latitude'].replace({-0.00000002:np.nan})
    numeric_dtypes = ['int16', 'int32', 'int64', 
                      'float16', 'float32', 'float64']
    for i in range(0, len(dataset)): 
        if m.isnan(dataset.latitude[i]) == True:
            for j in ("subvillage", "ward", "lga", "district_code", "region", "basin"):
                if m.isnan(dataset.latitude[dataset[j] == dataset[j].iloc[i]].mean()) == False:
                    dataset.latitude.iloc[i] = dataset.latitude[dataset[j] == dataset[j].iloc[i]].mean()
                    break
                elif j == "basin":
                    dataset.latitude.iloc[i] = train_data['latitude'].mean()
    return dataset

def fix_latitude(dataset):
    for i in range(0, len(dataset)):
        if dataset.latitude[i] == -0.00000002:
            dataset.latitude[i] = dataset.latitude[dataset['region']==dataset.region[i]].mean()
    return dataset

"""
#Impute Longitude by the mean of the region

Usage:
train_data = impute_long(train_data)
"""
def impute_long(dataset):
    dataset['longitude'] = dataset['longitude'].replace({0:np.nan})
    numeric_dtypes = ['int16', 'int32', 'int64', 
                      'float16', 'float32', 'float64']
    for i in range(0, len(dataset)): 
        if m.isnan(dataset.longitude[i]) == True:
            for j in ("subvillage", "ward", "lga", "district_code", "region", "basin"):
                if m.isnan(dataset.longitude[dataset[j] == dataset[j].iloc[i]].mean()) == False:
                    dataset.longitude.iloc[i] = dataset.longitude[dataset[j] == dataset[j].iloc[i]].mean()
                    break
                elif j == "basin":
                    dataset.longitude.iloc[i] = train_data['longitude'].mean()
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
    tanz_pop = pd.read_csv("Tanzania_pop.csv", delimiter=';')
    dataset.insert(40,'region_pop', dataset['region'].map(tanz_pop.set_index('Region')['population']))
    dataset['density'] = dataset['population'] / dataset['region_pop']
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

         submit.to_csv('pump_predictions.csv', index=False)