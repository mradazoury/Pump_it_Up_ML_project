import pandas as pd
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