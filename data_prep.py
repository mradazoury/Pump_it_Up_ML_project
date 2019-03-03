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
    data = data.drop(columns=['id', 'recorded_by', 'scheme_name', 'ward', 'wpt_name', 'subvillage'])
    assert (not bool(set(['id', 'recorded_by', 'scheme_name', 'ward', 'wpt_name', 'subvillage']) & set(data.columns)))
    print("['id', 'recorded_by', 'scheme_name', 'ward', 'wpt_name', 'subvillage'] removed from dataset \n")
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

    dataset.construction_year = dataset.construction_year.map(year_convert)
    assert(max(dataset.construction_year) < 100)
    print("`construction_year` converted to elapsed years (zeroes ignored) \n")
    return dataset