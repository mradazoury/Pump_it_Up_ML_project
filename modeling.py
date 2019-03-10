import pandas as pd
from gplearn.genetic import SymbolicTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.linear_model import SGDRegressor

random_seed = 6666

def numerical_features(df):
    columns = df.columns
    return df._get_numeric_data().columns

def categorical_features(df):
    numerical_columns = numerical_features(df)
    return(list(set(df.columns) - set(numerical_columns)))

def onehot_encode(df):
    numericals = df.get(numerical_features(df))
    new_df = numericals.copy()

    print("# of features before: {}".format(len(df.columns)))

    for categorical_column in categorical_features(df):
        new_df = pd.concat([new_df, 
                            pd.get_dummies(df[categorical_column], 
                                           prefix=categorical_column)], 
                           axis=1)
    print("# of features after: {}".format(len(new_df.columns)))
    return new_df

## Genetic programming function that will create new features
def Genetic_P(dataset,target):
    y = target.copy()
    X = numerical_features(dataset)
    
    function_set = ['add', 'sub', 'mul', 'div',
                'sqrt', 'log', 'abs', 'neg', 'inv',
                'max', 'min','sin',
                 'cos',
                 'tan']
    gp = SymbolicTransformer(generations=20, population_size=2000,
                         hall_of_fame=100, n_components=15,
                         function_set=function_set,
                         parsimony_coefficient=0.0005,
                         max_samples=0.9, verbose=1,
                         random_state=random_seed, n_jobs=-1)
    gp_features = gp.fit_transform(X,y)
    print('Number of features created out of genetic programing: {}'.format(gp_features.shape))
    n = pd.DataFrame(gp_features)
    n =n.set_index(dataset.index.values)
    new_X = pd.concat([dataset, n],axis=1)
    new_X = new_X.dropna()
    return new_X



def label_encoder(df):
    categorical = categorical_features(df)
    # Creating the label encoder object
    le =  LabelEncoder()
    
    # Iterating over the "object" variables to transform the categories into numbers 
    for col in categorical:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

