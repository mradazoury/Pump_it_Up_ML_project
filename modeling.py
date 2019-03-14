import pandas as pd
from gplearn.genetic import SymbolicTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import LabelBinarizer, RobustScaler, LabelEncoder, scale, MinMaxScaler, PolynomialFeatures
from vecstack import stacking
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


def test_score( dataset , name='test',train_id = False ):
    
    ### PLease specify the name!!!
    K = KFold(5, random_state  = random_seed)
    
    to_int = {'functional':1,'non functional':2,'functional needs repair':3}
    to_cat = {1:'functional',2:'non functional',3:'functional needs repair'}

    y_Train = dataset['status_group'].loc[(dataset['is_test'].isin([0]) )]
    y_Train   = y_Train.replace(to_int).copy()
    dataset = dataset.drop('status_group',axis=1).copy()
    
    ### Label encode
    dataset = label_encoder(dataset).copy()
    ### Divide test and train 
    X_Train= dataset.loc[(dataset['is_test'].isin([0]) )].drop('is_test',axis=1)
    test = dataset.loc[(dataset['is_test'].isin([1]) )].drop(['is_test'],axis=1)

    #### Set ID aside for test 
    ID = test['id'] 

    ### If train_id set to Trues the the ID will be kept as feature
    if train_id == False:
        X_Train = X_Train.set_index('id')
        test = test.set_index('id')
        
    #### Random forest with params from a gridsearch
    RFC = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=85, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=2, min_samples_split=5,
            min_weight_fraction_leaf=0.0, n_estimators=230, n_jobs=-1,
            oob_score=False, random_state=6666, verbose=0,
            warm_start=False)
                      
    ###Scores from cross val 
    scores = cross_val_score(RFC, X_Train, y_Train,scoring='accuracy', cv=K)
    print('The average of the cross validation with Random Forest:{}'.format(scores.mean(), scores.std() * 2))
    
    #### Refitting on the whole data 
    RFC.fit(X_Train , y_Train)
    
    ### Predict on our test                 
    predictions = RFC.predict(test)
    
    ### Save the prediction file with the right format to submit 
    data = {'ID': ID, 'status_group': predictions}
    submit = pd.DataFrame(data=data)
    submit['status_group'] = submit.status_group.replace(to_cat)
    submit.to_csv('predictions/'+name+'.csv', index=False)
    
    
    return scores , predictions , FEI
    
