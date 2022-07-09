# Introduction to Data Science with Python

# In this exercise you should implement a classification pipeline which aim at predicting the amount of hours
# a worker will be absent from work based on the worker characteristics and the work day missed.
# Download the dataset from the course website, which is provided as a .csv file. The target label is 'TimeOff'.
# You are free to use as many loops as you like, and any library functions from numpy, pandas and sklearn, etc...import numpy as np

import numpy as np
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

import os
import glob

def load_dataset(train_csv_path):
    data = pd.read_csv(train_csv_path, sep=',')
    return data

def load_dataset(train_csv_path):
    data = pd.read_csv(train_csv_path, sep=',')
    return data

class DataPreprocessor(object):

    """ 
    This class is a mandatory API. More about its structure - few lines below.

    The purpose of this class is to unify data preprocessing step between the training and the testing stages. 
    This may include, but not limited to, the following transformations:
    1. Filling missing (NA / nan) values
    2. Dropping non descriptive columns
    3 ...

    The test data is unavailable when building the ML pipeline, thus it is necessary to determine the 
    preprocessing steps and values on the train set and apply them on the test set.


    *** Mandatory structure ***
    The ***fields*** are ***not*** mandatory
    The ***methods***  - "fit" and "transform" - are ***required***.

    You're more than welcome to use sklearn.pipeline for the "heavy lifting" of the preprocessing tasks, but it is not an obligation. 
    Any class that implements the methods "fit" and "transform", with the required inputs & outps will be accepted. 
    Even if "fit" performs no tasks at all.
    """

    def __init__(self):
      self.transformer:Pipeline = None

    def fit(self, dataset_df):

        """
        Input:
        dataset_df: the training data loaded from the csv file as a dataframe containing only the features (not the target - see the main function).

        Output:
        None

        Functionality:
        Based on all the provided training data, this method learns with which values to fill the NA's, 
        how to scale the features, how to encode categorical variables etc.

        *** This method will be called exactly once during evaluation. See the main section for details ***

        Note that implementation below is a boilerplate code which performs very basic categorical and numerical fields preprocessing.
        """
        dataset_df.apply(lambda col: col.fillna(col.mode()[0], inplace=True) if col.dtype == 'O' else col)
        # This section can be hard-coded
        numerical_columns = ['Height', 'Weight', 'Season', 'Transportation expense', 'Reason', 'Service time', 'Day', 'Residence Distance']
        remove_numerical_columns, remove_categorical_columns = ['Pet', 'Son', 'Month', 'ID'], ['Education']
        categorical_columns = list(set(dataset_df.columns) - set(numerical_columns) - set(remove_numerical_columns) - set(remove_categorical_columns))

        # Handling Numerical Fields
        num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median"))])

        # Handling Categorical Fields
        categorical_transformer = OneHotEncoder(drop=None, sparse=False, handle_unknown='ignore')
        cat_pipeline = Pipeline([
          ('1hot', categorical_transformer)
        ])

        preprocessor = ColumnTransformer(
          transformers=[
            ("dropId", 'drop','ID'),
            ("num", num_pipeline, numerical_columns),
            ("cat", cat_pipeline, categorical_columns),
          ]
        )

        self.transformer = Pipeline(steps=[
          ("preprocessor", preprocessor), ('scaler', StandardScaler())
        ])

        self.transformer.fit(dataset_df)

    def transform(self, df):
    
        """
        Input:
        df:  *any* data similarly structured to the train data (dataset_df input of "fit")

        Output: 
        A processed dataframe or ndarray containing only the input features (X).

        It should maintain the same row order as the input.
        Note that the labels vector (y) should not exist in the returned ndarray object or dataframe.

        Functionality:
        Based on the information learned in the "fit" method, apply the required transformations to the passed data (df)

        """

        return self.transformer.transform(df)

def train_model(processed_X, y):
      
    """
    This function gets the data after the pre-processing stage  - after running DataPreprocessor.transform on it, 
    a vector of labels, and returns a trained model. 

    Input:
    processed_X (ndarray or dataframe): the data after the pre-processing stage
    y: a vector of labels

    Output:
    model: an object with a "predict" method, which accepts the ***pre-processed*** data and outputs the prediction
    """
    

    model = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid={'splitter': ['best', 'random'], 'criterion': ['gini', 'entropy', 'log_loss'], 'max_depth': [2, 4, 6, 8, 10, 12]}, scoring='average_precision')
    model.fit(processed_X, y)
    return model.best_estimator_
  
def get_csv_file(): # Assuming only one csv file exists
    path = os.getcwd()
    extension = 'csv'
    os.chdir(path)
    csv_file = glob.glob('*.{}'.format(extension))
    return csv_file[0]

if __name__ == '__main__':
    preprocessor = DataPreprocessor()
    train_csv_path = get_csv_file()
    train_dataset_df = load_dataset(train_csv_path)

    X_train = train_dataset_df.iloc[:, :-1]
    y_train = train_dataset_df['TimeOff']
    preprocessor.fit(X_train)
    model = train_model(preprocessor.transform(X_train), y_train)

    ### Evaluation Section ####
    # test_csv_path = 'time_off_data_test.csv'
    # Obviously, this will be different during evaluation. For now, you can keep it to validate proper execution
    test_csv_path = train_csv_path
    test_dataset_df = load_dataset(test_csv_path)

    X_test = test_dataset_df.iloc[:, :-1]
    y_test = test_dataset_df['TimeOff']

    processed_X_test = preprocessor.transform(X_test)
    predictions = model.predict(processed_X_test)
    test_score = accuracy_score(y_test, predictions)
    print("test:", test_score)

    predictions = model.predict(preprocessor.transform(X_train))
    test_score = accuracy_score(y_train, predictions)
    print('train:', test_score)