import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
# from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

np.set_printoptions(suppress=True)
pd.set_option('mode.chained_assignment', None)

class DataPreprocessing():

    def __init__(self,target,features,df,date_period_column_name):

        self.df = df
        self.target = target
        self.features = features
        self.date_period_column_name = date_period_column_name



    # def merge_categorial_with_continuous_features(self):
    #
    #    # number_of_categorial_variables = len(X_categorial_columns)
    #
    #     label_encoder = LabelEncoder()
    #     if len(self.X_categorial_columns) >=1:
    #         for column in self.X_categorial_columns:
    #             self.df[column] = label_encoder.fit_transform(self.df[column])


    def drop_na(self,categorial_features=[],combined_categorial_features = {}):
        all_columns =         self.features.copy()
        all_columns.append(self.target)
        all_columns = self.df.columns
        for column in all_columns:
            if 'standardized_' in column and '_expected' not in column :
                column_name = column.replace("standardized_","")
                all_columns.append(column_name)


        for categorial_feature in categorial_features:
            if categorial_feature not in all_columns:
                all_columns.append(categorial_feature)

            if isinstance(categorial_features,dict) is True:
                other_column_name = categorial_features[categorial_feature]
                if other_column_name != None:
                    all_columns.append(other_column_name)

        for key,  comb_categorial_features in combined_categorial_features.items():
            for comb_feature in comb_categorial_features:
                if comb_feature not in all_columns:
                    all_columns.append(comb_feature)

       # if self.date_period_column_name  in self.df:
        #    all_columns.append(self.date_period_column_name )
        droppedna_df = self.df[   all_columns ].dropna()
        return droppedna_df



    def create_train_test_data(self,droppedna_df,train_data_gets_all_data=False,test_data_min_date=None,test_data_max_date=None ):
        if test_data_max_date != None and test_data_min_date != None:
            test_df = droppedna_df[droppedna_df[self.date_period_column_name ].between( test_data_min_date, test_data_max_date)]
            indexes = test_df.index.tolist()
            train_df = droppedna_df.loc[~droppedna_df.index.isin(indexes)]
        elif test_data_max_date != None:
            test_df = droppedna_df[
                droppedna_df[self.date_period_column_name]< test_data_max_date]
            indexes = test_df.index.tolist()
            train_df = droppedna_df.loc[~droppedna_df.index.isin(indexes)]

        elif test_data_min_date != None:
            test_df = droppedna_df[
                droppedna_df[self.date_period_column_name] < test_data_max_date]
            indexes = test_df.index.tolist()
            train_df = droppedna_df.loc[~droppedna_df.index.isin(indexes)]

        elif train_data_gets_all_data is True:
            test_df = droppedna_df
            train_df = droppedna_df
        else:
            train_df,test_df = train_test_split(droppedna_df, test_size=0.33, random_state=42)

        return train_df,test_df


    # def convert_to_multi_columns(self,column_name):
    #     unique_values = self.df[column_name].unique().tolist()
    #     new_columns = []
    #     for unique_value in unique_values:
    #         str_unique_value = str(unique_value)
    #         self.df[str_unique_value] = 0
    #         self.df.loc[
    #             (self.df[column_name] == unique_value)
    #             , str_unique_value] = 1
    #
    #         new_columns .append(   str_unique_value)
    #
    #     return  self.remove_lowest_frequency(new_columns)
    #
    #
    # def remove_lowest_frequency(self,columns):
    #     min_value_for_drop = max(29,len(self.df)/3000)
    #     removed_columns = []
    #     for column in columns:
    #         sum = self.df[column].sum()
    #         if sum <=   min_value_for_drop:
    #             removed_columns.append(column)
    #             pass
    #
    #     for removed_column in removed_columns:
    #         columns.remove(removed_column)
    #
    #     return columns


    def standardize_pandas_series(self,X,scaler):
        mean = scaler.mean_
        std = scaler.scale_
        return (X-mean)/std

    # def pca_features(self,X):
    #     pca = PCA(n_components=2)
    #     pca.fit(X)
    #     self.standardized_pca_X = pca.transform(X)
    #     return  self.standardized_pca_X

    def scale_data(self,test_df,train_df,categorial_features):
        standardized_features = []
        scaler =preprocessing.StandardScaler().fit(train_df[self.features])
        scaled_X = self.standardize_pandas_series(train_df[self.features],scaler)

        for column in scaled_X.columns:
            train_df[column+'_standardized'] = scaled_X[column]
            standardized_features.append(column+'_standardized')

        standardized_train_X =pd.DataFrame( scaled_X,columns=self.features)

        if len(self.features) >=1:
            for column in self.features:

                standardized_train_X[column ] = train_df[column ].to_numpy()


        if len(test_df) >=1:
            scaled = self.standardize_pandas_series(test_df[self.features],scaler)
            standardized_test_X = pd.DataFrame(scaled,columns=self.features)

            for column in standardized_test_X.columns:

                test_df[column + '_standardized'] =  standardized_test_X[column]

        return test_df,train_df,scaler





