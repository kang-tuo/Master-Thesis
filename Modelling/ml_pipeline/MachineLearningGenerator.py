import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, confusion_matrix, accuracy_score, f1_score

from ml_pipeline.multiclassification import GridSearchMultiClass
from Modelling.ml_pipeline.binary import GridSearchBinary
from Modelling.ml_pipeline.regression import GridSearchRegression
from Modelling.ml_pipeline.preprocessing import DataPreprocessing
from Modelling.ml_pipeline.default_model_parameters import *
from sklearn.linear_model import LogisticRegression
from settings import *
import pandas as pd
import pickle
import itertools
from constants import *


# from sklearn.preprocessing import OrdinalEncoder

def get_f1_from_CM(CM):
    # true negatives is 00, false negatives is 10 , true positives is 11  and false positives is 01.
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]

    presion_for_p = TP / (TP + FP)
    recall_for_p = TP / (TP + FN)
    F1_P = 2 * presion_for_p * recall_for_p / (presion_for_p + recall_for_p)

    presion_for_n = TN / (TN + FN)
    recall_for_n = TN / (TN + FP)
    F1_N = 2 * presion_for_n * recall_for_n / (presion_for_n + recall_for_n)

    return F1_P, F1_N


class MachineLearningGenerator():

    def __init__(self,
                 machine_learning_models={LogisticRegression(): None},
                 loss_function='default',
                 ### The machine-learning algorithm will be optimized vased on this loss function. If set to default, it will find the most commonly used one for the specific task.
                 find_optimal_features=False,
                 ## If set to true will iterate over all combinations of features and return the one with higheswt accuracy
                 test_period_min_date=None,
                 ## Minimum date period for test dataset. If test_period_min_date is set to None and test_period_max_date is set to None, test_split will be done randomly.
                 test_period_max_date=None,
                 ## Maximum date period for test dataset. If test_period_min_date is set to None and test_period_max_date is set to None, test_split will be done randomly.
                 date_period_column_name='start_date_time',
                 ### The name of date period. If test dataframe is based on date period, the column name of the dataframe needs to match this name
                 categorical_features=[],  ### features that need to underto categorical transformation.
                 df=None,
                 combined_categorical_features={},
                 ## Will create a unique combination of several categorias and transform the newly created category.
                 # Is written as combined_categorical_features = {0:
                 #                    ['team1','team2'],
                 #                  }
                 categorical_betas=[150],
                 ### A lower beta value will result in less sample size on categorical transformation weighting heigher.
                 # This is a dictionary that will be iterated through in order to find the beta values for each category that returns the highest accuracy.
                 average_categorical_variables={},  ## WiP
                 target_type=None,  ### binary, regression or multiclassification
                 verbose=False,  ## If vebose is True, it will print out the progress of the machine learning model.
                 predict_standardized_target=False,
                 target_name="target"
                 ):

        if isinstance(machine_learning_models, list):
            machine_learning_models = {m: None for m in machine_learning_models}
        elif isinstance(machine_learning_models, dict) is False:
            ## Assumes its a single variable then
            machine_learning_models = {machine_learning_models: None}

        self.df = df
        self.machine_learning_models = machine_learning_models
        self.verbose = verbose
        self.target = target_name
        self.features = []
        self.predict_standardized_target = predict_standardized_target
        self.target_type = target_type
        self.loss_function = loss_function
        self.test_period_min_date = test_period_min_date
        self.test_period_max_date = test_period_max_date
        self.date_period_column_name = date_period_column_name
        self.find_optimal_features = find_optimal_features
        self.standardized_features = []
        self.categorical_features_target_value = {}
        self.average_categorical_variables = average_categorical_variables
        self.combined_categorical_features = []
        self.scaler_dict = {
        }
        self.categorical_betas = categorical_betas
        self.train_df = None
        self.test_df = None
        self.target_mean = None,
        self.target_scaled = None
        self.base_target_name = self.target
        self.combined_categorical_features = combined_categorical_features
        self.categorical_features = categorical_features
        self.model_performance_tracker = {}
        self.categorical_encoding_mapping = {}

    def fit(self, X, y):

        global single_categorical_accuracy, single_categorical_CM, best_feature_combination, best_model_name, best_tuned_parameters, best_categorical_beta_combination, best_categorical_features_target_values, best_model, f1_p, f1_n
        if self.predict_standardized_target is True:
            self.convert_target_to_standardized()

        if self.target_type == None:
            self.target_type = self.get_target_type(y)

        if isinstance(self.df, pd.DataFrame) == False:
            self.df = X.copy()
            self.features = [f for f in self.df]
        else:
            for feature in X:
                self.features.append(feature)

        self.df[self.target] = y

        self.df = self.add_combined_categories_as_features(self.df, self.combined_categorical_features)

        self.train_df, scaler, self.test_df = self.preprocess_data(only_split_data=True)

        for categorical_feature in self.categorical_features:
            if categorical_feature not in self.features:
                continue
            self.features.remove(categorical_feature)
            self.features.append(self.target + '_' + categorical_feature + EXPECTED_NAME)

        categorical_beta_combinations = self.create_categorical_beta_combinations()
        new_ml_algorithms = self.add_parameters(self.machine_learning_models)

        min_error = 99999999999  ### Min error. Set to highest value.
        # Is any of the ml model iterations below perform better, the minimum error will be updated.
        iteration_number = 0
        ml_number = -1
        for categorical_beta_combination in categorical_beta_combinations:
            iteration_number += 1
            for beta_combination, categorical_feature in categorical_beta_combination:
                self.set_categorical_feature_equal_to_expected_target_value(self.train_df, categorical_feature,
                                                                            factor=beta_combination)

            self.train_df, scaler, self.test_df = self.preprocess_data()
            self.standardized_features = self.get_standardized_feature_names()

            for ml_model, ml_parameters in new_ml_algorithms.items():
                ml_model = self.initialise_model_if_not_initialised(ml_model)

                self.get_results_without_tuning(ml_model)

                if self.find_optimal_features is True:
                    GridSearch = self.feature_selection_best_model(ml_model, ml_parameters)

                else:

                    GridSearch = GridSearchBinary(features, self.target, ml_model, ml_parameters,
                                                  train_df=self.train_df, test_df=self.test_df,
                                                  loss_function=self.loss_function)
                    GridSearch.get_best_model()

                single_categorical_best_model = GridSearch.best_model
                single_categorical_variation_error = GridSearch.best_model_error
                single_categorical_accuracy = GridSearch.best_model_accuracy
                single_categorical_CM = GridSearch.best_model_cm

                f1_p, f1_n = get_f1_from_CM(single_categorical_CM)

                ml_number += 1
                self.model_performance_tracker[ml_number] = {
                    'error': single_categorical_variation_error,
                    'model_name': type(ml_model).__name__,
                    'parameters': ml_parameters,
                    'model_method': GridSearch.best_model_method,
                    'categorical_beta_combination': categorical_beta_combination,
                    'full_model': single_categorical_best_model,
                }

                if single_categorical_variation_error < min_error:

                    if self.find_optimal_features is False:
                        best_feature_combination = self.features.copy()
                    else:
                        best_feature_combination = self.used_features
                    min_error = single_categorical_variation_error
                    try:
                        best_model = single_categorical_best_model.best_estimator_
                    except Exception:
                        best_model = single_categorical_best_model
                    best_tuned_parameters = self.get_model_parameters(best_model, ml_parameters)
                    best_model_name = type(best_model).__name__
                    best_model_method = GridSearch.best_model_method

                    best_categorical_beta_combination = categorical_beta_combination
                    best_categorical_features_target_values = self.categorical_features_target_value

                if self.verbose == True:
                    print(type(ml_model).__name__, " logloss: ", single_categorical_variation_error, "accuracy:",
                          single_categorical_accuracy, "f1_p:", f1_p, "f1_n:", f1_n)

            if self.verbose == True:
                print("Finished with iteration " + str(iteration_number) + " out of " + str(
                    len(categorical_beta_combinations)), " Best score metric currently: ", min_error, "Accuracy:",
                      single_categorical_accuracy, "f1_p:", f1_p, "f1_n:", f1_n)
                print(time.asctime(time.localtime(time.time())))
        if self.verbose is True:
            print("Best model " + str(self.target), min_error, best_feature_combination, best_model_name,
                  best_tuned_parameters, best_categorical_beta_combination)

        self.get_scalers_into_dict_new(
            best_categorical_features_target_values, best_feature_combination)

        self.df = self.train_df.append(self.test_df)

        return best_model

    def get_results_without_tuning(self, model):
        res = model.fit(self.train_df[self.features].values, self.train_df[self.target].values)

        error = self.get_error_on_test_data(res)
        accuracy = self.get_accuracy_on_test_data(res)
        CM = self.get_confusion_matrix_on_test_data(res)

        f1_p_ohne, f1_n_ohne = get_f1_from_CM(CM)
        print("with out tuning, logloss:", error, "accuracy:", accuracy, "f1_p:", f1_p_ohne, "f1_n", f1_n_ohne)

    def get_model_parameters(self, model, ml_parameters):
        best_parameters = {}
        for column in ml_parameters:

            if column == "learning_rate":
                best_parameters[column] = model.learning_rate
            elif column == "n_estimators":
                best_parameters[column] = model.n_estimators
            elif column == "max_depth":
                best_parameters[column] = model.max_depth
            elif column == "penalty":
                best_parameters[column] = model.penalty
            elif column == "solver":
                best_parameters[column] = model.solver
            elif column == "kernel":
                best_parameters[column] = model.kernel
            elif column == "gamma":
                best_parameters[column] = model.gamma
            elif column == "C":
                best_parameters[column] = model.C
            elif column == "hidden_layer_sizes":
                best_parameters[column] = model.hidden_layer_sizes
            elif column == "max_iter":
                best_parameters[column] = model.max_iter
            elif column == "alpha":
                best_parameters[column] = model.alpha
            elif column == "num_leaves":
                best_parameters[column] = model.num_leaves
        return best_parameters

    def initialise_model_if_not_initialised(self, ml_model):
        model_name = type(ml_model).__name__
        if model_name == "type" or model_name == "ABCMeta":
            initialised_model_name = ml_model.__name__
            if 'GAM' not in initialised_model_name:
                return ml_model()
            else:
                return ml_model
        else:
            return ml_model

    def convert_target_to_standardized(self):
        self.base_target_name = self.target
        self.target = STANDARDIZED_TARGET_NAME + self.target
        self.target_mean = self.df[self.base_target_name].mean()
        self.target_scaled = self.df[self.base_target_name].std()
        self.df[self.target] = (self.df[self.base_target_name] - self.target_mean) / self.target_scaled

    def preprocess_data(self, only_split_data=False):

        # Prepares the data so it is ready to be optimized by a machine learning algorithm.

        DataP = DataPreprocessing(self.target, self.features, self.df,
                                  date_period_column_name=self.date_period_column_name
                                  )

        droppedna_df = DataP.drop_na(self.categorical_features, self.combined_categorical_features)

        if only_split_data is True:
            self.train_df, self.test_df = DataP.create_train_test_data(droppedna_df,
                                                                       test_data_min_date=self.test_period_min_date,
                                                                       test_data_max_date=self.test_period_max_date)
            scaler = None
        else:
            self.train_df, self.test_df = DataP.create_train_test_data(droppedna_df,
                                                                       test_data_min_date=self.test_period_min_date,
                                                                       test_data_max_date=self.test_period_max_date)
            self.test_df, self.train_df, scaler = DataP.scale_data(self.test_df, self.train_df,
                                                                   self.categorical_features)

        return self.train_df, scaler, self.test_df


    def set_multiple_categorical_feature_equal_to_expected_target_value(self, df, feature, factor=150):

        #### Creates a new column called called column_name+_expected.
        # The expected mean of the target value given the value of the category is inserted here

        mean_all = df[self.target].mean()
        unique_values = df[feature].unique().tolist()
        self.categorical_features_target_value[feature] = {}
        for category in unique_values:
            rows = df[df[feature] == category]
            if isinstance(self.categorical_features, list) is True:
                mean = rows[self.target].mean()
            else:  ## if Dict
                other_column_name = self.categorical_features[feature]
                if other_column_name == None:
                    mean = rows[self.target].mean()
                else:
                    mean = (rows[self.target] - rows[other_column_name]).mean()

            weight = (1 / (1 + 10 ** (
                    -len(rows) / factor)) - 0.5) * 2  ## returns a value between 0 and 1 dependant on sample size

            estimated_value = mean_all * (1 - weight) + weight * mean
            self.df.loc[self.df[feature] == category, self.target + '_' + feature + '_expected'] = estimated_value
            self.categorical_features_target_value[feature][category] = estimated_value

    def set_categorical_feature_equal_to_expected_target_value(self, df, feature, factor=150):

        mean_all = df[self.target].mean()
        unique_values = df[feature].unique().tolist()
        self.categorical_features_target_value[feature] = {}
        for category in unique_values:
            rows = df[df[feature] == category]
            if isinstance(self.categorical_features, list) is True:
                mean = rows[self.target].mean()
            elif isinstance(self.categorical_features, dict) is True:
                other_column_name = self.categorical_features[feature]
                if other_column_name == None:
                    mean = rows[self.target].mean()
                else:
                    mean = (rows[self.target] - rows[other_column_name]).mean()

            weight = (1 / (1 + 10 ** (-len(rows) / factor)) - 0.5) * 2
            estimated_value = mean_all * (1 - weight) + weight * mean
            new_column_name = self.target + '_' + feature + EXPECTED_NAME

            self.df.loc[self.df[feature] == category, new_column_name] = estimated_value
            self.categorical_features_target_value[feature][category] = estimated_value

    def get_standardized_feature_names(self):
        standardized_features = []
        for feature in self.features:
            expected_column_name = self.target + '_' + feature
            if expected_column_name in self.df.columns:
                standardized_features.append(expected_column_name + STANDARDIZED_NAME)
            else:
                standardized_features.append(feature + STANDARDIZED_NAME)

        return standardized_features

    def add_combined_categories_as_features(self, df, combined_categorical_features):

        ### Creates a new combined category. Currently 2 and 3 categories are supported.

        for key, combined_categories in combined_categorical_features.items():
            if len(combined_categories) == 2:
                column_name = combined_categories[0] + '_' + combined_categories[1]
                df[column_name] = COMBINED_CATEGORY0_NAME + (df[combined_categories[0]].fillna(0).map(int) * 1000).map(
                    str) + \
                                  COMBINED_CATEGORY1_NAME + (df[combined_categories[1]].fillna(0).map(int) * 1000).map(
                    str)

            else:
                column_name = combined_categories[0] + '_' + combined_categories[1] + '_' + combined_categories[2]
                df[column_name] = COMBINED_CATEGORY0_NAME + (df[combined_categories[0]].fillna(0).map(int) * 1000).map(
                    str) + \
                                  COMBINED_CATEGORY1_NAME + (df[combined_categories[1]].fillna(0).map(int) * 1000).map(
                    str) + \
                                  COMBINED_CATEGORY2_NAME + (df[combined_categories[2]].fillna(0).map(int) * 1000).map(
                    str)

            self.features.append(column_name)
            if isinstance(self.categorical_features, list) == True:
                self.categorical_features.append(column_name)
            else:
                self.categorical_features[column_name] = None

        return df

    def create_categorical_beta_combinations(self):

        ### Creates a list of tuple  values for every unique combination of categorical value to beta value.

        betas = self.categorical_betas

        all_combinations = [list(zip(each_permutation, self.categorical_features)) for each_permutation in
                            itertools.permutations(betas, len(self.categorical_features))]

        if len(self.categorical_features) >= 2:
            for beta in betas:
                li = []
                for categorical_feature in self.categorical_features:
                    li.append(tuple((beta, categorical_feature)))

                all_combinations.append(li)
        if len(all_combinations) == 0:
            all_combinations = []

        return all_combinations

    def add_parameters(self, ml_algorithms):

        ### If no parameters is set, the default parameters for the specific ml algorithm will be added.

        ml_algorithms_copy = {}

        for ml_model, raw_ml_parameters in ml_algorithms.items():
            ml_model_name = type(ml_model).__name__

            if ml_model_name == "type" or ml_model_name == "ABCMeta":
                ml_model_name = ml_model.__name__

            if raw_ml_parameters == None:
                ml_parameters = get_defalt_model_parameters(ml_model_name, self.target_type, self.features)
            else:
                ml_parameters = raw_ml_parameters
            ml_algorithms_copy[ml_model] = ml_parameters

        return ml_algorithms_copy

    def get_feature_combination_count(self):
        total_combinations = 0
        for feature_count in range(len(self.standardized_features)):
            feature_combinations = itertools.combinations(self.standardized_features, feature_count + 1)
            for feature_combination in feature_combinations:
                features = list(feature_combination)
                total_combinations += len(features)
        return total_combinations

    def feature_selection_best_model(self, ml_model, ml_parameters):

        ### Iterates over every combination of features.

        min_feature_combination_error = 9999999999999
        total_combinations = self.get_feature_combination_count()
        print("Processing Feature Selection. Unique combinations:", total_combinations)
        feature_combination_number = 0
        for feature_count in range(len(self.standardized_features)):
            feature_combinations = itertools.combinations(self.standardized_features, feature_count + 1)
            for feature_combination in feature_combinations:
                feature_combination_number += 1
                features = list(feature_combination)

                GridSearch = GridSearchBinary(features, self.target, ml_model, ml_parameters,
                                      train_df=self.train_df, test_df=self.test_df,
                                      loss_function=self.loss_function)
                GridSearch.get_best_model()

                error = GridSearch.best_model_error
                if error < min_feature_combination_error:
                    min_feature_combination_error = error
                    best_prediction_method = GridSearch
                    basic_feature_names = [f.replace("_standardized", "") for f in features]
                    self.used_features = basic_feature_names

                if self.verbose == True and total_combinations > 200 and feature_combination_number % 200 == 0:
                    print("Processed " + str(feature_combination_number) + " out of " + str(
                        total_combinations) + " feature combinations")
                    print(time.asctime(time.localtime(time.time())))

        return best_prediction_method

    def get_feature_importances_or_coefficients(self, grid_model):
        try:
            model = grid_model.best_estimator_  ### Uses best estimator as model if it exists
        except Exception:
            model = grid_model
        try:
            intecepts = model.intercept_
            coefs = model.coef_
            classes = model.classes_

            if len(classes) > 2:
                classes_used = [0, len(classes)]
            else:
                classes_used = classes.copy()

            for number in range(len(coefs)):
                class_ = classes_used[number]
                coef = coefs[number]
                print(class_, coef)
                my_formatted_list = [round(elem, 3) for elem in coef]
                for n, c in enumerate(my_formatted_list):
                    try:
                        print(self.used_features[n], c)
                    except IndexError:
                        pass  ## TEMP
                intercept = intecepts[number]
                pass

        except AttributeError:
            pass

        try:
            print("Feature importance", model.feature_importances_)
        except AttributeError:
            pass

    def get_scalers_into_dict_new(self, best_categorical_features_target_values, features):

        ### Insert the values needed for  future data transformations into a dictionary.

        mean = self.train_df[self.base_target_name].mean()
        std = self.train_df[self.base_target_name].std()

        self.scaler_dict[self.target] = {
            MEAN: mean,
            STD: std,
            CATEGORY_EXPECTED_MEAN: None,
            BASE_COLUMN_NAME: self.base_target_name,
            ENCODE: self.categorical_encoding_mapping,

        }

        for feature_number, feature in enumerate(features):
            feature_name = feature.replace(STANDARDIZED_NAME, "")

            mean = self.train_df[feature_name].mean()
            std = self.train_df[feature_name].std()
            base_feature_name = feature_name.replace(EXPECTED_NAME, "")
            base_feature_name = base_feature_name.replace(self.target + '_', "")

            if base_feature_name in best_categorical_features_target_values:
                category_expected_mean = best_categorical_features_target_values[base_feature_name]
            else:
                category_expected_mean = None

            self.scaler_dict[feature] = {
                MEAN: mean,
                STD: std,
                CATEGORY_EXPECTED_MEAN: category_expected_mean,
                BASE_COLUMN_NAME: base_feature_name,

            }

    def insert_model(self, ml_model, model_name, insert_ml_model_to_local_filepath=True, insert_ml_model_to_aws=False,
                     insert_dataframe_to_local_filepath=False, insert_dataframe_to_aws=False):

        if insert_ml_model_to_local_filepath is True:
            pickle.dump(ml_model, open(ml_models_filepath + "/" + model_name + '_model', 'wb'))
            pickle.dump(self.scaler_dict, open(ml_models_filepath + "/" + model_name + '_scale', 'wb'))

    def visualise_feature_importance(self, model):
        try:
            intecepts = model.intercept_
            coefs = model.coef_
            classes = model.classes_

            if len(classes) > 2:
                classes_used = [0, len(classes)]
            else:
                classes_used = classes.copy()

            for number in range(len(coefs)):
                class_ = classes_used[number]
                coef = coefs[number]
                print(class_, coef)
                my_formatted_list = [round(elem, 3) for elem in coef]
                for n, c in enumerate(my_formatted_list):
                    try:
                        print(self.used_features[n], c)
                    except IndexError:
                        pass  ## TEMP
                intercept = intecepts[number]
                pass

        except AttributeError:
            pass

        try:
            print("Feature importance", model.feature_importances_)
        except AttributeError:
            pass

    def get_error_on_test_data(self, best_estimator):
        probabilities = best_estimator.predict_proba(self.test_df[self.features].values)

        error = log_loss(self.test_df[self.target], probabilities)
        return error

    def get_accuracy_on_test_data(self, best_estimator):
        probabilities = best_estimator.predict_proba(self.test_df[self.features].values)
        df_p = pd.DataFrame(probabilities)
        df_p['difference'] = df_p[0] - df_p[1]
        df_p.loc[df_p['difference'] > 0, 'result'] = 0
        df_p.loc[df_p['difference'] < 0, 'result'] = 1
        df_p.loc[df_p['difference'] == 0, 'result'] = -1
        accuracy = accuracy_score(self.test_df[self.target], df_p['result'])
        return accuracy

    def get_confusion_matrix_on_test_data(self, best_estimator):
        probabilities = best_estimator.predict_proba(self.test_df[self.features].values)
        df_p = pd.DataFrame(probabilities)
        df_p['difference'] = df_p[0] - df_p[1]
        df_p.loc[df_p['difference'] > 0, 'result'] = 0
        df_p.loc[df_p['difference'] < 0, 'result'] = 1
        df_p.loc[df_p['difference'] == 0, 'result'] = -1

        even_df = df_p.loc[df_p['result'] == -1]
        for index, row in even_df.iterrows():
            predict_value = 1
            real_value = self.test_df.iloc[index].at[self.target]
            if real_value == 1:
                predict_value = 0
            df_p.iloc[index].at['result'] = predict_value

        CM = confusion_matrix(self.test_df[self.target], df_p['result'])
        return CM
