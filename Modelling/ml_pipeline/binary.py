from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss, accuracy_score, f1_score, confusion_matrix
from pygam import LogisticGAM, LinearGAM
import numpy as np
# from functions import get_nsplines_for_gam
import pandas as pd


class GridSearchBinary():

    def __init__(self, features, target, ml_model, ml_parameters, train_df, test_df, loss_function='neg_log_loss'):
        self.features = features
        self.target = target
        self.train_df = train_df
        self.test_df = test_df
        self.ml_parameters = ml_parameters
        self.ml_model = ml_model
        if loss_function == "default":
            self.loss_function = 'neg_log_loss'
        else:
            self.loss_function = loss_function
        self.score_tracker = {}
        self.best_model_parameters = {}
        self.best_model_error = 9999999999999
        self.best_model_method = None
        self.best_model = None

    def get_best_model(self):

        ml_model_name = type(self.ml_model).__name__
        if ml_model_name == 'type' or ml_model_name == "ABCMeta":
            ml_model_name = self.ml_model.__name__
        if ml_model_name != 'LogisticGAM':
            grid_search = GridSearchCV(self.ml_model, self.ml_parameters, n_jobs=3,
                                       verbose=0, scoring=self.loss_function, cv=5)
            grid_search.fit(self.train_df[self.features].values, self.train_df[self.target].values)
            # grid_search.score()
            error = self.get_error_on_test_data(grid_search.best_estimator_)
            accuracy = self.get_accuracy_on_test_data(grid_search.best_estimator_)
            CM = self.get_confusion_matrix_on_test_data(grid_search.best_estimator_)


        if error < 0:
            error *= -1
        self.best_model_error = error
        self.best_model_accuracy = accuracy
        self.best_model_cm = CM

        if ml_model_name != 'LogisticGAM':
            self.best_model = grid_search
        else:
            self.best_model = model
            return self.ml_model
        return grid_search.best_estimator_

    def get_error_on_test_data(self, best_estimator):
        probabilities = best_estimator.predict_proba(self.test_df[self.features].values)
        if self.loss_function == "neg_log_loss":
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

        CM = confusion_matrix(self.test_df[self.target], df_p['result'])
        # f1 = f1_score(self.test_df[self.target], df_p['result'])
        return CM

    def get_ml_model_with_highest_accuracy_from_dict(self):
        return min(self.score_tracker, key=self.score_tracker.get)


