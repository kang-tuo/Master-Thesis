from sklearn.model_selection import GridSearchCV

class GridSearchRegression():



    def __init__(self,features,target,ml_model,ml_parameters,train_df,test_df,loss_function='neg_mean_squared_error'):
        self.features = features
        self.target = target

        self.train_df = train_df
        self.test_df = test_df
        self.ml_parameters = ml_parameters
        self.ml_model = ml_model
        self.score_tracker = {}
        self.best_model_parameters = {}
        self.best_model_name = None
        self.best_model_error = None
        self.best_model_method = None

        self.best_model_error = None
        if loss_function == "default":
            self.loss_function = 'neg_mean_squared_error'
        else:
            self.loss_function = loss_function

    def get_best_model(self):
        self.best_model_error = 999999999999

        ml_model_name = type(self.ml_model).__name__

        if ml_model_name == "LogitRegression":
            if self.train_df[self.target].max() > 1 or self.train_df[self.target].min() < 0:
                self.best_model_error= 999999999 # BUG in data
            else:
                best_estimator = self.ml_model.fit(self.train_df[self.features], self.train_df[self.target])

        elif ml_model_name == "RegressionGam":
            self.ml_model.fit(self.train_df[self.features].values, self.train_df[self.target].values)
            best_estimator = self.ml_model

        else:
            grid_search = GridSearchCV(self.ml_model, self.ml_parameters, n_jobs=3,
                                       verbose=0, scoring=self.loss_function, cv=5, )

            grid_search.fit(self.train_df[self.features].values, self.train_df[self.target].values)
            best_estimator = grid_search.best_estimator_
        self.best_model = best_estimator
        self.best_model_error = self.get_error_on_test_data(best_estimator, self.ml_model)


        return    best_estimator

    def get_error_on_test_data(self,best_estimator,ml_model):

        if  type(ml_model).__name__ =="LogitRegression":
            predictions = ml_model.predict(self.test_df[self.features].values)
        else:
            predictions = best_estimator.predict(self.test_df[self.features].values)
        if self.loss_function == "neg_mean_squared_error":
            mean_squared_error = ((abs(self.test_df[self.target]-predictions))**2).mean()


        return mean_squared_error


