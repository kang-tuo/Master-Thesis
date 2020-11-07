def get_defalt_model_parameters(model_name, target_type, features):
    if "XGBRegressor" in model_name or 'LGBMRegressor' in model_name:
        return {
            'learning_rate': [0.1, 0.25, 0.5],
            'n_estimators': [40, 100, 250],
            'max_depth': [3],
            'objective': ["reg:squarederror"],
        }

    elif "XGBClassifier" in model_name or 'LGBMClassifier' in model_name:
        return {
            'learning_rate': [0.05, 0.2, 0.5],
            'n_estimators': [40, 100, 250],
            'max_depth': [3, 5, 10, 20, 30, 50],
            'num_leaves': [30, 50, 80]
        }


    elif 'RandomForest' in model_name:
        return {
            'n_estimators': [40, 100, 250],
            'max_depth': [3, 5, 10],
        }


    elif model_name == "LogisticRegression":
        return {
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']

        }

    elif model_name == "LinearRegression" or model_name == "LogitRegression":
        return {
        }

    # elif 'GAM' in model_name:
    #
    #     lambdas = []
    #     for i in range(len(features)):
    #
    #         lamb= min(0.35+i*0.15,1)
    #         lambdas.append(lamb)
    #
    #     n_splines = good_function(features)
    #     return {
    #         'lam':lamb,
    #         'n_splines':n_splines}
