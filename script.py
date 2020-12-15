import pandas as pd
import lightgbm as lgb

def regression_model_lightgbm(X_train, y_train, X_test):
    lgb_train = lgb.Dataset(X_train, y_train)
    evals_result = {}

    params = {'bagging_fraction': 0.9223944053685549, 
    'boosting_type': 'gbdt', 
    'drop_rate': 0.29847846764282016, 
    'feature_fraction':  0.30501714460224083, 
    'lambda_l1': 0.21894440516534, 
    'lambda_l2': 1.3848191603958375, 
    'learning_rate': 0.08210966706622723, 
    'max_depth': 3, 'max_leaves': 17, 
    'min_data_in_leaf': 8, 
    'objective': 'regression', 
    'metrics': ['l1', 'l2', 'huber'], 
    'verbose': -1}

    gbm = lgb.train(params, lgb_train, 1800)
    
    light_gbm_test = gbm.predict(X_test)
    
    return light_gbm_test