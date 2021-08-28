# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 01:10:27 2021

@author: monxu
"""

# Familiar imports
import numpy as np
import pandas as pd
# For ordinal encoding categorical variables, splitting data
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# For training random forest model
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBRegressor

# Load the training data
X = pd.read_csv("train.csv", encoding='utf-8', index_col=0)
test = pd.read_csv("test.csv", encoding='utf-8', index_col=0)

cont_features = [f for f in X.columns.tolist() if f.startswith('cont')]
cat_features = [f for f in X.columns.tolist() if f.startswith('cat')]
features = cat_features + cont_features
data = X[features]
y = X['target']
X = X.drop(['target'], axis= 1)

all_data = pd.concat([data, test])


object_cols = [col for col in X.columns if 'cat' in col]

ordinal_encoder = OrdinalEncoder()
X[cat_features] = ordinal_encoder.fit_transform(X[cat_features])
test[cat_features] = ordinal_encoder.transform(test[cat_features])

# Optional to drop weak feature columns

# for i in (0, 1, 3, 5, 6):
#     X = X.drop([f'cat{i}'], axis=1)
#     test = test.drop([f'cat{i}'], axis=1)


label = LabelEncoder()
categorical_feature = np.where(X.dtypes != 'float64')[0].tolist()
categorical_feature_columns = X.select_dtypes(exclude=['float64']).columns

for column in categorical_feature_columns:
        label.fit(X[column])
        X[column] = label.transform(X[column])
        test[column] = label.transform(test[column])

print(X.head)

lgbm_parameters = {
    'metric': 'rmse', 
    'n_jobs': -1,
    'n_estimators': 50000,
    'num_trees': 80000,
    'reg_alpha': 10.924491968127692,
    'reg_lambda': 17.396730654687218,
    'colsample_bytree': 0.21497646795452627,
    'subsample': 0.7582562557431147,
    'learning_rate': 0.009985133666265425,
    'max_depth': 18,
    'num_leaves': 44,
    'min_child_samples': 27,
    'max_bin': 254,
    'cat_l2': 0.025083670064082797
#     'boosting': 'dart',
#     'xgboost_dart_mode': True
}

# uniform_drop


lgbm_val_pred = np.zeros(len(y))
lgbm_test_pred = np.zeros(len(test))
mse = []
kf = KFold(n_splits=15, shuffle=True)

for trn_idx, val_idx in tqdm(kf.split(X,y)):
    x_train_idx = X.iloc[trn_idx]
    y_train_idx = y.iloc[trn_idx]
    x_valid_idx = X.iloc[val_idx]
    y_valid_idx = y.iloc[val_idx]

    lgbm_model = LGBMRegressor(**lgbm_parameters)
    lgbm_model.fit(x_train_idx, y_train_idx, eval_set = ((x_valid_idx,y_valid_idx)),verbose = -1, early_stopping_rounds = 500,categorical_feature=categorical_feature)  
    lgbm_test_pred += lgbm_model.predict(test)/15
    mse.append(mean_squared_error(y_valid_idx, lgbm_model.predict(x_valid_idx)))
    
np.mean(mse)
pd.DataFrame({'id':test.index,'target':lgbm_test_pred}).to_csv('submission.csv', index=False)

