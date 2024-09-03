# -*- coding: utf-8 -*-
"""
Created on Fri May 17 09:36:30 2024

@author: ieron
"""

import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lightgbm as lgb

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set display options
pd.set_option('display.max_rows', 125)
pd.set_option('display.max_columns', 125)
sns.set_style('darkgrid')

# Define custom colors/cmaps/palettes
denim='#6F8FAF'
salmon='#FA8072'
slate_gray = '#404040'
palette = 'colorblind'

# Load data
DATA_PATH = r'C:\Users\ieron\Desktop\python'
train = pd.read_csv(os.path.join(DATA_PATH, 'application_train.csv'), index_col='SK_ID_CURR')
target = 'TARGET'
X_train = train.drop(target, axis=1).copy()
y_train = train[target].copy()
X_test = pd.read_csv(os.path.join(DATA_PATH, 'application_test.csv'), index_col='SK_ID_CURR')
submission = pd.read_csv(os.path.join(DATA_PATH, 'application_test.csv'))[['SK_ID_CURR']]

# Customized Preprocessing Classes
class MissingFlagger(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_flag=None):
        self.columns_to_flag = columns_to_flag
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for col in self.columns_to_flag:
            X_transformed[f'MISSING_FLAG_{col}'] = X_transformed[col].isnull().astype(int)
        return X_transformed

class MissingValueFiller(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.median_values = X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']].median()
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed['NAME_TYPE_SUITE'] = X_transformed['NAME_TYPE_SUITE'].fillna('MISSING')
        X_transformed['OCCUPATION_TYPE'] = X_transformed['OCCUPATION_TYPE'].fillna('MISSING')
        X_transformed['CNT_FAM_MEMBERS'] = X_transformed['CNT_FAM_MEMBERS'].fillna(2)
        X_transformed[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']] = X_transformed[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']].fillna(self.median_values)
        X_transformed = X_transformed.fillna(0)
        return X_transformed

class Merger(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        X_transformed['CODE_GENDER'] = X_transformed['CODE_GENDER'].apply(lambda x: 'F' if x == 'XNA' else x)
        X_transformed['NAME_INCOME_TYPE'] = X_transformed['NAME_INCOME_TYPE'].apply(lambda x: 'Other' if x in {'Unemployed', 'Student', 'Businessman', 'Maternity leave'} else x)
        X_transformed['NAME_FAMILY_STATUS'] = X_transformed['NAME_FAMILY_STATUS'].apply(lambda x: 'Married' if x == 'Unknown' else x)
        X_transformed['CNT_CHILDREN'] = X_transformed['CNT_CHILDREN'].apply(lambda x: 4 if x > 4 else x)
        X_transformed['CNT_FAM_MEMBERS'] = X_transformed['CNT_FAM_MEMBERS'].apply(lambda x: 6 if x > 6 else x)
        X_transformed['DAYS_EMPLOYED'] = X_transformed['DAYS_EMPLOYED'].apply(lambda x: 0 if x > 0 else x)
        return X_transformed

class CategoricalConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.categories_ = {}

    def fit(self, X, y=None):
        cat_columns = X.select_dtypes(include=['object', 'category']).columns
        for col in cat_columns:
            self.categories_[col] = X[col].astype('category').cat.categories
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col, categories in self.categories_.items():
            X_transformed[col] = pd.Categorical(X_transformed[col], categories=categories, ordered=False)
        return X_transformed

class CustomOneHotEncoder(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.feature_names = None

    def fit(self, X, y=None):
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        self.ohe.fit(X[categorical_columns].astype('category'))
        self.feature_names = list(X.columns)
        return self

    def transform(self, X):
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        X_ohe = X[categorical_columns].copy()
        X_ohe = self.ohe.transform(X_ohe)
        ohe_column_names = self.ohe.get_feature_names_out(categorical_columns)
        X_ohe = pd.DataFrame(X_ohe, columns=ohe_column_names, index=X.index)
        X_transformed = pd.concat([X.drop(categorical_columns, axis=1), X_ohe], axis=1).copy()
        return X_transformed

class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_standardize=None):
        self.columns_to_standardize = columns_to_standardize

    def fit(self, X, y=None):
        if self.columns_to_standardize is None:
            self.columns_to_standardize = list(X.columns)
        if self.columns_to_standardize:
            self.scaler = StandardScaler()
            self.scaler.fit(X[self.columns_to_standardize])
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        if self.columns_to_standardize:
            X_transformed[self.columns_to_standardize] = self.scaler.transform(X_transformed[self.columns_to_standardize])
        return X_transformed

# Preprocessing and Model Training with LightGBM
cols_to_flag = ['OCCUPATION_TYPE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AMT_REQ_CREDIT_BUREAU_YEAR']
cols_to_drop = ['FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']

lgbm_preprocessor = Pipeline([
    ('missing_flagger', MissingFlagger(cols_to_flag)),
    ('missing_value_filler', MissingValueFiller()),
    ('merger', Merger()),
    ('column_dropper', FunctionTransformer(lambda X: X.drop(cols_to_drop, axis=1), validate=False)),
    ('categorical_converter', CategoricalConverter())
])

X = train.drop(target, axis=1).copy()
y = train[target].copy()

X_processed = lgbm_preprocessor.fit_transform(X)
X_test_processed = lgbm_preprocessor.transform(X_test)

X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

params = {
    'objective': 'binary',
    'metric': 'binary_error',
    'num_leaves': 11,
    'learning_rate': 0.05,
    'verbose': -1,
    'early_stopping_rounds': 250
}

lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val)
model = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=[lgb_train, lgb_val])

# Make probability predictions
y_pred_train_proba = model.predict(X_train, num_iteration=model.best_iteration, raw_score=True)
y_pred_val_proba = model.predict(X_val, num_iteration=model.best_iteration, raw_score=True)
y_pred_test_proba = model.predict(X_test_processed, num_iteration=model.best_iteration)

# Performance Evaluation
roc_auc_train = roc_auc_score(y_train, y_pred_train_proba)
roc_auc_val = roc_auc_score(y_val, y_pred_val_proba)
print(f'roc_auc_score:\n- Train: {roc_auc_train}\n- Validation: {roc_auc_val}')

# SHAP Values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

# Summary Plot
shap.summary_plot(shap_values, X_val, plot_type="bar")

# Feature Importance Plot
shap.summary_plot(shap_values, X_val)

# Submission
lgbm_submission = submission.copy()
lgbm_submission['TARGET'] = y_pred_test_proba
lgbm_submission.to_csv('lgbm_submission.csv', index=False)
