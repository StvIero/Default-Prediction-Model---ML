# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:05:13 2024

@author: ieron
"""
import os
import gc
import numpy as np

import pandas as pd
# Set maximum number of rows and columns shown
pd.set_option('display.max_rows', 125)
pd.set_option('display.max_columns', 125)

import matplotlib.pyplot as plt
import matplotlib.colors

import seaborn as sns
# Define custom colors/cmaps/palettes for visualization purposes.
denim='#6F8FAF'
salmon='#FA8072'
slate_gray = '#404040'
cmap=matplotlib.colors.LinearSegmentedColormap.from_list("",[denim,salmon])
palette = 'colorblind'
sns.set_style('darkgrid')

from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, recall_score, make_scorer, roc_auc_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb

import warnings
# Suppress all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

DATA_PATH = r'C:\Users\ieron\Desktop\python'

train = pd.read_csv(os.path.join(DATA_PATH, 'application_train.csv'), index_col='SK_ID_CURR')

target = 'TARGET'

X_train = train.drop(target,axis=1).copy()
y_train = train[target].copy()

X_test = pd.read_csv(os.path.join(DATA_PATH, 'application_test.csv'), index_col='SK_ID_CURR')
submission = pd.read_csv(os.path.join(DATA_PATH, 'application_test.csv'))[['SK_ID_CURR']]

###############################################################################
##### Exploratory Data Analysis: Part I #######################################
###############################################################################


### Target ####################################################################

print(f"Target: '{target}'\n\n\
Data type: {y_train.dtype}\n\
Unique values: {y_train.dropna().unique()}\n\
Missing {y_train.isnull().sum()} values")

fig = plt.figure(figsize=(6.4,4.8))

# Sort unique values
unique_values = y_train.dropna().unique()
unique_values.sort()

# Value counts
val_counts = y_train.dropna().value_counts()
val_counts = val_counts.reindex(unique_values)

val_counts_pct = val_counts/len(y_train)*100

# Countplot
ax = sns.countplot(x=y_train,palette=palette)
ax.xaxis.grid(False)

# Annotating the bars with value counts percentages
lp_thresh = 1
for i, p in enumerate(ax.patches):
    pct = val_counts_pct.iloc[i]
    ax.annotate(f'{pct:.2f}%', (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='bottom', xytext=(0,0),
                textcoords='offset points')
    # Showing count value if rare (less than 1%)
    if pct < lp_thresh:
        ax.annotate(val_counts.iloc[i],
                    (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', xytext=(0,10),
                    textcoords='offset points',color='red')
                 
ax.set_title("Target's Distribution")

plt.tight_layout()
plt.show()


### Features Overview #########################################################
X_train.head()
X_train.shape
X_train.dtypes

# Numerical (continuous/discrete) and categorical features
num_feats = X_train.select_dtypes(include='number').columns.tolist()

thresh = 25

cont_feats = [feat for feat in num_feats if X_train[feat].nunique() > thresh]
disc_feats = [feat for feat in num_feats if X_train[feat].nunique() <= thresh]

cat_feats = X_train.select_dtypes(exclude='number').columns.tolist()

print(f'Features: {X_train.shape[1]}\n\n\
Continuous: {len(cont_feats)}\n\
{cont_feats}\n\n\
Discrete: {len(disc_feats)}\n\
{disc_feats}\n\n\
Categorical: {len(cat_feats)}\n\
{cat_feats}')

cont_feat, disc_feats, cat_feats = set(cont_feats), set(disc_feats), set(cat_feats)

'''
Due to the high number of features (120), it is reasonable to explore them in groups. 
Based on their description, we form a number of feature groups as the following.
'''

# Feature groups based on their desciption
demographics = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR',
                'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE',
                'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE']

count = ['CNT_CHILDREN', 'CNT_FAM_MEMBERS']

duration = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION',
            'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE', 'OWN_CAR_AGE']

    
social = ['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
          'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']

contact = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE',
           'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']

address = ['REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
           'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
           'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY']

region = ['REGION_POPULATION_RELATIVE', 'REGION_RATING_CLIENT',
          'REGION_RATING_CLIENT_W_CITY']

process = ['HOUR_APPR_PROCESS_START', 'WEEKDAY_APPR_PROCESS_START']

external = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

amount = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']

inquiry = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
           'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
           'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']

document = ['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4',
            'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7',
            'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10',
            'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13',
            'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
            'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19',
            'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']

building = ['APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG',
            'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG',
            'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG',
            'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG',
            'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE',
            'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE',
            'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE',
            'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE',
            'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE',
            'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI',
            'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI',
            'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI',
            'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI',
            'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI',
            'NONLIVINGAREA_MEDI', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE',
            'TOTALAREA_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']

# Missing values by feature
def null_percentage(df):
    return df.isna().sum()/len(df)*100

null_X_train = null_percentage(X_train)

fig = plt.figure(figsize=(19.2,4.8))
ax = null_X_train.loc[null_X_train>0].sort_values()\
    .plot(kind='bar',title='Percentage of missing values')

for p in ax.patches:
    ax.annotate(f'{p.get_height():.0f}',
                (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='bottom', xytext=(0,0),
                textcoords='offset points')

ax.set_ylabel('missing %')
ax.set_xlabel('feature')

ax.xaxis.grid(False)

plt.show()

'''
Features missing most values are mainly those related to the building, external sources and recent inquiries. 
In fact, there is no building feature with less than 47% missing values. We shall ignore the building features.
'''


### Customized Description / Plot Functions ###################################
'''
We use carefully crafted functions to investigate the features and determine what preprocessing steps to take.

    Summary functions: For each feature type (continuous, discrete/binary, and categorical), 
                        a summary function (cont_summary(), disc_summary(), and cat_summary()) is defined. 
                        This function returns a customized description of the feature, including the number 
                        of missing values, the number of unique values, correlation with target, 
                        among other properties.

    Plot functions: For each feature type (continuous, discrete/binary, and categorical), 
                    some plot functions (cont_plots(), disc_plots(), and cat_plots()) are crafted to visualize 
                    the feature distributions for different target values.

    Missing flag plot: For a feature with missing values, this plot depicts 
                        the credit default rate in the instances where the feature value is missing 
                        and contrasts it with the credit default rate in the instances not missing the feature value. 
                        It is meant to examine whether creating a binary 'missing flag' feature 
                        based on the feature is warranted. The binary 'missing flag' feature 
                        takes value 1 where the feature value is missing, and 0 otherwise.

These functions are all combined into a single function summary() to avoid repetition later.

    correlation heatmap: For each feature group, the function corr_heatmap() 
                        demonstrate the correlation between the feature within that group. 
'''
# Customized description and plots for any given feature
def summary(feat):
    
    if feat in cont_feats:
        cont_summary(feat)
        cont_plots(feat)
    elif feat in disc_feats:
        disc_summary(feat)
        disc_plots(feat)
    else:
        cat_summary(feat)
        cat_plots(feat)
    
    missing_flag_plot(feat)
    
    return

# --------------------------------
# Customized correlation heatmap for each feature group

def corr_heatmap(key):

    sns.set_style('white')

    group = feature_groups[key].copy()
    scale = 1 if len(pd.get_dummies(X_train[group])) < 5 else 2
    
    corr = pd.concat([pd.get_dummies(X_train[group]),y_train],axis=1)\
        .corr(numeric_only=True)
    fig = plt.figure(figsize=(6.4*scale,5.6*scale))
    ax = sns.heatmap(corr,annot=True,fmt='.2f',cmap='viridis')
    ax.set_title(f'Correlation heatmap: {key}')

    fig.tight_layout()
    plt.show()
    sns.set_style('darkgrid')
    
    return

# --------------------------------
# Customized description for continuous features

def cont_summary(feat):

    # Create an empty summary
    columns = ['dtype', 'count', 'unique', 'top_value_counts', 'missing_count',
               'missing_percentage','mean', 'std', 'min', 'median', 'max',
               'corr_with_target']
    summary = pd.DataFrame(index=[feat],columns=columns,dtype=float)
    
    # Pull the feature column in question
    col = X_train[feat].copy()
    
    # Basic statistics using the original describe method
    summary.loc[feat,['count','mean', 'std', 'min', 'median', 'max']]\
        = col.describe(percentiles=[.5]).values.transpose()
    
    # Number of unique values
    summary.loc[feat,'unique'] = col.nunique()

    # Missing values count
    summary.loc[feat,'missing_count'] = col.isnull().sum()

    # Missing values percentage
    summary.loc[feat,'missing_percentage'] = col.isnull().sum()/len(col)*100

    # Correlation with target
    summary.loc[feat,'corr_with_target'] = col.corr(y_train)
    
    int_cols = ['count', 'unique', 'missing_count']
    summary[int_cols] = summary[int_cols].astype(int)
    summary = summary.round(2).astype(str)

    # Top 3 value_counts
    value_counts = X_train[feat].value_counts().head(3)
    value_counts.index = value_counts.index.astype(float).to_numpy().round(2)
    summary.loc[feat,'top_value_counts'] = str(value_counts.to_dict())

    # Data type
    summary.loc[feat,'dtype'] = col.dtypes
    
    return display(summary)

# --------------------------------
# Customized plots for continuous features

def cont_plots(feat,bins='auto'):
    
    n_cols = 3
    fig, axes = plt.subplots(1, n_cols, figsize=(6.4*n_cols, 4.8))
    
    # Histogram
    sns.histplot(data=X_train,
                 x=feat,
                 bins=bins,
                 ax=axes[0],
                 color=slate_gray)
    
    # Box plots with the target as hue
    sns.boxplot(data=X_train,
                x=feat,
                y=y_train,
                ax=axes[1],
                palette=palette,
                orient='h')
    
#     KDE plots with the target as hue
    sns.kdeplot(data=X_train,
                x=feat,
                hue=y_train,
                palette=palette,
                fill=True,
                common_norm=False,
                ax=axes[2])
    
    axes[0].title.set_text('Histogram')
    axes[1].title.set_text('Box Plots')
    axes[2].title.set_text('KDE Plots')
    
    fig.tight_layout()
    plt.show()
    return

# --------------------------------
# Customized description for discrete features

def disc_summary(feat):
    
    # Create an empty summary
    columns = ['dtype', 'count', 'unique', 'missing_count',
               'missing_percentage', 'mean', 'std', 'min', 'median',
               'max', 'cv', 'corr_with_target']
    summary = pd.DataFrame(index=[feat],columns=columns,dtype=float)
    
    # Pull the feature column in question
    col = X_train[feat].copy()
    
    # Basic statistics using the original describe method
    summary.loc[feat,['count','mean', 'std', 'min', 'median', 'max']]\
    = col.describe(percentiles=[.5]).values.transpose()

    # Number of unique values
    summary.loc[feat,'unique'] = col.nunique()

    # Coefficient of Variation (CV)    
    summary.loc[feat,'cv'] = np.NaN if not col.mean() else col.std()/col.mean()

    # Missing values count
    summary.loc[feat,'missing_count'] = col.isnull().sum()

    # Missing values percentage
    summary.loc[feat,'missing_percentage'] = col.isnull().sum()/len(col)*100
    
    # Correlation with target
    summary.loc[feat,'corr_with_target'] = col.corr(y_train)
    
    int_cols = ['count','unique','missing_count']
    summary[int_cols] = summary[int_cols].astype(int)
    summary = summary.round(2).astype(str)
    
    # Data type
    summary.loc[feat,'dtype'] = col.dtypes
        
    return display(summary)

# --------------------------------
# Customized plots for discrete features

def disc_plots(feat):

    col = X_train[feat].copy()    

    n_rows = 1
    n_cols = 2

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6.4 * n_cols, 4.8 * n_rows))

    # Sort unique values
    unique_values = col.dropna().unique()
    unique_values.sort()

    # Value counts
    val_counts = col.dropna().value_counts()
    val_counts = val_counts.reindex(unique_values)
    val_counts_pct = val_counts/len(col)*100
    
    # Countplot
    sns.countplot(x=col, order=unique_values, palette=palette, ax=axes[0])
    axes[0].xaxis.grid(False)
    
    # Show count value if rare (less than 1%)
    lp_thresh = 1
    for i, p in enumerate(axes[0].patches):
        pct = val_counts_pct.iloc[i]
        axes[0].annotate(f'{pct:.2f}%',
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', xytext=(0,0),
                         textcoords='offset points')
        if pct < lp_thresh:
            axes[0].annotate(val_counts.iloc[i],
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='bottom', xytext=(0,10),
                             textcoords='offset points',color='red')
    
    # Barplot
    df = pd.concat([X_train,y_train],axis=1).groupby(feat)[target].mean()*100
    df = df.reindex(unique_values)  # Reindex to match the order
    sns.barplot(x=df.index, y=df.values, palette=palette, ax=axes[1])
    axes[1].set_ylabel('Default %')
    axes[1].xaxis.grid(False)

    fig.tight_layout()
    plt.show()
    
    return

# --------------------------------
# Customized description for categorical features

def cat_summary(feat):
    
    # Create an empty summary
    columns = ['dtype', 'count', 'unique', 'missing_count',
               'missing_percentage']
    summary = pd.DataFrame(index=[feat],columns=columns,dtype=float)
    
    # Pull the feature column in question
    col = X_train[feat].copy()
    
    # Count
    summary.loc[feat,'count'] = col.count()

    # Number of unique values
    summary.loc[feat,'unique'] = col.nunique()

    # Missing values count
    summary.loc[feat,'missing_count'] = col.isnull().sum()

    # Missing values percentage
    summary.loc[feat,'missing_percentage'] = col.isnull().sum()/len(col)*100
    
    int_cols = ['count', 'unique', 'missing_count']
    summary[int_cols] = summary[int_cols].astype(int)
    summary = summary.round(2).astype(str)

    # Data type
    summary.loc[feat,'dtype'] = col.dtypes
    
    return display(summary)

# --------------------------------
# Customized plots for categorical features

def cat_plots(feat):
    
    col = X_train[feat].copy()
    
    n_rows = 1
    n_cols = 2

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6.4 * n_cols, 4.8 * n_rows))
    
    # Value counts
    val_counts = col.dropna().value_counts()
    
    # Unique values
    unique_values = val_counts.index

    # Countplot with sorted order
    sns.countplot(x=col, order=unique_values, palette=palette, ax=axes[0])
    axes[0].xaxis.grid(False)

    val_counts_pct = val_counts/len(col)*100
    
    # Show count value if rare (less than 1%)
    lp_thresh = 1
    for i, p in enumerate(axes[0].patches):
        pct = val_counts_pct.iloc[i]
        axes[0].annotate(f'{pct:.2f}%',
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', xytext=(0,0),
                         textcoords='offset points')
        if pct < lp_thresh:
            axes[0].annotate(val_counts.iloc[i],
                             (p.get_x() + p.get_width()/2., p.get_height()),
                             ha='center', va='bottom', xytext=(0,10),
                             textcoords='offset points',color='red')
            
    # Barplot with the same order
    df = pd.concat([X_train,y_train],axis=1).groupby(feat)[target]\
        .mean()*100
    sns.barplot(x=df.index, y=df.values, order=unique_values, palette=palette,
                ax=axes[1])
    axes[1].set_ylabel('Default %')
    axes[1].xaxis.grid(False)

    fig.tight_layout()
    plt.show()
    
    return

# --------------------------------
# Plot for the missing flag associated with a feature

def missing_flag_plot(feat):
    col = X_train[feat].isnull().astype(int)

    if not col.sum():
        return

    df = (pd.concat([col,y_train],axis=1).groupby(feat).mean()*100)\
        .reset_index()
    cols = [f'MISSING_FLAG_{feat}', 'Default %']
    df.columns = cols
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = sns.barplot(data=df,x=cols[0], y=cols[1], palette=palette)
    
    fig.tight_layout()
    plt.show()
    
    return



###############################################################################
##### Exploratory Data Analysis: Part II ######################################
###############################################################################
'''
We perform exploratory data analysis (EDA) on different feature groups by following a routine:

- List the data types in the feature group to adjust our expectation as to what we're dealing with.
- Provide detailed descriptions and vistualizations for all the features in the group.
- Explore the correlations between the features within the group.
- Summarize our decisions with respect to the feature group.

The code blocks corresponding to some of the feature groups are commented to limit the number of plots 
shown in this notebook. However, the analysis and decisions, which are based on the output of the commented blocks, 
are left intact.
'''
feature_groups = {'Demographics': demographics,
                  'Family Count': count,
                  'Age / Duration': duration,
                  'Social Circle': social,
                  'Contact Info': contact,
                  'Address Discrepancy': address,
                  "Region's Data": region,
                  'Process Start Time': process,
                  'External Source Scores': external,
                  'Amounts': amount,
                  'Recent Inquiries': inquiry,
                  'Provided Documents': document,
                  "Building's Data": building
                 }


### Feature Group: Demographics ###############################################

# key = 'Demographics'
# group = feature_groups[key].copy()
# pd.concat([X_train[group].dtypes.rename('dtype'),X_train[group].nunique().rename('unique')],axis=1)

# for feat in group:
#     summary(feat)
#     print('-'*45,'\n')

# corr_heatmap(key)

'''
- The value 'XNA' in 'CODE_GENDER' is virtually non-existent (only 4 instances). It can be replaced by either 
  'M' or 'F'.
- 'NAME_TYPE_SUITE' has some missing values (0.42%). They can form a new category.
- 'OCCUPATION_TYPE' has many missing values (31.35%). They can also form a new category.
- 'NAME_INCOME_TYPE' has several extremely rare categories. They can be merged into a single category.
- 'NAME_FAMILY_STATUS' has a rare category ('Unknown'). It can simply merge into 'Married'.
- 'OCCUPATION_TYPE' and 'ORGANIZATION_TYPE' have many categories. This can cause problems for some slow 
  classifiers as well as those prone to overfitting.
'''

### Feature Group: Family Count ###############################################

key = 'Family Count'
group = feature_groups[key].copy()
pd.concat([X_train[group].dtypes.rename('dtype'),X_train[group].nunique().rename('unique')],axis=1)

for feat in group:
    summary(feat)
    print('-'*45,'\n')

corr_heatmap(key)

'''
- 'CNT_CHILDREN' and 'CNT_FAM_MEMBERS' are highly correlated. The latter can be dropped, 
  as the former has no missing values and a higher correlation with the target.
- In case 'CNT_FAM_MEMBERS' isn't dropped, its couple of missing values can simply be replaced by 2.0.
- All the values of 'CNT_CHILDREN' that are greater than 4 can simply be replaced by 4. (winsorize)
- All the values of 'CNT_FAM_MEMBERS' that are greater than 6 can simply be replaced by 6. (winsorize)
'''


### Feature Group: Age / Duration #############################################

key = 'Age / Duration'
group = feature_groups[key].copy()
pd.concat([X_train[group].dtypes.rename('dtype'),X_train[group].nunique().rename('unique')],axis=1)

for feat in group:
    summary(feat)
    print('-'*45,'\n')

corr_heatmap(key)

'''
- 'DAYS_EMPLOYED' must further be investigated. There are positive values whose meaning must be determined. 
  This will be done next.
- 'OWN_CAR_AGE' has many missing values (65.99%). The missing value simply indicates that the applicant 
  doesn't own a car.
- 'DAYS_LAST_PHONE_CHANGE' has a missing value. It can be replaced by 0.0.(mode)
- 'DAYS_BIRTH' and 'DAYS_EMPLOYED' may be highly correlated (after treating the positive values of 'DAYS_EMPLOYED), 
   but it's not reasonable to drop either of them as one expects age and employment to be 
   strong predictors of default risk.
'''

pd.DataFrame(X_train.query('DAYS_EMPLOYED > 0')
             [['DAYS_EMPLOYED','OCCUPATION_TYPE']]
             .value_counts(dropna=False).rename('Count'))

'''
- 'DAYS_EMPLOYED' can only have one positive value, that is 365243.
- Where it occurs, which is often, the value of 'OCCUPATION_TYPE' is missing (except for two instances).
- It appears that the positive value of 'DAYS_EMPLOYED' simply indicates unemployment. 
  Therefore, it's reasonable to just replace this value with 0. 
  Let's also take a look at the distribution of 'DAYS_EMPLOYED' where it's values are negative.
'''

X_train_copy = X_train.copy()

X_train = X_train.query('DAYS_EMPLOYED <= 0')
summary('DAYS_EMPLOYED')

X_train = X_train_copy.copy()

del X_train_copy
gc.collect();


### Feature Group: Social Circle ##############################################

# key = 'Social Circle'
# group = feature_groups[key].copy()
# pd.concat([X_train[group].dtypes.rename('dtype'),X_train[group].nunique().rename('unique')],axis=1)

# for feat in group:
#     summary(feat)
#     print('-'*45,'\n')

# corr_heatmap(key)

'''
- All the features in this group have some missing values (0.33%). They can be replaced by 0.
- 'OBS_30_CNT_SOCIAL_CIRCLE' and 'OBS_30_CNT_SOCIAL_CIRCLE' have perfect correlation. One can be droped.
'''


###Feature Group: Contact Info ################################################

# key = 'Contact Info'
# group = feature_groups[key].copy()
# pd.concat([X_train[group].dtypes.rename('dtype'),X_train[group].nunique().rename('unique')],axis=1)

# for feat in group:
#     summary(feat)
#     print('-'*45,'\n')

# corr_heatmap(key)

'''
Since 'FLAG_MOBIL' and 'FLAG_CONT_MOBILE' are virtually constant, they can be dropped.
'''


### Feature Group: Address Discrepancy ########################################

key = 'Address Discrepancy'
group = feature_groups[key].copy()
pd.concat([X_train[group].dtypes.rename('dtype'),X_train[group].nunique().rename('unique')],axis=1)

for feat in group:
    summary(feat)
    print('-'*45,'\n')

corr_heatmap(key)

'''
- The high correlation between 'REG_REGION_NOT_WORK_REGION' and 'LIVE_REGION_NOT_WORK_REGION' (0.86) 
  suggests that one can be dropped.
- The high correlation between 'REG_CITY_NOT_LIVE_CITY' and 'REG_CITY_NOT_WORK_CITY' (0.83) 
  suggests that one can be dropped.
- Considering the missing percentages (0% for all features) and correlations with target, 
  if we were to drop features,'LIVE_REGION_NOT_WORK_REGION' and 'REG_CITY_NOT_LIVE_CITY' would be dropped.
'''


### Feature Group: Region's Data ##############################################

# key = "Region's Data"
# group = feature_groups[key].copy()
# pd.concat([X_train[group].dtypes.rename('dtype'),X_train[group].nunique().rename('unique')],axis=1)

# for feat in group:
#     summary(feat)
#     print('-'*45,'\n')

# corr_heatmap(key)

'''
- The high correlation between REGION_RATING_CLIENT and 'REGION_RATING_CLIENT_W_CITY' (0.95) 
  suggests that one can be dropped.
'''


### Feature Group: Process Start Time #########################################

# key = 'Process Start Time'
# group = feature_groups[key].copy()
# pd.concat([X_train[group].dtypes.rename('dtype'),X_train[group].nunique().rename('unique')],axis=1)

# for feat in group:
#     summary(feat)
#     print('-'*45,'\n')

# corr_heatmap(key)

'''
- HOUR_APPR_PROCESS_START can potentially be converted to a cyclic feature, 
  but it seems unnecessarly since the data around 0 and 23 hours is estremely sparce.
'''


### Feature Group: External Source Scores #####################################

key = 'External Source Scores'
group = feature_groups[key].copy()
pd.concat([X_train[group].dtypes.rename('dtype'),X_train[group].nunique().rename('unique')],axis=1)

for feat in group:
    summary(feat)
    print('-'*45,'\n')

corr_heatmap(key)

'''
- All three features contain missing values (EXT_SOURCE_1: 56.38%, EXT_SOURCE_2: 0.21%, EXT_SOURCE_3: 19.83%).
- All three features have high correlations with target (EXT_SOURCE_1: 0.16, EXT_SOURCE_2: 0.16, EXT_SOURCE_3: 0.18) 
  and low correlations between each other. So, none should be dropped.
- The distribution of each of the three features skews right. 
  It appears that replacing the missing values with median is reasonable, 
  but other imputation techniques can also be examined.
'''


### Feature Group: Amounts ####################################################

# key = 'Amounts'
# group = feature_groups[key].copy()
# pd.concat([X_train[group].dtypes.rename('dtype'),X_train[group].nunique().rename('unique')],axis=1)

# for feat in group:
#     summary(feat)
#     print('-'*45,'\n')

# corr_heatmap(key)

'''
- 'AMT_INCOME_TOTAL' contains outliers. They have to be examined, which is done next.
- The high correlation between 'AMT_CREDIT' and 'AMT_GOODS_PRICE' (0.99) suggests that one can be dropped. 
  The former contains no missing value, so the latter appears to be a better choice to drop.
- 'AMT_ANNUITY' has a few missing values (12). They can be imputed via the method of choice 
  or the corresponding rows can be removed. It shouldn't really matter.
'''

# feat = 'AMT_INCOME_TOTAL'
# income_thresh = 750000
# high_income = len(X_train.query(f'{feat} > {income_thresh}'))
# high_income_percentage = high_income / len(X_train[feat]) * 100
# print(f"{high_income} applications with '{feat}' greater than {income_thresh}.")
# print(f"This is {high_income_percentage:.2f}% of the applications.")
# print(f"\nFocusing on where '{feat}' is less than or equal to {income_thresh}:")

# # Make a copy of X_train to later restore it
# X_train_copy = X_train.copy()

# X_train = X_train.query(f'{feat} <= {income_thresh}').copy()
# cont_summary(feat)
# cont_plots(feat=feat,bins=15)

# # Restore X_train
# X_train = X_train_copy.copy()

# del X_train_copy
# gc.collect();

'''
Investigating 'AMT_INCOME_TOTAL' for different thresholds suggest that 
the outliers have not appeared due to erroneous data. 
These outliers should not cause any problem for tree-based models, 
such as LightGBM Classifier that we will use.
'''


### Feature Group: Recent Inquiries ###########################################

# key = 'Recent Inquiries'
# group = feature_groups[key].copy()
# pd.concat([X_train[group].dtypes.rename('dtype'),X_train[group].nunique().rename('unique')],axis=1)

'''
Looking into what these features represent, we alter them a bit to make them more meaningful:

- 'AMT_REQ_CREDIT_BUREAU_HOUR' is the number of inquiries in the past hour. It looks fine.
- 'AMT_REQ_CREDIT_BUREAU_DAY' represents the number of inquiries in the past day, excluding the past hour. 
  For better interpretability, we modify it to include the past hour as well.
- We apply the same process on the rest of the features in this group. For instance, 
  we modify 'AMT_REQ_CREDIT_BUREAU_WEEK' so to also include the inquiries in the past day.
'''

# # Make a copy of X_train to later restore it
# X_train_copy = X_train.copy()

# # Modify the features to make them more interpretable
# for i in range(1,len(group)):
#     X_train[group[i]] = X_train[group[i]] + X_train[group[i-1]]

# for feat in group:
#     summary(feat)
#     print('-'*45,'\n')

# corr_heatmap(key)

# # Restore X_train
# X_train = X_train_copy.copy()

# del X_train_copy
# gc.collect();

'''
- Since 'AMT_REQ_CREDIT_BUREAU_HOUR' and 'AMT_REQ_CREDIT_BUREAU_DAY' are virtually constant, they can be dropped. 
  It's only from 'AMT_REQ_CREDIT_BUREAU_WEEK' that the inquiries start to come in.
- All features in this group miss 13.5% of the values. 
  In fact, it's the same 13.5% of the applications that miss all these feature values. 
  All the missing values can be replaced by 0, while creating a MISSING_FLAG feature is warranted.
- We note that each of these features would generate the same MISSING_FLAG feature. 
  So, only one MISSING_FLAG feature will be created to prevent redundancy.
'''


### Feature Group: Provided Documnets #########################################

# key = 'Provided Documents'
# group = feature_groups[key].copy()
# pd.concat([X_train[group].dtypes.rename('dtype'),X_train[group].nunique().rename('unique')],axis=1)

# for feat in group:
#     summary(feat)
#     print('-'*45,'\n')

# corr_heatmap(key)

'''
- None of the features has any missing value.
- No pair of features are highly correlated.
- 9 of the features are virtually constant and can be dropped:
        'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_10', 
        'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21'.
- 8 of the features are nearly constant and candidates to be dropped:
        'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_13', 
        'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_18'.
'''


###############################################################################
##### Customized Evaluation Criterion: Declic Evaluation ######################
###############################################################################
'''
 Overview:

The Declic Evaluation criterion is a customized evaluation method designed specifically 
for loan default risk assessment models. 
It provides a segmented analysis of the model's predictions, allowing for a detailed examination of its effectiveness 
across different levels of predicted risk.

Functionality:

The Declic Evaluation function takes two essential inputs: 
    the true labels (y_true) indicating whether a loan has led to default or not, 
    and the predicted probabilities (y_pred_proba) generated by the risk assessment model. 
    Additionally, it allows for customization by specifying the number of sets (num_sets) 
    into which the data will be divided for analysis.

Key Features:

    Segmented Analysis: The function divides the dataset into deciles based on the predicted probabilities, 
                        allowing for a segmented analysis of the model's performance.
    Visualization: It generates a bar chart illustrating the average default rate within each decile, 
                    providing a visual representation of the model's effectiveness across different risk segments.
    Comparison to Mean Default Rate: The function compares the default rate within each segment 
                                    to the overall mean default rate of the dataset, 
                                    highlighting areas of relative strength or weakness in the model's predictions.
    Annotations: Annotations on the bars and a horizontal dashed line representing the mean default rate 
                provide additional context and clarity to the assessment results.
    Flexibility: The function allows for the customization of the number of sets, 
                enabling users to adjust the granularity of the analysis based on their specific requirements.

Interpretation:

    A decreasing default rate trend across segments indicates good model performance, 
    as it demonstrates the model's ability to effectively identify and assign higher probabilities to default-prone loans.
    Conversely, increasing or inconsistent default rate trends suggest potential issues with the model's performance, 
    requiring further investigation and potential model refinement.

Benefits:

    Facilitates a detailed assessment of the model's performance across different risk segments.
    Provides actionable insights for fine-tuning the model and optimizing default detection strategies.
    Enables effective resource allocation and risk management based on segmented default rate analysis.

Conclusion:

The Declic Evaluation criterion offers a comprehensive and insightful approach to default risk assessment models, 
providing stakeholders with valuable insights into the model's performance and guiding decision-making processes 
for enhancing risk assessment capabilities.
'''

# Declic Evaluation
def declic_eval(y_true,y_pred_proba,num_sets=10):
    df_y = pd.DataFrame(data= {'true': y_true, 'predicted': y_pred_proba})
    df_y = df_y.sort_values(by='predicted',ascending = False).reset_index()
    step_size = 100 / num_sets
    labels = [f"{i*step_size:.0f}-{(i+1)*step_size:.0f}%" for i in range(num_sets)]
    df_y['set'] = pd.qcut(df_y.index, num_sets, labels=labels)
    
    fig_width, fig_height = 6.4*1.5, 4.8
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = (df_y.groupby('set')['true'].mean()*100).plot\
        (kind='bar',
#          title='Declic Evaluation: Default Rate for Deciles of Sorted Predictions'
        )
    
    ax.axhline(y=y_true.mean()*100, color='red', linestyle='--')
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%',
                    xy=(p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', xytext=(0,0),
                    textcoords='offset points')

    ax.annotate(f'Mean Default Rate: {y_true.mean()*100:.3f}%',
                xy=(fig_width*.5, y.mean()*100), xytext=(0,2),
                ha='left', va='bottom', color='red',
                textcoords='offset points')

    ax.set_ylabel('Default %')
    ax.set_xlabel('Decile of Sorted Predictions')
    ax.xaxis.grid(False)
    
    plt.show()
    
    return


###############################################################################
##### Preprocessing Blocks ####################################################
###############################################################################
'''
We shall use some or all of the following blocks for preprocessing purposes:

MissingFlagger: This class creates a binary 'missing flag' feature for some of the features with missing values. 
                The values 0 and 1 indicate a non-missing and missing values, respectively.

MissingValueFiller: Depending on the selected feature, we may use different techniques to impute the missing values. 
                    This class coelesses coalesces all the necessary imputations given the dataset in hand.

Merger: Some of the features contain rare categories that can be merged together. 
        This class performs the mergeing of categories for various features. 
        This is particularly important when simpler classifiers such as Logistic Regression are to be employed next.

CategoricalConverter: This class converts dtype of categorical features to 'category'. 
                    It is meant to be employed in combination with classifiers capable of natively 
                    dealing with categorical featuers, LightGBM classifier in particular.

CustomOneHotEncoder: This class performs one-hot-encoding, only customized to return a dataframe instead of a numpy array. 
                    It is meant to be employed in combination with traditional classifiers such as Logistic Regression.

CustomScaler: This class performs standardization, only customized to return a dataframe with column names identical to the original ones.
'''

# Create MissingFlagger class
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

# --------------------------------
# Create MissingValueFiller class
class MissingValueFiller(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.median_values = X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 
                                'AMT_ANNUITY', 'AMT_GOODS_PRICE']].median()
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed['NAME_TYPE_SUITE'] = X_transformed['NAME_TYPE_SUITE']\
            .fillna('MISSING')
        X_transformed['OCCUPATION_TYPE'] = X_transformed['OCCUPATION_TYPE']\
            .fillna('MISSING')
        X_transformed['CNT_FAM_MEMBERS'] = X_transformed['CNT_FAM_MEMBERS']\
            .fillna(2)
        X_transformed[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 
                       'AMT_ANNUITY', 'AMT_GOODS_PRICE']] = \
            X_transformed[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 
                           'AMT_ANNUITY', 'AMT_GOODS_PRICE']]\
            .fillna(self.median_values)
        X_transformed = X_transformed.fillna(0)
        return X_transformed

# --------------------------------
# Create Merger class to merge some of the categories of categorical features
class Merger(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        X_transformed['CODE_GENDER'] = X_transformed['CODE_GENDER']\
            .apply(lambda x: 'F' if x == 'XNA' else x)
        X_transformed['NAME_INCOME_TYPE'] = X_transformed['NAME_INCOME_TYPE']\
            .apply(lambda x: 'Other' if x in\
                   {'Unemployed', 'Student', 'Businessman', 'Maternity leave'} else x)
        X_transformed['NAME_FAMILY_STATUS'] = X_transformed['NAME_FAMILY_STATUS']\
            .apply(lambda x: 'Married' if x == 'Unknown' else x)
        X_transformed['CNT_CHILDREN'] = X_transformed['CNT_CHILDREN']\
            .apply(lambda x: 4 if x > 4 else x)
        X_transformed['CNT_FAM_MEMBERS'] = X_transformed['CNT_FAM_MEMBERS']\
            .apply(lambda x: 6 if x > 6 else x)
        X_transformed['DAYS_EMPLOYED'] = X_transformed['DAYS_EMPLOYED']\
            .apply(lambda x: 0 if x > 0 else x)
        return X_transformed

# --------------------------------
# Create CategoricalConverter class, converting dtype of categorical features to 'category'
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

# --------------------------------
# Create CustomOneHotEncoder class for one-hot-encoding all categorical features
# and returning a dataframe
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

        # Concatenate the one-hot-encoded columns with the remaining columns
        X_transformed = pd.concat([X.drop(categorical_columns, axis=1), X_ohe], axis=1).copy()

        return X_transformed

# --------------------------------
# Create CustomScalar class for standardization and column name adjustment
# If no columns given, scales all
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
            X_transformed[self.columns_to_standardize]\
            = self.scaler.transform(X_transformed[self.columns_to_standardize])
        return X_transformed


###############################################################################
##### Model Building: Logistic Regression #####################################
###############################################################################
'''
We create, train and evaluate a pipeline to solve the default risk problem in hand. 
The pipeline includes a preprocessor and a classifier that is a logistic regression model.

- The declic evaluation of the model's performance shows that the model is effective 
  in predicting default risk in the applications.
- The private score of the output is 0.725 when submitted.
'''

# -------- Preprocessor ----------

cols_to_flag = ['OCCUPATION_TYPE', 'EXT_SOURCE_1', 'EXT_SOURCE_2',
                'EXT_SOURCE_3', 'AMT_REQ_CREDIT_BUREAU_YEAR']

cols_to_select = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR',
                  'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE',
                  'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                  'NAME_HOUSING_TYPE', 'CNT_CHILDREN', 'DAYS_BIRTH',
                  'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
                  'DAYS_LAST_PHONE_CHANGE', 'OWN_CAR_AGE',
                  'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
                  'DEF_60_CNT_SOCIAL_CIRCLE', 'FLAG_EMP_PHONE',
                  'FLAG_WORK_PHONE', 'FLAG_PHONE', 'EXT_SOURCE_1',
                  'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AMT_CREDIT',
                  'AMT_ANNUITY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
                  'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
                  'AMT_REQ_CREDIT_BUREAU_YEAR', 'FLAG_DOCUMENT_3',
                  'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_8']

cols_to_select += [f'MISSING_FLAG_{col}' for col in cols_to_flag]

# Create the preprocessor
logreg_preprocessor = Pipeline([
    ('missing_flagger', MissingFlagger(cols_to_flag)),
    ('missing_value_filler', MissingValueFiller()),
    ('merger', Merger()),
    ('column_selector', FunctionTransformer(lambda X: X[cols_to_select], validate=False)),
    ('one_hot_encoder', CustomOneHotEncoder()),
    ('custom_scaler', CustomScaler())
])

# -------- Pipeline --------

# Define logistic regression model
logreg_model = LogisticRegression(class_weight='balanced')

# Create pipeline for logistic regression
logreg_pipeline = Pipeline([
    ('logreg_preprocessor', logreg_preprocessor),
    ('logisticregression', logreg_model)
])

# Define hyperparameter grid
param_grid = {'logisticregression__C': [1]}

# Create StratifiedKFold object
stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create GridSearchCV object
grid_search = GridSearchCV(logreg_pipeline, param_grid, cv=stratified_cv, scoring='roc_auc')

# -------- Training --------

X = train.drop(target,axis=1).copy()
y = train[target].copy()

# Perform grid search
grid_search.fit(X, y)

# Get best model
best_model = grid_search.best_estimator_

# Print best model parameters
print(f"Best Model Parameters:\
{best_model.named_steps['logisticregression'].get_params()}\n")

# -------- Performance --------

cv_results = grid_search.cv_results_

print(f"Mean ROC_AUC (Cross-Validation): {cv_results['mean_test_score'][0]:.4f}\n")

for i in range(stratified_cv.get_n_splits()):
    print(f"fold_{i+1} ROC_AUC: {cv_results[f'split{i}_test_score'][0]:.4f}")

print('\n')
    
# -------- Feature Importance --------

# Get best logistic regression model from pipeline
logreg_model = best_model.named_steps['logisticregression']

# Get feature names from output of logreg_preprocessor
feature_names = best_model.named_steps['logreg_preprocessor'].transform(X).columns

# Extract coefficients from logistic regression model
coefficients = logreg_model.coef_[0]

# Create DataFrame to store feature names and corresponding coefficients
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Number of top features to plot
num_to_plot = 10

# Sort by the absolute value of coefficients to get feature importance
feature_importance_df['Abs_Coefficient'] = feature_importance_df['Coefficient'].abs()
sorted_feature_importance_df = feature_importance_df.sort_values\
(by='Abs_Coefficient', ascending=False).head(num_to_plot)

# Plot top features by coefficient magnitude
ax = sns.barplot(sorted_feature_importance_df,x='Feature',y='Coefficient',palette=palette)
ax.set_title(f'Top {num_to_plot} Features by Coefficient Magnitude')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
ax.set_xlabel('Feature')
ax.set_ylabel('Importance')

fig.tight_layout()
plt.show()

# -------- Submission --------

y_pred_test_proba = best_model.predict_proba(X_test)[:,1]

logreg_submission = submission.copy()
logreg_submission['TARGET'] = y_pred_test_proba

logreg_submission.to_csv('logreg_submission.csv', index = False)


###############################################################################
##### Model Building: LightGBM ################################################
###############################################################################
'''
We now use Light GBM as the classifier after preprocessing the data.

- The declic evaluation of the model's performance shows that the model is effective 
  in predicting default risk in the applications.
- The private score of the output is 0.743 when submitted, 
  indicating better performance than the case where logistic regression is used as the classifier. 
  However, there's room for improvement as we have not yet taken advantage of the data from the previous applications.
'''

# -------- Preprocessing ----------

cols_to_flag = ['OCCUPATION_TYPE', 'EXT_SOURCE_1', 'EXT_SOURCE_2',
                'EXT_SOURCE_3', 'AMT_REQ_CREDIT_BUREAU_YEAR']

cols_to_drop = ['FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'AMT_REQ_CREDIT_BUREAU_HOUR',
                'AMT_REQ_CREDIT_BUREAU_DAY', 'FLAG_DOCUMENT_2',
                'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_10',
                'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_19',
                'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']

# Create the preprocessor

lgbm_preprocessor = make_pipeline(
    MissingFlagger(cols_to_flag),
    MissingValueFiller(),
    Merger(),
    FunctionTransformer(lambda X: X.drop(cols_to_drop, axis=1), validate=False),
    CategoricalConverter()
)

X = train.drop(target,axis=1).copy()
y = train[target].copy()

# Preprocess the data
X_processed = lgbm_preprocessor.fit_transform(X)
X_test_processed = lgbm_preprocessor.transform(X_test)

# Train/validation split with stratified sampling
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

# -------- Training and Prediction ----------

# Define LGBM parameters
params = {
    'objective': 'binary',
    'metric': 'binary_error',
    'num_leaves':11,
    'learning_rate': 0.05,
    'verbose': -1,
    'early_stopping_rounds': 250
}

# Create dataset for LGBM
lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val)

# Train the model
model = lgb.train(params, lgb_train, num_boost_round=1000,
                  valid_sets=[lgb_train, lgb_val])

# # Make probability predictions on train and test
y_pred_train_proba = model.predict\
    (X_train, num_iteration=model.best_iteration, raw_score=True)
y_pred_val_proba = model.predict\
    (X_val, num_iteration=model.best_iteration, raw_score=True)
y_pred_test_proba = model.predict\
    (X_test_processed, num_iteration=model.best_iteration)

# -------- Performance --------

# Training performance
roc_auc_train = roc_auc_score(y_train, y_pred_train_proba)

# Validation performance
roc_auc_val = roc_auc_score(y_val, y_pred_val_proba)

# Print roc_auc_score for train, validation
print(f'\nroc_auc_score:\n- Train: {roc_auc_train}\
    \n- Validation: {roc_auc_val}')

for i, (y_true, y_pred_proba) in enumerate([(y_val, y_pred_val_proba)]):
    print(f"\n{['Validation'][i]} \
Declic Evaluation | Default Rate for Deciles of Sorted Predictions:")
    declic_eval(y_true, y_pred_proba)
    i += 1
    
# -------- Feature Importance --------

# Get feature importances
feature_importance = model.feature_importance()
    
# Create a DataFrame to store feature names and their corresponding coefficients
feature_importance_df = pd.DataFrame({'Feature': X_processed.columns,
                                      'Coefficient': feature_importance})

# Number of top features to plot
num_to_plot = 10

# Sort the DataFrame by the absolute value of coefficients
feature_importance_df['Abs_Coefficient'] =\
    feature_importance_df['Coefficient'].abs()
sorted_feature_importance_df = feature_importance_df.sort_values\
    (by='Abs_Coefficient', ascending=False).head(num_to_plot)

# Plot top features by coefficient magnitude
ax = sns.barplot(sorted_feature_importance_df,
                 x='Feature',y='Coefficient',palette=palette)
ax.set_title('Top Features by Coefficient Magnitude')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
ax.set_xlabel('Feature')
ax.set_ylabel('Importance')

fig.tight_layout()
plt.show()

# -------- Submission --------

lgbm_submission = submission.copy()
lgbm_submission['TARGET'] = y_pred_test_proba

lgbm_submission.to_csv('lgbm_submission.csv', index = False)


###############################################################################
###############################################################################
###############################################################################

##### SHAP Values #############################################################
import shap

# Train a LightGBM model (already trained in your case)
model = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=[lgb_train, lgb_val])

# Create a SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_processed)

# Plot SHAP summary
shap.summary_plot(shap_values, X_test_processed)


##### Gini Coefficient ########################################################
import numpy as np

def gini(actual, pred):
    assert len(actual) == len(pred)
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    total_losses = all[:, 0].sum()
    gini_sum = all[:, 0].cumsum().sum() / total_losses

    gini_sum -= (len(actual) + 1) / 2.
    return gini_sum / len(actual)

def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

# Compute Gini coefficient
gini_coefficient = gini_normalized(y_val, y_pred_val_proba)
print(f'Gini Coefficient: {gini_coefficient}')


##### C-Statistic (ROC-AUC) ###################################################

##### Replacing Declic Evaluation #############################################
# -------- Performance with SHAP Values --------

# Create a SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

# Plot SHAP summary
shap.summary_plot(shap_values, X_val)


##### Integration ##############################################################
import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from IPython.display import display
import shap

# Define file path
DATA_PATH = r'C:\Users\ieron\Desktop\python'

# Load datasets
train = pd.read_csv(os.path.join(DATA_PATH, 'application_train.csv'), index_col='SK_ID_CURR')
X_test = pd.read_csv(os.path.join(DATA_PATH, 'application_test.csv'), index_col='SK_ID_CURR')
submission = pd.read_csv(os.path.join(DATA_PATH, 'application_test.csv'))[['SK_ID_CURR']]

# Define target variable
target = 'TARGET'

# Check if the target column is in the DataFrame
if target in train.columns:
    X_train = train.drop(target, axis=1).copy()
    y_train = train[target].copy()
else:
    raise KeyError(f"Column '{target}' not found in the DataFrame")

# Include the target column in the training DataFrame for WoE and IV calculation
X_train_with_target = train.copy()

# Preprocessing and feature engineering steps here
# Define preprocessor classes, e.g., MissingFlagger, MissingValueFiller, etc.

# Example of preprocessing and LightGBM training
cols_to_flag = ['OCCUPATION_TYPE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AMT_REQ_CREDIT_BUREAU_YEAR']
cols_to_drop = ['FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']

# Preprocessor pipeline
lgbm_preprocessor = make_pipeline(
    MissingFlagger(cols_to_flag),
    MissingValueFiller(),
    Merger(),
    FunctionTransformer(lambda X: X.drop(cols_to_drop, axis=1), validate=False),
    CategoricalConverter()
)

X = train.drop(target, axis=1).copy()
y = train[target].copy()

# Preprocess the data
X_processed = lgbm_preprocessor.fit_transform(X)
X_test_processed = lgbm_preprocessor.transform(X_test)

# Train/validation split with stratified sampling
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

# LightGBM parameters
params = {
    'objective': 'binary',
    'metric': 'binary_error',
    'num_leaves': 11,
    'learning_rate': 0.05,
    'verbose': -1,
    'early_stopping_rounds': 250
}

# Create dataset for LightGBM
lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val)

# Train the model
model = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=[lgb_train, lgb_val])

# Predictions
y_pred_train_proba = model.predict(X_train, num_iteration=model.best_iteration)
y_pred_val_proba = model.predict(X_val, num_iteration=model.best_iteration)
y_pred_test_proba = model.predict(X_test_processed, num_iteration=model.best_iteration)

# Compute Gini coefficient
def gini(actual, pred):
    assert len(actual) == len(pred)
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    total_losses = all[:, 0].sum()
    gini_sum = all[:, 0].cumsum().sum() / total_losses

    gini_sum -= (len(actual) + 1) / 2.
    return gini_sum / len(actual)

def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

gini_coefficient = gini_normalized(y_val, y_pred_val_proba)
print(f'Gini Coefficient: {gini_coefficient}')

# Compute ROC-AUC score
roc_auc = roc_auc_score(y_val, y_pred_val_proba)
print(f'ROC-AUC Score: {roc_auc}')

# SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)
shap.summary_plot(shap_values, X_val)

# WoE and IV calculation for categorical features
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

def calculate_woe_iv(df, feature, target):
    event = df[target].sum()
    non_event = len(df) - event
    feature_event_rate = df.groupby(feature)[target].sum() / event
    feature_non_event_rate = (df.groupby(feature)[target].count() - df.groupby(feature)[target].sum()) / non_event
    woe = np.log(feature_event_rate / feature_non_event_rate)
    iv = (feature_event_rate - feature_non_event_rate) * woe
    woe_iv_df = pd.DataFrame({'WoE': woe, 'IV': iv})
    iv_sum = iv.sum()
    return woe_iv_df, iv_sum

woe_iv_dict = {}
for feature in categorical_features:
    woe_iv_df, iv_sum = calculate_woe_iv(X_train_with_target, feature, target)
    woe_iv_dict[feature] = (woe_iv_df, iv_sum)

for feature, (woe_iv_df, iv_sum) in woe_iv_dict.items():
    print(f"Feature: {feature}\nInformation Value (IV): {iv_sum}\n")
    print(woe_iv_df)
    print('-'*30)

# Feature importance
feature_importance = model.feature_importance()
feature_importance_df = pd.DataFrame({'Feature': X_processed.columns, 'Coefficient': feature_importance})
num_to_plot = 10
feature_importance_df['Abs_Coefficient'] = feature_importance_df['Coefficient'].abs()
sorted_feature_importance_df = feature_importance_df.sort_values(by='Abs_Coefficient', ascending=False).head(num_to_plot)

ax = sns.barplot(data=sorted_feature_importance_df, x='Feature', y='Coefficient', palette=palette)
ax.set_title('Top Features by Coefficient Magnitude')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
ax.set_xlabel('Feature')
ax.set_ylabel('Importance')

plt.tight_layout()
plt.show()

# Submission
lgbm_submission = submission.copy()
lgbm_submission['TARGET'] = y_pred_test_proba
lgbm_submission.to_csv('lgbm_submission.csv', index=False)



##############################################################################
#############################################################################
#############################################################################


