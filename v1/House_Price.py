import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

#预处理
train = pd.read_csv(r'data\train.csv')

test = pd.read_csv(r'data\test.csv')
Id = test.Id
y = train['SalePrice']
x = train.drop(['SalePrice'], axis=1)
x['sheet'] = 1
test['sheet'] = 0
df = pd.concat([x, test], axis=0,ignore_index=True)
# 缺失值处理
col = ['Alley', 'MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', ]
for i in col:
    df[i].fillna('NaN', inplace=True)
df['LotFrontage'].fillna(int(df['LotFrontage'].median()))
df['MasVnrArea'].fillna(int(df['MasVnrArea'].median()))
df['BsmtFinSF1'].fillna(int(df['BsmtFinSF1'].median()))
df['BsmtFinSF2'].fillna(int(df['BsmtFinSF2'].median()))
df['BsmtUnfSF'].fillna(int(df['BsmtUnfSF'].median()))
df['TotalBsmtSF'].fillna(int(df['TotalBsmtSF'].median()))
df['GarageYrBlt'].fillna(int(df['GarageYrBlt'].median()))
df['GarageArea'].fillna(int(df['GarageArea'].median()))
df['BsmtFullBath'].fillna(np.random.randint(0, 3))
df['BsmtHalfBath'].fillna(np.random.randint(0, 2))
df['GarageCars'].fillna(np.random.randint(0, 5))
# 特征处理
col1 = ['RoofMatl', 'ExterQual', 'Street', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC', 'CentralAir', 'PavedDrive', 'SaleCondition', 'Alley', 'MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', ]
df = pd.get_dummies(df, columns=col1)
p_x = df['sheet'] == 1
x = df.loc[p_x, :]
test = df.loc[~p_x, :]
x = x.drop(['sheet', 'Id'], axis=1)
test = test.drop(['sheet', 'Id'], axis=1)
# 回归预测
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=.30)
cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)
n_estimator = range(100, 1500, 100)
param_ = {
    'n_eastimator':n_estimator,
}

XGB = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
XGB.fit(x, y)
h_price = XGB.predict(test)


result = pd.DataFrame({
    'Id': Id,
    'SalePrice': h_price,
})
result.to_csv('result.csv')