import pandas as pd
import numpy as np
import seaborn as sns
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import scipy.stats as stats
warnings.filterwarnings('ignore')
from datetime import datetime
from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print(os.listdir())

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print('数据已经读取完成')
quantitative = [i for i in train.columns if train.dtypes[i]!='object'] # 取定量指标 （其中包含ID[无用]，SalePrice[需要预测，为y]）
quantitative.remove('SalePrice')
quantitative.remove('Id')
qualitive = [i for i in train.columns if train.dtypes[i] =='object'] #取定性指标
plt.figure(figsize=(15, 8))
sns.set_style('whitegrid')
missing = train.isnull().sum()
missing = missing[missing>0]
missing.sort_values(inplace=True)
missing.plot.bar()
y = train['SalePrice']
plt.figure(1)
plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=stats.johnsonsu)
plt.figure(2)
plt.title('Normal')
sns.distplot(y, kde=False, fit=stats.norm)
plt.figure(3)
plt.title('Log Normal')
sns.distplot(y, kde=False, fit=stats.lognorm)
test_normality = lambda x: stats.shapiro(x.fillna(int(x.median())))[1] < 0.01
normal = pd.DataFrame(train[quantitative])
normal = normal.apply(test_normality)
print(not normal.any())  # 证明跟正态分布关系不大
stats.shapiro(train[quantitative[0:5]].fillna(0))[1]


def encode(frame, feature):
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    ordering['spmean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
    ordering = ordering.sort_values('spmean')
    ordering['ordering'] = range(1, ordering.shape[0] + 1)
    ordering = ordering['ordering'].to_dict()  # 每个特征所对应的房价的均值 由高到低 标注为1、2、3 生成特征1：1，特征2：2的字典

    for cat, o in ordering.items():
        frame.loc[frame[feature] == cat, feature + '_E'] = o

# qual_encoded = []
# for q in qualitive:
#     encode(train, q)
#     qual_encoded.append(q+'_E')
# print(qual_encoded)


def spearman(frame, features):
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['spearman'] = [frame[f].corr(frame['SalePrice'], 'spearman') for f in features]
    spr = spr.sort_values('spearman')
    plt.figure(figsize=(6, 0.25 * len(features)))
    sns.barplot(data=spr, y='feature', x='spearman', orient='h')


# features = quantitative + qual_encoded
features = qualitive + quantitative
from sklearn.manifold import TSNE # 降维
from sklearn.preprocessing import StandardScaler # 标准化
model = TSNE(perplexity=50)
from sklearn.cluster import KMeans

# from sklearn.decomposition import PCA
# X = train[features].fillna(0).values
# tsne = model.fit_transform(X)
#
# std = StandardScaler()
# st = std.fit_transform(X)
# PCA = PCA(n_components=30)
# pca = PCA.fit_transform(st)
# kmeans = KMeans(n_clusters=5)
# kmeans.fit(pca)
#
# fr = pd.DataFrame(
#     {'tsne1':tsne[:,0],
#      'tsne2':tsne[:,1],
#      'cluster':kmeans.labels_ # tesne 降维后，用k近邻标注
#     }
# )
# sns.lmplot(data=fr, x='tsne1', y='tsne2', hue='cluster', fit_reg=False)
# plt.show()

train['label_'] =1
train.drop(['Id'],axis=1,inplace=True)
Id = test.Id
test['label_'] = 0
test.drop(['Id'],axis=1,inplace=True)

train = train[train.GrLivArea<4000]
train.reset_index(drop=True, inplace=True)

train_features = train.drop(['SalePrice'], axis=1)
test_features = test
features = pd.concat([train_features, test_features]).reset_index(drop=True)

train.reset_index(drop=True, inplace=True)
y = np.log1p(train['SalePrice']).reset_index(drop=True)# 对y进行标准化
for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:
    features[col] = features[col].fillna(0)
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:
    features[col] = features[col].fillna('None')
features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

object_ = []
numeries = []
for i in features.columns:
    if features[i].dtypes ==object:
        object_.append(i)
    else:
        numeries.append(i)
features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
features.update(features[object_].fillna('None'))
features.update(features[numeries].fillna(0))

from scipy.stats import skew
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics2 = []
for i in features.columns:
    if features[i].dtype in numeric_dtypes:
        numerics2.append(i)
skew_feature = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_feature[skew_feature>0.5]

skew_index = high_skew.index
for i in skew_index:
    features[i] = boxcox1p(features[i], boxcox_normmax(features[i]+1))

features = features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)

features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']
features['TotalSF']=features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']

features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +
                                 features['1stFlrSF'] + features['2ndFlrSF'])

features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +
                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))

features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +
                              features['EnclosedPorch'] + features['ScreenPorch'] +
                              features['WoodDeckSF'])
features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x>0 else 0)
features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

final_features = pd.get_dummies(features).reset_index(drop=True)

test_x = final_features[final_features['label_']==0]
train_x = final_features[final_features['label_']==1]

test_x.drop(['label_'],axis=1, inplace=True)
train_x.drop(['label_'],axis=1, inplace=True)
final_features.drop(['label_'], axis=1, inplace=True)

X = final_features.iloc[:len(y),:]
X_sub = final_features.iloc[len(y):,:]

outliers = [30, 88, 462, 631, 1322]
X = X.drop(X.index[outliers])
y = y.drop(y.index[outliers])
# Y = Y.drop(y.index[outliers])

overfit = []
for i in X.columns:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 99.94:
        overfit.append(i) # 删除只有一个值或者一个值占比99.94的特征

overfit = list(overfit)
X = X.drop(overfit, axis=1)
X_sub = X_sub.drop(overfit, axis=1)

test_x = test_x.drop(overfit, axis=1)

Kfold = KFold(n_splits=10, shuffle=True, random_state=42)
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=Kfold))
    return rmse

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=Kfold))
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=Kfold))
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=Kfold, l1_ratio=e_l1ratio))
svr = make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003))
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)
lightgbm = LGBMRegressor(objective='regression',
                                       num_leaves=4,
                                       learning_rate=0.01,
                                       n_estimators=5000,
                                       max_bin=200,
                                       bagging_fraction=0.75,
                                       bagging_freq=5,
                                       bagging_seed=7,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=7,
                                       verbose=-1,
                                       )
xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)

score = cv_rmse(ridge)
print(f"Ridge: {score.mean()} ({score.std()})\n{datetime.now()}")
score = cv_rmse(lasso)
print(f"Lasso: {score.mean()} ({score.std()})\n{datetime.now()}")
score = cv_rmse(elasticnet)
print(f"Elasticnet: {score.mean()} ({score.std()})\n{datetime.now()}")
score = cv_rmse(svr)
print(f"SVR: {score.mean()} ({score.std()})\n{datetime.now()}")
score = cv_rmse(lightgbm)
print(f"Lightbgm: {score.mean()} ({score.std()})\n{datetime.now()}")
score = cv_rmse(gbr)
print(f"GBR: {score.mean()} ({score.std()})\n{datetime.now()}")
score = cv_rmse(xgboost)
print(f"XGBoost: {score.mean()} ({score.std()})\n{datetime.now()}")

print("Start fit")
print('stack_gen fit...')
stack_gen_model = stack_gen.fit(np.array(X),np.array(y))
print('elasticnet fit...')
elasticnet_model_full_data = elasticnet.fit(X,y)
print('Lasso fit...')
lasso_model_full_data = lasso.fit(X,y)
print('Ridge fit...')
ridge_model_full_data = ridge.fit(X,y)
print('SVR fit...')
svr_model_full_data = svr.fit(X,y)
print('GradientBoosting fit...')
gbr_model_full_data = gbr.fit(X, y)
print('xgboost fit...')
xgb_model_full_data = xgboost.fit(X, y)
print('lightgbm fit...')
lgb_model_full_data = lightgbm.fit(X, y)

def blend_models_predict(X):
    return ((0.1 * elasticnet_model_full_data.predict(X)) + \
            (0.05 * lasso_model_full_data.predict(X)) + \
            (0.1 * ridge_model_full_data.predict(X)) + \
            (0.1 * svr_model_full_data.predict(X)) + \
            (0.1 * gbr_model_full_data.predict(X)) + \
            (0.15 * xgb_model_full_data.predict(X)) + \
            (0.1 * lgb_model_full_data.predict(X)) + \
            (0.3 * stack_gen_model.predict(np.array(X))))
print('RMSLE score on train data:')
print(rmsle(y, blend_models_predict(X)))

Sale_ = np.expm1(blend_models_predict(X_sub))
# Sale_ = np.expm1(xgb_model_full_data.predict(X_sub))
df = pd.DataFrame(
    {'Id':Id,
     'SalePrice':Sale_,
    }
)
c1 = pd.read_csv('best_submission.csv')
c2 = pd.read_csv('Submission.csv')
df.iloc[:,1] = df.iloc[:,1]/3 + c1.iloc[:,1]/3 + c2.iloc[:,1]/3
df.to_csv('all2.csv',index=False)