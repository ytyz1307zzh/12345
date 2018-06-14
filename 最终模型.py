import pandas as pd
import lightgbm as lgb
import time
from imblearn.over_sampling import SMOTE
import numpy as np
from xgboost import XGBClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer
from sklearn.feature_selection import SelectKBest,chi2,f_classif,VarianceThreshold,SelectFromModel
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from scipy import sparse
np.set_printoptions(threshold=np.inf)
dir_path=r'./preliminary_contest_data'

feature_path=dir_path+r'/feature_importance.txt'
feature_file=open(feature_path,'r')
imp=feature_file.readline()
imp=[int(x) for x in imp.split()]
print(len(imp))
features=[i for i in range(len(imp)) if imp[i]>5]
print(features)
print(len(features))

ad_feature=pd.read_csv(dir_path+r'/adFeature.csv')
user_feature=pd.read_csv(r'../sf2/preliminary_contest_data/userFeature.csv')
data=pd.read_csv(dir_path+r'/train.csv')

data=pd.merge(data,ad_feature,on='aid',how='left')
data=pd.merge(data,user_feature,on='uid',how='left')
data=data.fillna('0')

one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType']
vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']

for feature in one_hot_feature:
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])

data.loc[data['label']==-1,'label']=0
train,test=train_test_split(data,test_size=0.2,random_state=2018)
test_y=test.pop('label')

labels=train['label'].tolist()
print(labels.count(0))
print(labels.count(1))
train_neg=train[train.label==0][:3500000]
train_pos=train[train.label==1]
train=train_neg
for _ in range(10):
    train=pd.concat([train,train_pos])
train=train.reset_index(drop=True)
train=train.sample(frac=1).reset_index(drop=True)
labels=train['label'].tolist()
print(labels.count(0))
print(labels.count(1))
train_y=train.pop('label')


train_x=train[['creativeSize']]
test_x=test[['creativeSize']]
print(train_x.shape)
print(test_x.shape)

enc = OneHotEncoder()
for feature in one_hot_feature:
    enc.fit(data[feature].values.reshape(-1, 1))
    train_a=enc.transform(train[feature].values.reshape(-1, 1))
    test_a = enc.transform(test[feature].values.reshape(-1, 1))
    train_x= sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
print('one-hot prepared !')

cv=CountVectorizer()
for feature in vector_feature:
    cv.fit(data[feature])
    train_a = cv.transform(train[feature])
    test_a = cv.transform(test[feature])
    train_x = sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
print('cv prepared !')

print(train_x.shape)
print(test_x.shape)

train_x=train_x.tocsc()
test_x=test_x.tocsc()

train_x_new=train_x[:,features]
test_x_new=test_x[:,features]
train_x_new=train_x_new.tocoo()
test_x_new=test_x_new.tocoo()
print(test_x_new.shape)
print(train_x_new.shape)

def LGB_test(train_x,train_y,test_x,test_y):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=50, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=3000, objective='binary',
        subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
        learning_rate=0.1, min_child_weight=50, random_state=2018, n_jobs=-1
    )
    clf.fit(train_x, train_y,eval_set=[(train_x,train_y),(test_x,test_y)],eval_metric='auc',early_stopping_rounds=200)
    return clf,clf.best_score_

clf,best_score=LGB_test(train_x_new,train_y,test_x_new,test_y)
score_output=open(dir_path+r'/best_score.txt','w')
print(best_score,file=score_output)


def stack_test(train_x,train_y,test_x,test_y):
    print("start stacking test")
    clf1 = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=50, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=2000, objective='binary', min_child_weight=50,
        subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
        learning_rate=0.1,  random_state=2018, n_jobs=-1
    )
    clf2 = lgb.LGBMClassifier(
        boosting_type='dart', num_leaves=50, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=2000, objective='binary', min_child_weight=50,
        subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
        learning_rate=0.1,  random_state=2018, n_jobs=-1
    )
    clf3 = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=50, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=2000, objective='binary', min_child_weight=50,
        subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
        learning_rate=0.1,  random_state=2018, n_jobs=-1
    )

    clf4=XGBClassifier(
        max_depth=5, learning_rate=0.1, n_estimators=2000, objective='binary:logistic',
        booster='gbtree', n_jobs=-1, min_child_weight=50,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0, reg_lambda=1,random_state=2018
     )

    stack_clf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                          meta_classifier=clf4,use_probas=True,average_probas=True,verbose=1)

    stack_clf.fit(train_x,train_y)
    pred_score = stack_clf.predict_proba(test_x)[:,1]
    auc_score = roc_auc_score(test_y,pred_score)
    output=open(dir_path+r'/auc_score.txt','w')
    print("auc score is {}".format(auc_score),file=output)
    print("auc score is {}".format(auc_score))

    return stack_clf

stack_clf=stack_test(train_x_new,train_y,test_x_new,test_y)

