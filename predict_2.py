import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.externals import joblib
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
#from xgboost import XGBClassifier
from scipy import sparse
import numpy as np
np.set_printoptions(threshold=np.inf)

dir_path=r'C:/Users/Zhihan Zhang/Desktop/算分/preliminary_contest_data/preliminary_contest_data'

ad_feature=pd.read_csv(dir_path+r'/adFeature.csv')
user_feature=pd.read_csv(dir_path+r'/userFeature_tiny.csv')
train=pd.read_csv(dir_path+r'/train_10000.csv')
predict=pd.read_csv(dir_path+r'/test_10000.csv')

predict['label']=0
data=pd.concat([train,predict]).reset_index(drop=True)
data=pd.merge(data,ad_feature,on='aid',how='left')
data=pd.merge(data,user_feature,on='uid',how='left')

data['house']=data['house'].fillna(0)
data=data.fillna('0')
one_hot_feature=['LBS','carrier','os','ct','marriageStatus','gender','creativeId',
                 'advertiserId','campaignId', 'productType','adCategoryId', 'productId']
vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4',
                'interest5','kw1','kw2','kw3','topic1','topic2','topic3']
other_features=['creativeSize','age','consumptionAbility','education','house']

for feature in one_hot_feature:
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])

for feature in other_features:
    data[feature]=data[feature].apply(int)

train=data[data.label!=0]
train_y=train.pop('label')

predict=data[data.label==0]
res=predict[['aid','uid']]
predict=predict.drop('label',axis=1)

train_x=train[['creativeSize','age','consumptionAbility','education','house']]
predict_x=predict[['creativeSize','age','consumptionAbility','education','house']]

print(train_x.shape)
print(predict_x.shape)

enc = OneHotEncoder()
for feature in one_hot_feature:
    enc.fit(data[feature].values.reshape(-1, 1))
    train_a=enc.transform(train[feature].values.reshape(-1, 1))
    predict_a=enc.transform(predict[feature].values.reshape(-1, 1))
    train_x= sparse.hstack((train_x, train_a))
    predict_x= sparse.hstack((predict_x, predict_a))
print('one-hot prepared !')

print(train_x.shape)
print(predict_x.shape)

cv=HashingVectorizer(n_features=1000)
#cv=CountVectorizer()
for feature in vector_feature:
    cv.fit(data[feature])
    train_a = cv.transform(train[feature])
    predict_a = cv.transform(predict[feature])
    train_x = sparse.hstack((train_x, train_a))
    predict_x = sparse.hstack((predict_x, predict_a))
print('cv prepared !')

print(train_x.shape)
print(predict_x.shape)

sfm=SelectFromModel(GradientBoostingClassifier())
sfm.fit(train_x,train_y)
train_x=sfm.transform(train_x)
predict_x=sfm.transform(predict_x)

print(train_x.shape)
print(predict_x.shape)

def stack_test(train_x,train_y,predict_x,res):
    print("start test")
    clf1 = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,max_bin=150,
        max_depth=-1, n_estimators=500, objective='binary',
        subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
        learning_rate=0.1,  random_state=2018, n_jobs=-1
    )
    clf2 = lgb.LGBMClassifier(
        boosting_type='dart', num_leaves=31, reg_alpha=0.0, reg_lambda=1,max_bin=150,
        max_depth=-1, n_estimators=500, objective='binary',
        subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
        learning_rate=0.1,  random_state=2018, n_jobs=-1
    )
    clf3=MLPClassifier(hidden_layer_sizes=(100,100,100),activation='relu',
                       solver='adam',alpha=0.001,random_state=2018,learning_rate_init=0.1
                       )
    clf4 = lgb.LGBMClassifier(
        boosting_type='rf', num_leaves=31, reg_alpha=0.0, reg_lambda=1,max_bin=150,
        max_depth=-1, n_estimators=500, objective='binary',
        subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
        learning_rate=0.1,  random_state=2018, n_jobs=-1
    )
    clf5=MLPClassifier(hidden_layer_sizes=(100,100,100),activation='relu',learning_rate='invscaling',
                       solver='sgd',alpha=0.001,random_state=2018,learning_rate_init=0.1
                       )
    '''
    clf5=XGBClassifier(
        max_depth=5, learning_rate=0.1, n_estimators=500, objective='binary:logistic',
        booster='gbtree', n_jobs=-1, min_child_weight=5,scale_pos_weight=10,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0, reg_lambda=1,random_state=2018
     )
     
    clf6=XGBClassifier(
        max_depth=5, learning_rate=0.1, n_estimators=500, objective='binary:logistic',
        booster='gbtree', n_jobs=-1, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0, reg_lambda=1,random_state=2018
     )
     '''
    clf6 = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,max_bin=150,
        max_depth=-1, n_estimators=500, objective='binary',
        subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
        learning_rate=0.1,  random_state=2018, n_jobs=-1
    )
    stack_clf = StackingClassifier(classifiers=[clf1, clf2, clf3,clf4,clf5],
                          meta_classifier=clf6,use_probas=True,verbose=1)
    stack_clf.fit(train_x,train_y)
    res['score'] = stack_clf.predict_proba(predict_x)[:,1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv(dir_path+r'/submission.csv', index=False)
    '''
    for clf, label in zip([clf1, clf2, clf3, clf4,stack_clf],
                      ['lgbm1', 'lgbm2', 'mlp','lgbm3', 'stack_clf']):
        scores =cross_val_score(clf, train_x, train_y, cv=4, scoring='f1')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))
          '''
    return stack_clf

stack_clf=stack_test(train_x,train_y,predict_x,res)

