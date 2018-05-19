import pandas as pd
import lightgbm as lgb
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.externals import joblib
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from scipy import sparse
import numpy as np
np.set_printoptions(threshold=np.inf)

dir_path=r'C:/Users/Zhihan Zhang/Desktop/算分/preliminary_contest_data/preliminary_contest_data'
params_path=dir_path+r'/params_importance.txt'

ad_feature=pd.read_csv(dir_path+r'/adFeature.csv')
user_feature=pd.read_csv(dir_path+r'/userFeature_tiny_100000.csv')
train=pd.read_csv(dir_path+r'/train_100000.csv')
predict=pd.read_csv(dir_path+r'/test1_100000.csv')

predict['label']=0
data=pd.concat([train,predict])
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
'''
train_pos=data[data.label==1]
train=data[data.label==-1]
for _ in range(10):
    train=pd.concat([train,train_pos])
train=train.reset_index(drop=True)
train=train.sample(frac=1).reset_index(drop=True)
'''
train=data[data.label!=0]
train_y=train.pop('label')

test=data[data.label==0]
res=test[['aid','uid']]
test=test.drop('label',axis=1)
train_x=train[['creativeSize','age','consumptionAbility','education','house']]
test_x=test[['creativeSize','age','consumptionAbility','education','house']]

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

print(train_x.shape)
print(test_x.shape)

cv=HashingVectorizer(n_features=1000)
#cv=CountVectorizer()
for feature in vector_feature:
    cv.fit(data[feature])
    train_a = cv.transform(train[feature])
    test_a = cv.transform(test[feature])
    train_x = sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
print('cv prepared !')

print(train_x.shape)
print(test_x.shape)
del train,test

sfm=SelectFromModel(GradientBoostingClassifier())
sfm.fit(train_x,train_y)
train_x=sfm.transform(train_x)
test_x=sfm.transform(test_x)
print(train_x.shape)
print(test_x.shape)

def LGB_test(train_x,train_y,test_x,res):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,max_bin=150,
        max_depth=-1, n_estimators=1000, objective='binary',
        subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
        learning_rate=0.05,  random_state=2018, n_jobs=-1
    )
    clf.fit(train_x, train_y,eval_set=[(train_x,train_y)],eval_metric='auc',early_stopping_rounds=100)
    res['score'] = clf.predict_proba(test_x)[:,1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv(dir_path+r'/submission_1.csv', index=False)
    #os.system('zip ../data/baseline.zip ../data/submission.csv')
    return clf

clf=LGB_test(train_x,train_y,test_x,res)


#params_file=open(params_path,'w')
#print(clf.feature_importances_,file=params_file)
'''
imp=clf.feature_importances_
train_x=np.delete(train_x.toarray(),[i for i in range(len(imp)) if imp[i]==0],axis=1)
test_x=np.delete(test_x.toarray(),[i for i in range(len(imp)) if imp[i]==0],axis=1)
clf=LGB_test(train_x,train_y,test_x,res)
'''
#joblib.dump(clf,'./clf.model')
