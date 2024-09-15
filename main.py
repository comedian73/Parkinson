import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb

df = pd.read_csv('./input/parkinsons.data')

all_features = df.loc[:, df.columns != 'status'].values[:, 1:]
y = df.loc[:, 'status'].values

scaler = MinMaxScaler((-1, 1))
X = scaler.fit_transform(all_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)

xgb_clf = xgb.XGBClassifier()
xgb_clf = xgb_clf.fit(X_train, y_train)
print('Точность XGBoost Classifier на обучающих данных составляет : {:.2f}'.format(xgb_clf.score(X_train, y_train)*100))
print('Точность XGBoost Classifier на тестовых данных составляет : {:.2f}'.format(xgb_clf.score(X_test, y_test)*100))
