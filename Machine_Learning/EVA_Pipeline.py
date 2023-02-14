# 표준화 스케일링
import imp
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn. preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

raw_iris = datasets.load_iris()

X = raw_iris.data
y = raw_iris.target

# 트레이닝 / 테스트 데이터 분할
X_tn, X_te, y_tn, y_te = train_test_split(X,y,random_state=7)

'''
# 표준화 스케일링
std_scale = StandardScaler()
X_tn_std = std_scale.fit_transform(X_tn)
X_te_std = std_scale.transform(X_te)
'''

# 스케일링 대신 파이프라인 사용
linear_pipline = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_regression', LinearRegression())
])

# 학습
linear_pipline.fit(X_tn, y_tn)

# 예측
pred_linear = linear_pipline.predict(X_te)

# 평가
mean_squared_error(y_te, pred_linear)




