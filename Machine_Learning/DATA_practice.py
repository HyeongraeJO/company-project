# 머신러닝 라이브러리
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets 


'''
# 꽃 구분하기
raw_iris = datasets.load_iris()
X_iris = pd.DataFrame(raw_iris.data)
y_iris = pd.DataFrame(raw_iris.target)
df_iris = pd.concat([X_iris, y_iris], axis=1)

feature_iris = raw_iris.feature_names
print(feature_iris)

col_iris = np.append(feature_iris, ['target'])
df_iris.columns = col_iris
df_iris.head()
print(df_iris.head())

# 와인 구분하기
raw_wine = datasets.load_wine()
X_wine = pd.DataFrame(raw_wine.data)
y_wine = pd.DataFrame(raw_wine.target)
df_wine = pd.concat([X_wine,y_wine], axis = 1)

feature_wine = raw_wine.feature_names
print(feature_wine)

col_wine = np.append(feature_wine, ['target'])
df_wine.columns = col_wine
df_wine.head()
print(df_wine.head())


# 당뇨병 예측하기 
raw_diab = datasets.load_diabetes()
X_diab = pd.DataFrame(raw_diab.data)
y_diab = pd.DataFrame(raw_diab.target)
df_diab = pd.concat([X_diab, y_diab], axis=1)

feature_diab = raw_diab.feature_names
print(feature_diab)

col_diab=np.append(feature_diab, ['target'])
df_diab.columns = col_diab
df_diab.head()
print(df_diab.head())


# 유방암 예측하기
raw_bc = datasets.load_breast_cancer()
X_bc = pd.DataFrame(raw_bc.data)
y_bc = pd.DataFrame(raw_bc.target)
df_bc = pd.concat([X_bc,y_bc], axis=1)

feature_bc = raw_bc.feature_names
print(feature_bc)

col_bc = np.append(feature_bc, ['target'])
df_bc.columns = col_bc
df_bc.head()
print(df_bc.head())
'''
# 데이터 프레임 생성
df = pd.DataFrame([

    [42, 'male', 12, 'reading', 'class2'],
    [35, 'unknown', 3, 'cooking', 'class1'],
    [1000, 'female', 7, 'cycling', 'class3'],
    [1000, 'unknown', 21, 'unknown', 'unknown']

])
df.columns = ['age', 'gender', 'month_birth', 'hobby', 'target']

print(df)

# 결측치 처리
df['age'].unique()
print(df['age'].unique())
df['gender'].unique()
print(df['gender'].unique())
df['month_birth'].unique()
print(df['month_birth'].unique())
df['hobby'].unique()
print(df['hobby'].unique())
df['target'].unique()
print(df['target'].unique())

# 결측치 나타내기
df.loc[df['age']>150, ['age']] = np.nan
df.loc[df['gender']=='unknown', ['gender']] = np.nan
df.loc[df['month_birth']>12, ['month_birth']] = np.nan
df.loc[df['hobby']=='unknown', ['hobby']] = np.nan
df.loc[df['target']=='unknown', ['target']] = np.nan

print(df)

# 결측치 개수파악
df.isnull().sum()
print(df.isnull().sum())

# 결측치를 포함한 행(row) 삭제
df2 = df.dropna(axis = 0)
print(df2)

# 결측치를 포함한 열(column) 삭제
df3 = df.dropna(axis=1)
print(df3)

# 모든 값이 결측치인 행 삭제
df4 = df.dropna(how='all')
print(df4)

# 값이 2개 미만인 행 삭제
df5 = df.dropna(thresh=2)
print(df5)

# 특정 열에 결측치가 있는 경우 행 삭제
df6 = df.dropna(subset=['gender'])
print(df6)


# 결측치 대체하기 (error????????????????)

alter_values = {'age' : 0, 'gender' : 'U', 'month_birth' : 0, 'hobby' : 'U', 'target' : 'class4'}
df7 = df.fillna(value = alter_values)
print(df7)

# 클래스 라벨 설정 
from sklearn.preprocessing import LabelEncoder
df8 =df7
class_label = LabelEncoder()
data_value = df8['target'].values
y_new = class_label.fit_transform(data_value)
y_new
print(y_new)

df8['target'] = y_new
print(df8)

# 클래스 라벨 설정 후 원래대로
y_ori = class_label.inverse_transform(y_new)
print(y_ori)

df8['target'] = y_ori
print(df8)

# 클래스 라벨링 (사이킷런X)
y_arr = df8['target'].values
y_arr.sort()
y_arr
print(y_arr)

num_y = 0
dic_y ={}
for ith_y in y_arr:
    dic_y[ith_y] = num_y
    num_y += 1

print(dic_y)
df8['target'] = df8['target'].replace(dic_y)
print(df8)

# 원-핫 인코딩
df9 = df8
df9['target'] = df9['target'].astype(str)
df10 = pd.get_dummies(df9['target'])
print(df10)

df9['target'] = df9['target'].astype(str)
df11 = pd.get_dummies(df9['target'], drop_first=True)
print(df11)

df12 = df8
df13 = pd.get_dummies(df12)
print(df13)

# 사이킷런 라이브러리를 이용한 원-핫 인코딩
from sklearn.preprocessing import OneHotEncoder
hot_encoder = OneHotEncoder()
y = df7[['target']]
y_hot = hot_encoder.fit_transform(y)
print(y_hot.toarray())

# 텐서플로 라이브러리를 이용한 원-핫 인코딩
from tensorflow.keras.utils import to_categorical
y_hotec = to_categorical(y)
print(y_hotec)

# 데이터 스케일링

# (1) 표준화 스케일링
from sklearn.preprocessing import StandardScaler

std = StandardScaler()
std.fit(df8[['month_birth']])
x_std = std.transform(df8[['month_birth']])
x_std2 = std.fit_transform(df8[['month_birth']])
print(x_std2)

print(np.mean(x_std))
print(np.std(x_std))

# (2) 로버스트 스케일링
from sklearn.preprocessing import RobustScaler
robust = RobustScaler()
robust.fit(df8[['month_birth']])
x_robust = robust.transform(df8[['month_birth']])
print(x_robust)

# (3) 최소 - 최대 스케일링
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
minmax.fit(df8[['month_birth']])
x_minmax = minmax.transform(df8[['month_birth']])
print(x_minmax)

# (4) 노멀 스케일링
from sklearn.preprocessing import Normalizer
normal = Normalizer()
normal.fit(df8['age', 'month_birth'])
x_normal = normal.transform(df8[['age', 'month_birth']])
print(x_normal)

# (5) 표준화 스케일링
from sklearn.preprocessing import StandardScaler
stand_scale = StandardScaler()
x_train_std = stand_scale.fit_transform(x_train)
x_test_std = stand_scale.transform(x_test)

