import pandas as pd

df = pd.read_excel('/Users/jhr96/Desktop/PYTHON/fault_feature2.xls')

X = df.filter(items=['VA_am','VA_bm','VA_cm','VA_aph','VA_bph','VA_cph','IA_am','IA_bm','IA_cm','IA_aph','IA_bph','IA_cph','VB_am','VB_bm','VB_cm','VB_aph','VB_bph','VB_cph','IB_am','IB_bm','IB_cm','IB_aph','IB_bph','IB_cph',])

y = df.filter(items=['target']) - 6

# PCA를 통한 차원축소
from sklearn.decomposition import PCA 
# 줄이고 싶은 차원수 = 3
pca = PCA(n_components = 3)
pca.fit(X)
# 차원축소
X_pca = pca.transform(X)

# 데이터 차원 축소 확인
print(X_pca.shape)

# 공분산 행렬
print(pca.get_covariance())

# 고유값, 고유 벡터 확인
print(pca.singular_values_)
print(pca.components_)

# 각 주성분 벡터가 설명하는 분산
print(pca.explained_variance_)
# 결과로 나타나는 분산의 전체 분산 대비 비율
print(pca.explained_variance_ratio_)

# 차원 축소 데이터 확인 (데이터 프레임 형태)
import pandas as pd
pca_columns = ['pca_comp1', 'pca_comp2', 'pca_comp3']
X_pca_df = pd.DataFrame(X_pca, columns=pca_columns)
X_pca_df['target'] = y
print(X_pca_df.head(5))



'''
# 라벨 적용 PCA 데이터 
import matplotlib.pyplot as plt

# pca 변환 후 데이터 프레임에서 가져올 것!!
df = X_pca_df
df_1 = df[df['target']==1]
df_2 = df[df['target']==2]
df_3 = df[df['target']==3]
df_4 = df[df['target']==4]
df_5 = df[df['target']==5]
df_6 = df[df['target']==6]

X_11 = df_1['pca_comp1']
X_12 = df_2['pca_comp1']
X_13 = df_3['pca_comp1']
X_14 = df_4['pca_comp1']
X_15 = df_5['pca_comp1']
X_16 = df_6['pca_comp1']

X_21 = df_1['pca_comp2']
X_22 = df_2['pca_comp2']
X_23 = df_3['pca_comp2']
X_24 = df_4['pca_comp2']
X_25 = df_5['pca_comp2']
X_26 = df_6['pca_comp2']
'''
'''
X_31 = df_1['pca_comp3']
X_32 = df_2['pca_comp3']
X_33 = df_3['pca_comp3']
X_34 = df_4['pca_comp3']
X_35 = df_5['pca_comp3']
X_36 = df_6['pca_comp3']
'''
'''
target_1 = X_pca_df.target_names[1]
target_2 = X_pca_df.target_names[2]
target_3=  X_pca_df.target_names[3]
target_4 = X_pca_df.target_names[4]
target_5 = X_pca_df.target_names[5]
target_6 = X_pca_df.target_names[6]

plt.scatter(X_11, X_21, marker='o',label=target_1)
plt.scatter(X_12, X_22, marker='x',label=target_2)
plt.scatter(X_13, X_23, marker='^',label=target_3)
plt.scatter(X_14, X_24, marker='!',label=target_4)
plt.scatter(X_15, X_25, marker='@',label=target_5)
plt.scatter(X_16, X_26, marker='#',label=target_6)

plt.xlabel('pca_component1')
plt.ylabel('pca_component2')
plt.legend()
plt.show()
'''



# 반복문을 이용한 PCA 데이터 시각화
'''
import matplotlib.pyplot as plt
df = X_pca_df
markers = ['o', 'x', '^', 's', 'p', 'h']
labels = ['Sect1', 'Sect2', 'Sect3', 'Sect4', 'Sect5', 'Sect6']

for i, mark in enumerate(markers):
    df_i = df[df['target'] == i+1]
    target_i = labels[i]
    X1 = df_i['pca_comp1']
    X2 = df_i['pca_comp2']
    plt.scatter(X1, X2, marker = mark, label=target_i)

plt.xlabel('pca_component1')
plt.ylabel('pca_component2')
plt.legend()
plt.show()
'''

# 3차원!!!!
import numpy as np
import matplotlib.pyplot as plt

# 3차원 그래프세팅
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')

# target 별 시각화

df = X_pca_df
markers = ['o', 'x', '^', 's', 'p', 'h']
labels = ['Sect1', 'Sect2', 'Sect3', 'Sect4', 'Sect5', 'Sect6']

for i, mark in enumerate(markers):
    df_i = df[df['target'] == i+1]
    target_i = labels[i]
    X1 = df_i['pca_comp1']
    X2 = df_i['pca_comp2']
    X3 = df_i['pca_comp3']
    ax.scatter(X1, X2, X3, label=target_i)

ax.set_xlabel('pca_component1')
ax.set_ylabel('pca_component2')
ax.set_zlabel('pca_component3')
ax.legend()
plt.show()

