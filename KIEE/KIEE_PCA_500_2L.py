import pandas as pd

df = pd.read_excel('/Users/jhr96/Desktop/PYTHON/excel/fault_feature_500_2L.xls')

X = df.filter(items=['VA_am','VA_bm','VA_cm','VA_aph','VA_bph','VA_cph','IA_am','IA_bm','IA_cm','IA_aph','IA_bph','IA_cph','VB_am','VB_bm','VB_cm','VB_aph','VB_bph','VB_cph','IB_am','IB_bm','IB_cm','IB_aph','IB_bph','IB_cph',])

y = df.filter(items=['target'])

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
