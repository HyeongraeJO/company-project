# GMM 선간 단락만
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



df=pd.read_excel('/Users/jhr96/Desktop/PYTHON/excel/fault_feature_type1_3.xls')

X = df.filter(items=['VA_am','VA_bm','VA_cm','VA_aph','VA_bph','VA_cph','IA_am','IA_bm','IA_cm','IA_aph','IA_bph','IA_cph','VB_am','VB_bm','VB_cm','VB_aph','VB_bph','VB_cph','IB_am','IB_bm','IB_cm','IB_aph','IB_bph','IB_cph'])
y = df.filter(items=['type'])


# GMM 적용
from sklearn.mixture import GaussianMixture
# n_components로 미리 군집 개수 설정
gmm = GaussianMixture(n_components=3, random_state= 42002)
gmm_labels = gmm.fit_predict(df)

# GMM 후 클러스터링 레이블을 따로 설정
df['gmm_cluster'] = gmm_labels

# 실제 레이블과 GMM 클러스터링 후 레이블과 비교해보기(두 레이블 수치가 동일해야 똑같은 레이블 의미 아님!)
print(df.groupby('type')['gmm_cluster'].value_counts())