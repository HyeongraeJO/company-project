# GMM 
from sklearn.mixture import GaussianMixture
import pandas as pd

df = pd.read_excel('/Users/jhr96/Desktop/PYTHON/excel/fault_feature_type1_3.xls')
X = df.filter(items=['IA_am','IA_bm','IA_cm','IA_aph','IA_bph','IA_cph','IB_am','IB_bm','IB_cm','IB_aph','IB_bph','IB_cph','target','type'])
print(X)

gmm = GaussianMixture(n_components=3, random_state=0).fit(X)
gmm_cluster_labels = gmm.predict(X)
print(gmm_cluster_labels)


X['gmm_cluster'] = gmm_cluster_labels
X['type'] = X.target

X_result = X.groupby(['target'])['gmm_cluster'].value_counts()
print(X_result)
