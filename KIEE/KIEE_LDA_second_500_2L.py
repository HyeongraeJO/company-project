# 데이터 불러오기
import pandas as pd
df = pd.read_excel('/Users/jhr96/Desktop/PYTHON/excel/fault_feature_500_2L.xls')

# 피처 / 타깃 데이터 지정
X = df.filter(items=['VA_am','VA_bm','VA_cm','VA_aph','VA_bph','VA_cph','IA_am','IA_bm','IA_cm','IA_aph','IA_bph','IA_cph','VB_am','VB_bm','VB_cm','VB_aph','VB_bph','VB_cph','IB_am','IB_bm','IB_cm','IB_aph','IB_bph','IB_cph',])
y = df.filter(items=['target'])

# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X, y)
X_lda = lda.transform(X)

# LDA 전/후 데이터 차원 비교
print('\nLDA 전 / 후 데이터 차원')
print(X.shape)
print(X_lda.shape)

# LDA 상수항, 가중벡터 확인
print( '\nLDA 상수항, 가중벡터')
print(lda.intercept_)
print(lda.coef_)

# LDA 적용 후 데이터 셋
print( '\nLDA 적용 후 데이터 셋')
import pandas as pd
lda_columns = ['lda_comp1', 'lda_comp2']
X_lda_df = pd.DataFrame(X_lda, columns = lda_columns)
X_lda_df['target'] = y
print(X_lda_df.head(5))



# LDA 시각화
import matplotlib.pyplot as plt
df = X_lda_df
markers = ['o','x','^']
labels = ['Sect1', 'Sect2', 'Sect3']

for i, mark in enumerate(markers):
    X_i = df[df['target']== i+1]
    target_i = labels[i]
    X1 = X_i['lda_comp1']
    X2 = X_i['lda_comp2']
    plt.scatter(X1, X2, label=target_i)


plt.xlabel('lda_component1')
plt.ylabel('lda_component2')
plt.legend()
plt.show()


