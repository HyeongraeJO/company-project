# 데이터 불러오기
from sklearn import datasets
raw_wine = datasets.load_wine()

# 피처 / 타깃 데이터 지정
X = raw_wine.data
y = raw_wine.target

# 트레이닝 / 테스트 데이터 분할
from sklearn.model_selection import train_test_split
X_tn, X_te, y_tn, y_te=train_test_split(X,y,random_state=1)

# 데이터 표준화
from sklearn.preprocessing import StandardScaler
std_scale = StandardScaler()
std_scale.fit(X_tn)
X_tn_std = std_scale.transform(X_tn)
X_te_std = std_scale.transform(X_te)

# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_tn_std, y_tn)
X_tn_lda = lda.transform(X_tn_std)
X_te_lda = lda.transform(X_te_std)

# LDA 전/후 데이터 차원 비교
print('\nLDA 전 / 후 데이터 차원')
print(X_tn_std.shape)
print(X_tn_lda.shape)

# LDA 상수항, 가중벡터 확인
print( '\nLDA 상수항, 가중벡터')
print(lda.intercept_)
print(lda.coef_)

# LDA 적용 후 데이터 셋
print( '\nLDA 적용 후 데이터 셋')
import pandas as pd
lda_columns = ['lda_comp1', 'lda_comp2']
X_tn_lda_df = pd.DataFrame(X_tn_lda, columns = lda_columns)
X_tn_lda_df['target'] = y_tn
print(X_tn_lda_df.head(5))

# LDA 시각화
import matplotlib.pyplot as plt
df = X_tn_lda_df
markers = ['o','x','^']

for i, mark in enumerate(markers):
    X_i = df[df['target']== i]
    target_i = raw_wine.target_names[i]
    X1 = X_i['lda_comp1']
    X2 = X_i['lda_comp2']
    plt.scatter(X1, X2, marker=mark, label=target_i)
    plt.xlabel('lda_componenet1')
    plt.ylabel('lda_componenet2')
    plt.legend()
    plt.show()

# LDA 적용 후 랜덤 포레스트 학습 및 예측
from sklearn.ensemble import RandomForestClassifier
clf_rf_lda = RandomForestClassifier(max_depth=2,random_state=0)
clf_rf_lda.fit(X_tn_lda, y_tn)
pred_rf_lda = clf_rf_lda.predict(X_te_lda)

# 정확도 평가
from sklearn.metrics import accuracy_score
accuracy_lda = accuracy_score(y_te, pred_rf_lda)
print(accuracy_lda)
