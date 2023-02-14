# k-최근접 이웃 실습 (fault_feature)

# 데이터 불러오기
import pandas as pd
df = pd.read_excel('/Users/jhr96/Desktop/PYTHON/excel/fault_feature_500_2L.xls')

# 피처 / 타깃 데이터 지정
X = df.filter(items=['VA_am','VA_bm','VA_cm','VA_aph','VA_bph','VA_cph','IA_am','IA_bm','IA_cm','IA_aph','IA_bph','IA_cph','VB_am','VB_bm','VB_cm','VB_aph','VB_bph','VB_cph','IB_am','IB_bm','IB_cm','IB_aph','IB_bph','IB_cph',])
y = df.filter(items=['target'])

 # 트레이닝 / 테스트 데이터 분할
from sklearn.model_selection import train_test_split
X_tn, X_te, y_tn, y_te=train_test_split(X,y,random_state=0)

# 데이터 학습 (k-최근접이웃)
from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier(n_neighbors=4)
clf_knn.fit(X_tn, y_tn)

# 데이터 예측
knn_pred = clf_knn.predict(X_te)
print(knn_pred)

# 정확도 평가
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_te, knn_pred)
print(accuracy)

# confusion matrix 확인
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_te, knn_pred)
print(conf_matrix)

# 분류 리포트 확인
from sklearn.metrics import classification_report
class_report = classification_report(y_te, knn_pred)
print(class_report)
