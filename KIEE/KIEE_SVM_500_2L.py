# 서포트 벡터 머신

# 데이터 불러오기
import pandas as pd
df = pd.read_excel('/Users/jhr96/Desktop/PYTHON/excel/fault_feature_500_2L.xls')

# 피처c / 타깃 데이터 지정
X = df.filter(items=['VA_am','VA_bm','VA_cm','VA_aph','VA_bph','VA_cph','IA_am','IA_bm','IA_cm','IA_aph','IA_bph','IA_cph','VB_am','VB_bm','VB_cm','VB_aph','VB_bph','VB_cph','IB_am','IB_bm','IB_cm','IB_aph','IB_bph','IB_cph'])
y = df.filter(items=['target'])

# 트레이닝 / 테스트 데이터 분할
from sklearn.model_selection import train_test_split
X_tn, X_te, y_tn, y_te=train_test_split(X,y,random_state=0)

# 데이터 학습
from sklearn import svm
clf_svm_lr = svm.SVC(kernel='rbf', C=0.001, random_state=0)
clf_svm_lr.fit(X_tn, y_tn)


# 데이터 예측
pred_svm = clf_svm_lr.predict(X_te)
print(pred_svm)


# 정확도 평가
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_te, pred_svm)
print(accuracy)

# confusion matrix 확인
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_te, pred_svm)
print(conf_matrix)


# 분류 리포트 확인
from sklearn.metrics import classification_report
class_report = classification_report(y_te, pred_svm)
print(class_report)


