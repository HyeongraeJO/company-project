# 서포트 벡터 머신

# 데이터 불러오기
import pandas as pd
df = pd.read_excel('C:/Users/jhr96/Desktop/PYTHON/excel/fault_feature 3L_300.xls')

# 피처 / 타깃 데이터 지정
X = df.drop(['target','type','m'], axis=1)
y = df.filter(items=['target'])


# 트레이닝 / 테스트 데이터 분할
from sklearn.model_selection import train_test_split
X_tn, X_te, y_tn, y_te=train_test_split(X,y,random_state=0)

# 데이터 학습
from sklearn import svm
clf_svm_lr = svm.SVC(kernel='rbf', C=10, random_state=0)
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


