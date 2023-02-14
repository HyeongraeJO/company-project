#  나이브 베이즈 (fault_feature)

# 데이터 불러오기
import pandas as pd
df = pd.read_excel('/Users/jhr96/Desktop/PYTHON/excel/fault_feature_500_2L.xls')

# 피처 / 타깃 데이터 지정
X = df.filter(items=['VA_am','VA_bm','VA_cm','VA_aph','VA_bph','VA_cph','IA_am','IA_bm','IA_cm','IA_aph','IA_bph','IA_cph','VB_am','VB_bm','VB_cm','VB_aph','VB_bph','VB_cph','IB_am','IB_bm','IB_cm','IB_aph','IB_bph','IB_cph',])
y = df.filter(items=['target'])

# 트레이닝 / 테스트 데이터 분할
from sklearn.model_selection import train_test_split
X_tn, X_te, y_tn, y_te=train_test_split(X,y,random_state=0)

# 데이터 학습 (나이브 네이즈)
from sklearn.naive_bayes import GaussianNB
clf_gnb = GaussianNB()
clf_gnb.fit(X_tn, y_tn)

# 데이터 예측
pred_gnb = clf_gnb.predict(X_te)
print(pred_gnb)

# 리콜 평가
from sklearn.metrics import recall_score
recall = recall_score(y_te, pred_gnb, average='macro')
print(recall)

# confusion matrix 확인
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_te, pred_gnb)
print(conf_matrix)

# 분류 리포트 확인
from sklearn.metrics import classification_report
class_report = classification_report(y_te, pred_gnb)
print(class_report)