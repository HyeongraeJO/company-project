# 서포트 벡터 머신

#LDA
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt



# 데이터 불러오기
df=pd.read_excel('C:/Users/jhr96/Desktop/PYTHON/excel/fault_feature 2L_1000.xls') # ab외 정상 데이터 

# 피처 / 타깃 데이터 지정
X= df.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph','VC_am','VC_aph','IC_am','IC_aph','VC_am','VC_aph','IC_am','IC_aph','VC_bm','VC_bph','IC_bm','IC_bph','VC_bm','VC_bph','IC_bm','IC_bph','VD_am','VD_aph','ID_am','ID_aph','VD_am','VD_aph','ID_am','ID_aph','VD_bm','VD_bph','ID_bm','ID_bph','VD_bm','VD_bph','ID_bm','ID_bph'])
y = df.filter(['type'])


# 트레이닝 / 테스트 데이터 분할
from sklearn.model_selection import train_test_split
X_tn, X_te, y_tn, y_te=train_test_split(X,y,random_state=0)

# 데이터 학습
from sklearn import svm
clf_svm_lr = svm.SVC(kernel='rbf', C=1000, random_state=0)
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