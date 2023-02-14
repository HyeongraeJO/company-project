# 크로스 밸리데이션 실습 (교차 검증)

# 데이터 불러오기
from sklearn import datasets
raw_wine = datasets.load_wine()

# 피처, 타깃 데이터 지정
X = raw_wine.data
y = raw_wine.target

# 트레이닝 / 테스트 데이터 분할
from sklearn.model_selection import train_test_split
X_tn, X_te, y_tn, y_te=train_test_split(X,y,random_state=0)

# 데이터 표준화
from sklearn.preprocessing import StandardScaler
std_scale = StandardScaler()
std_scale.fit(X_tn)
X_tn_std = std_scale.transform(X_tn)
X_te_std = std_scale.transform(X_te)

# 그리드 서치
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

param_grid = {'kernel': ('linear', 'rbf'),
             'C' : [0.5, 1, 10, 100]}

kfold = StratifiedKFold(n_splits=5, shuffle = True, random_state=0)
svc = svm.SVC(random_state=0)
grid_cv = GridSearchCV(svc, param_grid, cv=kfold, scoring = 'accuracy')
grid_cv.fit(X_tn_std, y_tn)

# 그리드 서치 결과 확인 (print시 에러!!!!!!)
grid_cv.cv_results_

# 그리드 서치 결과 확인 (데이터 프레임)
import numpy as np
import pandas as pd
np.transpose(pd.DataFrame(grid_cv.cv_results_))

# 베스트 스코어 & 하이퍼파라미터
grid_cv.best_score_
print(grid_cv.best_score_)

# 최종 모형 (독같이 나오지 않음)
clf = grid_cv.best_estimator_
print(clf)

# 크로스 밸리데이션 스코어 확인(1)
from sklearn.model_selection import cross_validate 
metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
cv_scores = cross_validate(clf, X_tn_std, y_tn, cv=kfold, scoring=metrics)
print(cv_scores)

# 크로스 밸리데이션 스코어 확인(2)
from sklearn.model_selection import cross_val_score
cv_score = cross_val_score(clf, X_tn_std, y_tn, cv=kfold, scoring='accuracy')
print(cv_score)
print(cv_score.mean())
print(cv_score.std())

# 예측 
pred_svm = clf.predict(X_te_std)
print(pred_svm)

# 정확도
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_te, pred_svm)
print(accuracy)

# confusion matrix 확인
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_te, pred_svm)
print(conf_matrix)

# 분류 리포트 확인 (그림???)
from sklearn.metrics import classification_report
class_report = classification_report(y_te, pred_svm)
print(class_report)

