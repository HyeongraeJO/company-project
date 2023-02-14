import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



# 꽃 데이터 불러오기 
raw_iris = datasets.load_iris()

# 피처/타깃
X = raw_iris.data
y = raw_iris.target

# 트레이닝 / 테스트 데이터 분할
X_tn, X_te, y_tn, y_te = train_test_split(X,y,random_state=0)

# 표준화 스케일
std_scale = StandardScaler()
std_scale.fit(X_tn)
X_tn_std = std_scale.transform(X_tn)
X_te_std = std_scale.transform(X_te)

# 그리드 서치
best_accuracy = 0

for k in [1,2,3,4,5,6,7,8,9,10]:
    clf_knn = KNeighborsClassifier(n_neighbors=k)
    clf_knn.fit(X_tn_std, y_tn)
    knn_pred = clf_knn.predict(X_te_std)
    accuracy = accuracy_score(y_te, knn_pred)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        final_k = {'k': k}

print(final_k)
print(accuracy)
