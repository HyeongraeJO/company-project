# k-최근접 이웃 실습

# 데이터 불러오기
from sklearn import datasets
raw_iris = datasets.load_iris()

# 피처, 타깃 데이터 지정
X = raw_iris.data
y = raw_iris.target

# 트레이닝 / 테스트 데이터 분할
from sklearn.model_selection import train_test_split
X_tn, X_te, y_tn, y_te=train_test_split(X,y,random_state=0)

# 데이터 표준화
from sklearn.preprocessing import StandardScaler
std_scale = StandardScaler()
std_scale.fit(X_tn)
X_tn_std = std_scale.transform(X_tn)
X_te_std = std_scale.transform(X_te)

# 데이터 학습 (k-최근접이웃)
from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier(n_neighbors=2)
clf_knn.fit(X_tn_std, y_tn)

# 데이터 예측
knn_pred = clf_knn.predict(X_te_std)
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

