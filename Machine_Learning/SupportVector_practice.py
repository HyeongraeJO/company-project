# 서포트 벡터 머신 실습

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

# 데이터 학습
from sklearn import svm
clf_svm_lr = svm.SVC(kernel='linear', random_state=0)
clf_svm_lr.fit(X_tn_std, y_tn)

# 데이터 예측
pred_svm = clf_svm_lr.predict(X_te_std)
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
