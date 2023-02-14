# 나이브 베이즈 실습

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

# 데이터 학습 (나이브 네이즈)
from sklearn.naive_bayes import GaussianNB
clf_gnb = GaussianNB()
clf_gnb.fit(X_tn_std, y_tn)

# 데이터 예측
pred_gnb = clf_gnb.predict(X_te_std)
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