# 의사 결정 나무 실습

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

# 데이터 학습 (의사결정나무)
from sklearn import tree
clf_tree = tree.DecisionTreeClassifier(random_state=0)
clf_tree.fit(X_tn_std, y_tn)

# 데이터 예측
pred_tree = clf_tree.predict(X_te_std)
print(pred_tree)

# f1 스코어 평가
from sklearn.metrics import f1_score
f1 = f1_score(y_te, pred_tree, average='macro')
print(f1)

# confusion matrix 확인
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_te, pred_tree)
print(conf_matrix)

# 분류 리포트 확인
from sklearn.metrics import classification_report
class_report = classification_report(y_te, pred_tree)
print(class_report)
