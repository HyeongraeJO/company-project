# 분류 문제에서의 성능 평가 (pred와 true를 비교했을 때)

from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
print(accuracy_score(y_true, y_pred))
print(accuracy_score(y_true, y_pred, normalize=False))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
confusion_matrix(y_true, y_pred)

# classification report
from sklearn.metrics import classification_report
y_true = [0, 1, 2, 2, 0]
y_pred = [0, 0, 2, 1, 0]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names = target_names))

# 회귀 문제에서의 성능 평가
# Mean Absolute Error 
from sklearn.metrics import mean_absolute_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(mean_absolute_error(y_true, y_pred))

# Mean Squared Error(MSE)
from sklearn.metrics import mean_squared_error
y_true = [3, -0.5, 2, 7]

# r2 score
from sklearn.metrics import r2_score
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(r2_score(y_true, y_pred))

# 군집 문제에서의 성능 평가
from sklearn.metrics import silhouette_score
X = [[1,2], [4,5], [2,1], [6,7], [2,3]]
labels = [0, 1, 0, 1, 0]
sil_score = silhouette_score(X, labels)
print(sil_score)