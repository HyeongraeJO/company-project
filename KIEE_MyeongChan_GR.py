#LDA
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

# train data
data=pd.read_excel('C:/Users/CHOI/Desktop/fault_feature_1000_new_j_train.xls') # b,c가 정상데이터로 인식

X=data.drop(['target','type','m'],axis=1)
y=data.filter(['target'])
z=data.filter(['type'])


X_a=X.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VC_am','VC_aph','IC_am','IC_aph','VD_am','VD_aph','ID_am','ID_aph'])
X_b=X.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph','VC_bm','VC_bph','IC_bm','IC_bph','VD_bm','VD_bph','ID_bm','ID_bph'])
X_c=X.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph','VC_cm','VC_cph','IC_cm','IC_cph','VD_cm','VD_cph','ID_cm','ID_cph'])

row=y.shape[0]
yy=y.to_numpy()
z=z.to_numpy()

# one-hot    
for i in range(0,row-1):
    if z[i]==1:
        yy[i]=yy[i]-1
    elif z[i]==2:
        yy[i]=yy[i]+11
    elif z[i]==3:
        yy[i]=yy[i]+23


yy=pd.DataFrame(yy)

print('yy')
print(yy)


#test data
data_te=pd.read_excel('C:/Users/CHOI/Desktop/fault_feature_1000_new_j_test.xls')

X_te=data_te.drop(['target','type','m'],axis=1)

X_a_te=X_te.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VC_am','VC_aph','IC_am','IC_aph','VD_am','VD_aph','ID_am','ID_aph'])
X_b_te=X_te.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph','VC_bm','VC_bph','IC_bm','IC_bph','VD_bm','VD_bph','ID_bm','ID_bph'])
X_c_te=X_te.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph','VC_cm','VC_cph','IC_cm','IC_cph','VD_cm','VD_cph','ID_cm','ID_cph'])


y_te=data_te.filter(['target'])
z_te=data_te.filter(['type'])


row_te=y_te.shape[0]
yy_te=y_te.to_numpy()
z_te=z_te.to_numpy()

for i in range(0,row_te-1):
    if z_te[i]==1:
        yy_te[i]=yy_te[i]-1
    elif z_te[i]==2:
        yy_te[i]=yy_te[i]+11
    elif z_te[i]==3:
        yy_te[i]=yy_te[i]+23


yy_te=pd.DataFrame(yy_te)
print('yy_te')
print(yy_te)
######################tn_lda#############################

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components=3)
lda1 = lda.fit(X_a,y)
lda2 = lda.fit(X_b,y)
lda3 = lda.fit(X_c,y)


Xa_lda=lda1.transform(X_a)
Xb_lda=lda2.transform(X_b)
Xc_lda=lda3.transform(X_c)


print(Xa_lda.shape)
print(Xb_lda.shape)
print(Xc_lda.shape)


print(lda.explained_variance_ratio_)

Xa_lda_df = pd.DataFrame(Xa_lda)
Xb_lda_df = pd.DataFrame(Xb_lda)
Xc_lda_df = pd.DataFrame(Xc_lda)

X=pd.concat([Xa_lda_df,Xb_lda_df,Xc_lda_df],axis=1)
print(X.shape)


################LDA_te##################################

Xa_lda_te=lda1.transform(X_a_te)
Xb_lda_te=lda2.transform(X_b_te)
Xc_lda_te=lda3.transform(X_c_te)

print(Xa_lda_te.shape)
print(Xb_lda_te.shape)
print(Xc_lda_te.shape)

Xa_lda_te_df = pd.DataFrame(Xa_lda_te)
Xb_lda_te_df = pd.DataFrame(Xb_lda_te)
Xc_lda_te_df = pd.DataFrame(Xc_lda_te)


X_te=pd.concat([Xa_lda_te_df,Xb_lda_te_df,Xc_lda_te_df],axis=1)
print(X_te.shape)


#ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
keras.utils.set_random_seed(10)



from keras.utils import to_categorical
yyy=to_categorical(yy)
yyy_te=to_categorical(yy_te)




def build_classifier(Dense_1, Dense_2, initializer_1, initializer_2,initializer_3, optimizer):
    classifier = Sequential()
# Adding the input layer and first hidden layer / Dense 노드의 수
# layer 값이 너무 많아도 gradient vanish를 유발할 수 있어서 조심 / input_dim 9개의 input /
    classifier.add(Dense(Dense_1, kernel_initializer=initializer_1, activation = 'leaky_relu', input_dim = 9))
# Adding the second hidden layer
    classifier.add(Dense(Dense_2, kernel_initializer=initializer_2, activation = 'leaky_relu'))
# Adding the output layer Dense 최종 아웃풋 층을 지정  18개(고장 위치 a,b,c) 값이 나와야함
    classifier.add(Dense(36, kernel_initializer=initializer_3, activation = 'softmax'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)


parameters = {
              'Dense_1': list(range(1,128)),
              'Dense_2': list(range(1,128)),
              'epochs': [200],
              'initializer_1' : ['he_noraml','glorot_normal','glorot_normal'],
              'initializer_2' : ['he_noraml','glorot_normal','glorot_normal'],
              'initializer_3' : ['he_noraml','glorot_normal','glorot_normal'],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)



grid_search = grid_search.fit(X, yyy)


best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print(best_parameters)
print(best_accuracy)