#LDA
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

data=pd.read_excel('C:/Users/jhr96/Desktop/PYTHON/excel/fault_feature_800_new_s_train.xls') # b,c가 정상데이터로 인식 

X=data.drop(['target','type'],axis=1)
X_bc=data.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph','VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph','VC_bm','VC_bph','IC_bm','IC_bph','VD_bm','VD_bph','ID_bm','ID_bph','VC_cm','VC_cph','IC_cm','IC_cph','VD_cm','VD_cph','ID_cm','ID_cph'])


#X_a=data.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph',] ['VC_am','VC_aph','IC_am','IC_aph','VD_am','VD_aph','ID_am','ID_aph'])
#X_b=data.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph',] ['VC_bm','VC_bph','IC_bm','IC_bph','VD_bm','VD_bph','ID_bm','ID_bph'])
#X_c=data.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph',] ['VC_cm','VC_cph','IC_cm','IC_cph','VD_cm','VD_cph','ID_cm','ID_cph'])

y=data.filter(['target'])
z=data.filter(['type'])


X_train, X_test, y_train, y_test = train_test_split(X_bc, y, 
                                                    test_size=0.2, random_state=121)

k = 32
parameters = {'k':[2,4,8,16,32,64,128], 'min_samples_split':[2,3]}




row=y.shape[0]
yy=y.to_numpy()
z=z.to_numpy()



# one-hot     
for i in range(0,row):
    if z[i]==4: # ab상 정상 0
        yy[i]=0
    elif z[i]==5: # bc상 정상 
        yy[i]=yy[i]
    elif z[i]==6: # ca상 정상 0
        yy[i]=0
    else:         # 정상데이터 있을 시에 
        yy[i]=0


yy=pd.DataFrame(yy)
z_1=pd.DataFrame(z)

print('yy')
print(yy)

#test data
data_te=pd.read_excel('C:/Users/jhr96/Desktop/PYTHON/excel/fault_feature_200_new_s_test.xls')

X_bc_te=data.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph','VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph','VC_bm','VC_bph','IC_bm','IC_bph','VD_bm','VD_bph','ID_bm','ID_bph','VC_cm','VC_cph','IC_cm','IC_cph','VD_cm','VD_cph','ID_cm','ID_cph'])


#X_a_te=data.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph'])
#X_b_te=data.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph'])
#X_c_te=data.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph'])


y_te=data.filter(['target'])
z_te=data.filter(['type'])

row_te=y_te.shape[0]
yy_te=y_te.to_numpy()
z_te=z_te.to_numpy()

for i in range(0,row_te-1):
    if z_te[i]==4:
        yy_te[i]=0
    elif z_te[i]==5:
        yy_te[i]=yy_te[i]
    elif z_te[i]==6:
        yy_te[i]=0
    else:
        yy_te[i]=0


yy_te=pd.DataFrame(yy_te)


#tn_lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components=3)
lda.fit(X_bc,y)
#lda.fit(X_b,y)
#lda.fit(X_c,y)

Xa_lda=lda.transform(X_bc)
#Xb_lda=lda.transform(X_b)
#Xc_lda=lda.transform(X_c)

print(Xa_lda.shape)
#print(Xb_lda.shape)
#print(Xc_lda.shape)

print(lda.explained_variance_ratio_)

Xa_lda_df = pd.DataFrame(Xa_lda)
#Xb_lda_df = pd.DataFrame(Xb_lda)
#Xc_lda_df = pd.DataFrame(Xc_lda)

X=pd.concat([Xa_lda_df],axis=1)
print(X.shape)

#LDA_te
lda.fit(X_bc_te,y)
#lda.fit(X_b_te,y)
#lda.fit(X_c_te,y)

Xa_lda_te=lda.transform(X_bc_te)
#Xb_lda_te=lda.transform(X_b_te)
#Xc_lda_te=lda.transform(X_c_te)

print(Xa_lda_te.shape)
#print(Xb_lda_te.shape)
#print(Xc_lda_te.shape)

Xa_lda_te_df = pd.DataFrame(Xa_lda_te)
#Xb_lda_te_df = pd.DataFrame(Xb_lda_te)
#Xc_lda_te_df = pd.DataFrame(Xc_lda_te)

X_te=pd.concat([Xa_lda_te_df],axis=1)
print(X_te.shape)


#ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
keras.utils.set_random_seed(10)

classifier = Sequential()

from keras.utils import to_categorical
yyy=to_categorical(yy)
yyy_te=to_categorical(yy_te)



#'leaky_relu' 'relu' 'elu'
# https://yeomko.tistory.com/39 active funtion 

# Adding the input layer and first hidden layer / Dense 노드의 수 
# layer 값이 너무 많아도 gradient vanish를 유발할 수 있어서 조심 / input_dim 9개의 input /
classifier.add(Dense(k, kernel_initializer='he_normal', activation = 'leaky_relu', input_dim = 3))
# Adding the second hidden layer
classifier.add(Dense(k, kernel_initializer='he_normal', activation = 'leaky_relu'))
# Adding the third hidden layer
# classifier.add(Dense(20, kernel_initializer='he_normal', activation = 'leaky_relu'))

# Adding the output layer Dense 최종 아웃풋 층을 지정  18개(고장 위치 a,b,c) 값이 나와야함
classifier.add(Dense(13, kernel_initializer='he_normal', activation = 'softmax'))
classifier.summary()
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
# yy 클래스이름 
results=classifier.fit(X, yyy, batch_size = 10, epochs = 300, validation_split = 0.2)


for i in range(50):

    if results.history['val_accuracy'][i] > results.history['val_accuracy'][i-1]:
        from keras.models import load_model
        classifier.save('C:/Users/jhr96/Desktop/PYTHON/excel/HR')
    else:
        pass

print(max(results.history['accuracy']))
print(max(results.history['val_accuracy']))
plt.plot(results.history['accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()  


####
loss_and_metrics = classifier.evaluate(X_te, yyy_te, batch_size=1)
where = classifier.predict(X_te,batch_size=1)

print('')
print('loss_and_metrics : ' + str(loss_and_metrics))
print(str(where))
#####

#from keras.models import load_model
#model = load_model("chan.h5")

#model.evaluate(X_te,yyy_te, batch_size=1)
#model.predict(X_te,batch_size=1)

grid_seq = GridSearchCV(classifier, param_grid=parameters, cv=3, refit=True)
grid_seq.fit(X_train, y_train)

scores_df = pd.DataFrame(grid_seq.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', \
           'split0_test_score', 'split1_test_score', 'split2_test_score']]
           

print('GridSearchCV 최적 파라미터:', grid_seq.best_params_)
print('GridSearchCV 최고 정확도: {0:.4f}'.format(grid_seq.best_score_))


estimator = grid_seq.best_estimator_

pred = estimator.predict(X_test)
print('테스트 데이터 세트 정확도: {0:.4f}'.format(accuracy_score(y_test,pred)))


