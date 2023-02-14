#LDA
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

# train data
data=pd.read_excel('C:/Users/jhr96/Desktop/PYTHON/excel/fault_feature_800_new_s_train.xls') 

X=data.drop(['target','type','m'],axis=1)
y=data.filter(['target'])
z=data.filter(['type'])


X_ab=X.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph','VC_am','VC_aph','IC_am','IC_aph','VD_am','VD_aph','ID_am','ID_aph','VC_bm','VC_bph','IC_bm','IC_bph','VD_bm','VD_bph','ID_bm','ID_bph'])
X_bc=X.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph','VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph','VC_bm','VC_bph','IC_bm','IC_bph','VD_bm','VD_bph','ID_bm','ID_bph','VC_cm','VC_cph','IC_cm','IC_cph','VD_cm','VD_cph','ID_cm','ID_cph'])
X_ca=X.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph','VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VC_cm','VC_cph','IC_cm','IC_cph','VD_cm','VD_cph','ID_cm','ID_cph','VC_am','VC_aph','IC_am','IC_aph','VD_am','VD_aph','ID_am','ID_aph'])


row=y.shape[0]
yy=y.to_numpy()
z=z.to_numpy()


# one-hot     
for i in range(0,row-1):
    if z[i]==4: 
        yy[i]=yy[i]-1
    elif z[i]==5: 
        yy[i]=yy[i]+11
    else: 
        yy[i]=yy[i]+23


yy=pd.DataFrame(yy)


print('yy')
print(yy)


#test data
data_te=pd.read_excel('C:/Users/jhr96/Desktop/PYTHON/excel/fault_feature_200_new_s_test.xls')

X_ab_te=data_te.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph','VC_am','VC_aph','IC_am','IC_aph','VD_am','VD_aph','ID_am','ID_aph','VC_bm','VC_bph','IC_bm','IC_bph','VD_bm','VD_bph','ID_bm','ID_bph'])
X_bc_te=data_te.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph','VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph','VC_bm','VC_bph','IC_bm','IC_bph','VD_bm','VD_bph','ID_bm','ID_bph','VC_cm','VC_cph','IC_cm','IC_cph','VD_cm','VD_cph','ID_cm','ID_cph'])
X_ca_te=data_te.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph','VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VC_cm','VC_cph','IC_cm','IC_cph','VD_cm','VD_cph','ID_cm','ID_cph','VC_am','VC_aph','IC_am','IC_aph','VD_am','VD_aph','ID_am','ID_aph'])

### 형래 있고 명찬 없음 ######

y_te=data_te.filter(['target'])
z_te=data_te.filter(['type'])


row_te=y_te.shape[0]
yy_te=y_te.to_numpy()
z_te=z_te.to_numpy()

for i in range(0,row_te-1):
    if z_te[i]==4:
        yy_te[i]=yy_te[i]-1
    elif z_te[i]==5:
        yy_te[i]=yy_te[i]+11
    else:
        yy_te[i]=yy_te[i]+23

    ''' 정상 데이터??
    else:         # 정상데이터 있을 시에 
        yy[i]=0
    '''

yy_te=pd.DataFrame(yy_te)


#tn_lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components=3)
lda1 = lda.fit(X_ab,y)
lda2 = lda.fit(X_bc,y)
lda3 = lda.fit(X_ca,y)


Xab_lda=lda.transform(X_ab)
Xbc_lda=lda.transform(X_bc)
Xca_lda=lda.transform(X_ca)


print(Xab_lda.shape)
print(Xbc_lda.shape)
print(Xca_lda.shape)


print(lda.explained_variance_ratio_)

Xab_lda_df = pd.DataFrame(Xab_lda)
Xbc_lda_df = pd.DataFrame(Xbc_lda)
Xca_lda_df = pd.DataFrame(Xca_lda)

X=pd.concat([Xab_lda_df,Xbc_lda_df,Xca_lda_df],axis=1)
print(X.shape)

### 형래 있고 명찬 없음 ####


Xab_lda_te=lda1.transform(X_ab_te)
Xbc_lda_te=lda2.transform(X_ab_te)
Xca_lda_te=lda3.transform(X_ca_te)


print(Xab_lda_te.shape)
print(Xbc_lda_te.shape)
print(Xca_lda_te.shape)

Xab_lda_te_df = pd.DataFrame(Xab_lda_te)
Xbc_lda_te_df = pd.DataFrame(Xbc_lda_te)
Xca_lda_te_df = pd.DataFrame(Xca_lda_te)

X_te=pd.concat([Xab_lda_te_df,Xbc_lda_te_df,Xca_lda_te_df],axis=1)
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



# dense 2의 배수, 2^7까지, kernel, dptimizer
# https://yeomko.tistory.com/39 active funtion 

# Adding the input layer and first hidden layer / Dense 노드의 수 
# layer 값이 너무 많아도 gradient vanish를 유발할 수 있어서 조심 / input_dim 9개의 input /
classifier.add(Dense(32, kernel_initializer='he_normal', activation = 'leaky_relu', input_dim = 9))
# Adding the second hidden layer
classifier.add(Dense(64, kernel_initializer='he_normal', activation = 'leaky_relu'))
# Adding the third hidden layer
# classifier.add(Dense(20, kernel_initializer='he_normal', activation = 'leaky_relu'))

# Adding the output layer Dense 최종 아웃풋 층을 지정  18개(고장 위치 a,b,c) 값이 나와야함
classifier.add(Dense(36, kernel_initializer='he_normal', activation = 'softmax'))
classifier.summary()
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
# yy 클래스이름 
results=classifier.fit(X, yyy, batch_size = 10, epochs = 150, validation_split = 0.2)


for i in range(150):

    if results.history['val_accuracy'][i] > results.history['val_accuracy'][i-1]:
        from keras.models import load_model
        classifier.save('C:/Users/jhr96/Desktop/PYTHON/excel/HR.h5')
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

