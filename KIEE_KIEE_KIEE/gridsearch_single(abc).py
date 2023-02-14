import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
from keras.utils import to_categorical

data=pd.read_excel('C:/Users/jhr96/Desktop/PYTHON/excel/train 1L 1000.xlsx')

epochs = 200

X_l = data.drop(['target','type'],axis=1)
y_l = data.filter(['target']) # '-1' to_categorical -> target 1~12 까지 되어있으면 칸을 13칸 줘 0~12 --> 그래서 -1을 해줘서 범위를 0~11로 변환
z_l = data.filter(['type'])

row_l = y_l.shape[0]
yy_l = y_l.to_numpy()
z_l = z_l.to_numpy()

lda = LinearDiscriminantAnalysis(n_components=3)

X_a=X_l.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VC_am','VC_aph','IC_am','IC_aph','VD_am','VD_aph','ID_am','ID_aph'])

yy_a = np.zeros((row_l, 1))

for i in range(0,row_l):
    if z_l[i]==1: 
        yy_a[i]=yy_l[i]
    elif z_l[i]==2:
        yy_a[i]=0
    elif z_l[i]==3:
        yy_a[i]=0
    else:
        yy_a[i]=0

yy_a=pd.DataFrame(yy_a)

lda6 = lda.fit(X_a,yy_a)
Xa_lda = lda6.transform(X_a)
Xa_lda_df = pd.DataFrame(Xa_lda)
Xa = pd.concat([Xa_lda_df],axis=1)

yyy_a = to_categorical(yy_a)
keras.utils.set_random_seed(10)

# SJ 서치 시작 준비~~~~
dense_var1 = [2, 4, 8, 16, 32, 64, 128, 256, 512]
dense_var2 = [2, 4, 8, 16, 32, 64, 128, 256, 512]

# 빈 데이터 프레임 생성
columns = ['dense1', 'dense2', 'accuracy', 'val_accuracy']
conclusion = pd.DataFrame(columns=columns)

# SJ 서치 (for 문 이용), Sequential : 순차적으로 레이어 층을 더해주는 모델
for dense1 in dense_var1:
    for dense2 in dense_var2:
        classifier = Sequential()


# ANN (Artificial Neural Network)

        classifier.add(Dense(dense1, kernel_initializer='he_normal', activation = 'leaky_relu', input_dim = 3))
        classifier.add(Dense(dense2, kernel_initializer='he_normal', activation = 'leaky_relu'))
        classifier.add(Dense(13, kernel_initializer='he_normal', activation = 'softmax'))
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        results=classifier.fit(Xa, yyy_a, batch_size = 10, epochs = epochs, validation_split = 0.2)
        
        # SJ 서치 어디까지 됐는지 편하게 확인
        print('dense1 : %.1d / dense2 : %.1d Complete!!' %(dense1, dense2))

        # 엑셀에 내용을 저장하기 위한 준비
        conclusion_2 = pd.DataFrame({'dense1': [dense1],
                                    'dense2': [dense2],
                                    'accuracy': [max(results.history['accuracy'])],
                                    'val_accuracy':[max(results.history['val_accuracy'])]})


        # 결과를 모두 합침
        conclusion = pd.concat([conclusion, conclusion_2], ignore_index=True)


        # 마무리 : 엑셀에 저장
conclusion.to_excel('conclusion_a.xlsx')

# conclusion 이용해서 제일 좋은 성능 나오는 dense1 dense2 를 빼오는 코딩 1줄

############################################################################################


X_b = X_l.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph','VC_bm','VC_bph','IC_bm','IC_bph','VD_bm','VD_bph','ID_bm','ID_bph'])

yy_b = np.zeros((row_l, 1))

for i in range(0,row_l):
    if z_l[i] == 1: 
        yy_b[i] = 0
    elif z_l[i] == 2:
        yy_b[i] = yy_l[i]
    elif z_l[i] == 3:
        yy_b[i] = 0
    else:
        yy_b[i] = 0

yy_b = pd.DataFrame(yy_b)

lda5 = lda.fit(X_b,yy_b)
Xb_lda = lda5.transform(X_b)
Xb_lda_df = pd.DataFrame(Xb_lda)
Xb = pd.concat([Xb_lda_df],axis=1)

yyy_b = to_categorical(yy_b)

# 빈 데이터 프레임 생성
columns = ['dense1', 'dense2', 'accuracy', 'val_accuracy']
conclusion = pd.DataFrame(columns=columns)

# SJ 서치 (for 문 이용), Sequential : 순차적으로 레이어 층을 더해주는 모델
for dense1 in dense_var1:
    for dense2 in dense_var2:
        classifier2 = Sequential()

# ANN (Artificial Neural Network)

        classifier2.add(Dense(dense1, kernel_initializer='he_normal', activation = 'leaky_relu', input_dim = 3))
        classifier2.add(Dense(dense2, kernel_initializer='he_normal', activation = 'leaky_relu'))
        classifier2.add(Dense(13, kernel_initializer='he_normal', activation = 'softmax'))
        classifier2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        results2=classifier2.fit(Xb, yyy_b, batch_size = 10, epochs = epochs, validation_split = 0.2)
        
        # SJ 서치 어디까지 됐는지 편하게 확인
        print('dense1 : %.1d / dense2 : %.1d Complete!!' %(dense1, dense2))

        # 엑셀에 내용을 저장하기 위한 준비
        conclusion_2 = pd.DataFrame({'dense1': [dense1],
                                    'dense2': [dense2],
                                    'accuracy': [max(results2.history['accuracy'])],
                                    'val_accuracy':[max(results2.history['val_accuracy'])]})

        # 결과를 모두 합침
        conclusion = pd.concat([conclusion, conclusion_2], ignore_index=True)

        # 마무리 : 엑셀에 저장
conclusion.to_excel('conclusion_b.xlsx')

#############################################################################################

X_c=X_l.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph','VC_cm','VC_cph','IC_cm','IC_cph','VD_cm','VD_cph','ID_cm','ID_cph'])

yy_c = np.zeros((row_l, 1))

for i in range(0,row_l):
    if z_l[i] == 1:
        yy_c[i] = 0
    elif z_l[i]==2:
        yy_c[i] = 0
    elif z_l[i] == 3:
        yy_c[i] = yy_l[i]
    else:
        yy_c[i] = 0

yy_c = pd.DataFrame(yy_c)

lda4 = lda.fit(X_c,yy_c)
Xc_lda = lda4.transform(X_c)
Xc_lda_df = pd.DataFrame(Xc_lda)
Xc = pd.concat([Xc_lda_df],axis=1)

yyy_c=to_categorical(yy_c)

# 빈 데이터 프레임 생성
columns = ['dense1', 'dense2', 'accuracy', 'val_accuracy']
conclusion = pd.DataFrame(columns=columns)

# SJ 서치 (for 문 이용), Sequential : 순차적으로 레이어 층을 더해주는 모델
for dense1 in dense_var1:
    for dense2 in dense_var2:
        classifier3 = Sequential()

# ANN (Artificial Neural Network)

        classifier3.add(Dense(dense1, kernel_initializer='he_normal', activation = 'leaky_relu', input_dim = 3))
        classifier3.add(Dense(dense2, kernel_initializer='he_normal', activation = 'leaky_relu'))
        classifier3.add(Dense(13, kernel_initializer='he_normal', activation = 'softmax'))
        classifier3.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        results3=classifier3.fit(Xc, yyy_c, batch_size = 10, epochs = epochs, validation_split = 0.2)
        
        # SJ 서치 어디까지 됐는지 편하게 확인
        print('dense1 : %.1d / dense2 : %.1d Complete!!' %(dense1, dense2))

        # 엑셀에 내용을 저장하기 위한 준비
        conclusion_2 = pd.DataFrame({'dense1': [dense1],
                                    'dense2': [dense2],
                                    'accuracy': [max(results3.history['accuracy'])],
                                    'val_accuracy':[max(results3.history['val_accuracy'])]})

        # 결과를 모두 합침
        conclusion = pd.concat([conclusion, conclusion_2], ignore_index=True)

        # 마무리 : 엑셀에 저장
conclusion.to_excel('conclusion_c.xlsx')

