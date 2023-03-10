# 제목 : 3개의 전원이 있는 양방향 조류 스마트그리드의 1선 지락, 2선 단락 고장 상/구간 판별 알고리즘
# 2선 단락 고장 
# 상 : CA상


#LDA Import
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt


# traning data : 2선 단락 고장 데이터 1000개 불러오기
data=pd.read_excel('C:/Users/jhr96/Desktop/PYTHON/excel/fault_feature 2L_1000(9.29).xls') 


# 변수 X : 데이터에서 ca상에 관련된 열
X=data.drop(['target','type'],axis=1)
X_ca=data.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph','VC_cm','VC_cph','IC_cm','IC_cph','VD_cm','VD_cph','ID_cm','ID_cph','VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VC_am','VC_aph','IC_am','IC_aph','VD_am','VD_aph','ID_am','ID_aph'])


# 변수 y : 고장 구간 ('target')
y=data.filter(['target'])

# 변수 z : 고장 구간 ('type')
z=data.filter(['type'])

# shape[0] : 행의 개수, shape[1] : 열의 개수
row=y.shape[0]

# 첫번째 열 : 순서, 두번째 열 : y 혹은 z 값 표시
yy=y.to_numpy()
z=z.to_numpy()


# one-hot : z(고장종류)가 4이면 y(고장구간)이 0이다. 원하는 ca상이면 고장 구간 각각 하나씩 넣어준다. (고장 종류가 아니면 yy = 0)
for i in range(0,row):
    if z[i]==4: 
        yy[i]=0
    elif z[i]==5: 
        yy[i]=0
    elif z[i]==6: 
        yy[i]=yy[i]
    else:         
        yy[i]=0

# yy(고장구간), z(고장종류)를 데이터 프레임화
yy=pd.DataFrame(yy)
z_1=pd.DataFrame(z)


# test data : 2선 단락 고장 데이터 200개 불러오기
data_te=pd.read_excel('C:/Users/jhr96/Desktop/PYTHON/excel/fault_feature_test_2L_200(9.29).xls')
X_ca_te=data_te.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph','VC_cm','VC_cph','IC_cm','IC_cph','VD_cm','VD_cph','ID_cm','ID_cph','VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VC_am','VC_aph','IC_am','IC_aph','VD_am','VD_aph','ID_am','ID_aph'])


# 변수 y_te : 고장 구간 ('target')
y_te=data_te.filter(['target'])

# 변수 z_te : 고장 구간 ('type')
z_te=data_te.filter(['type'])

#  shape[0] : 행의 개수, shape[1] : 열의 개수
row_te=y_te.shape[0]

# 첫번째 열 : 순서, 두번째 열 : y 혹은 z 값 표시
yy_te=y_te.to_numpy()
z_te=z_te.to_numpy()


# one-hot : z(고장종류)가 4이면 y(고장구간)이 0이다. 원하는 ca상이면 고장 구간 각각 하나씩 넣어준다
'''
-1은 왜 하였는지???
'''
for i in range(0,row_te-1):
    if z_te[i]==4:
        yy_te[i]=0
    elif z_te[i]==5:
        yy_te[i]=0
    elif z_te[i]==6:
        yy_te[i]=yy_te[i]
    else:
        yy_te[i]=0


# yy(고장구간), z(고장종류)를 데이터 프레임화
yy_te=pd.DataFrame(yy_te)
'''
z_te(고장종류)는 데이터 프레임화 왜 안해???
'''


# LDA - 트레이닝 데이터 (3차원으로 LDA를 이용하여 분석하겠다.)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components=3)

# fit : 주어진 데이터 셋의 정보를 파악하는 용도
lda.fit(X_ca,y)

# transform : fit으로 판단한 정보를 통해 데이터셋의 값을 변환
Xca_lda=lda.transform(X_ca)

# 변환하여 데이터프레임화
Xca_lda_df = pd.DataFrame(Xca_lda)

# concat : 데이터를 이어붙임. axis = 1 : 가로로 붙임 
X=pd.concat([Xca_lda_df],axis=1)

'''
한개만 있는데 이어 붙임의 의미??
'''

# LDA - 테스트 데이터 (3차원으로 LDA를 이용하여 분석하겠다.)
Xca_lda_te=lda.transform(X_ca_te)
#Xb_lda_te=lda.transform(X_b_te)
#Xc_lda_te=lda.transform(X_c_te)

Xca_lda_te_df = pd.DataFrame(Xca_lda_te)
#Xb_lda_te_df = pd.DataFrame(Xb_lda_te)
#Xc_lda_te_df = pd.DataFrame(Xc_lda_te)

X_te=pd.concat([Xca_lda_te_df],axis=1)



# ANN import
import keras
from keras.models import Sequential
from keras.layers import Dense

from keras.utils import to_categorical


# to_categorical : one-hot 인코딩을 해주는 함수
yyy=to_categorical(yy)
yyy_te=to_categorical(yy_te)

# random_seed : 내 컴퓨터의 랜덤한 값을 다른 컴퓨터에도 동일하게 얻게 함
keras.utils.set_random_seed(10)

# SJ 서치 시작 준비~~~~
dense_var1 = [2, 4, 8, 16, 32, 64, 128, 256, 512]
dense_var2 = [2, 4, 8, 16, 32, 64, 128, 256, 512]

# 빈 데이터 프레임 생성
columns = ['dense1', 'dense2', 'accuracy', 'val_accuracy', 'test_accuracy']
conclusion = pd.DataFrame(columns=columns)

# SJ 서치 (for 문 이용), Sequential : 순차적으로 레이어 층을 더해주는 모델
for dense1 in dense_var1:
    for dense2 in dense_var2:
        classifier = Sequential()


# ANN (Artificial Neural Network)
    
        # he_normal : 정규분포 초기값 설정기, activation : 활성 함수로 뉴런의 신호의 세기, leaky_relu : 기울기 소멸 문제 해결
        # 입력층 : 데이터를 받아들이는 층
        classifier.add(Dense(dense1, kernel_initializer='he_normal', activation = 'leaky_relu', input_dim = 3))

        # Adding the third hidden layer (layer : 층)
        # 은닉층 : 데이터를 처리한 노드로 구성된 층
        classifier.add(Dense(dense2, kernel_initializer='he_normal', activation = 'leaky_relu'))

        # Adding the third hidden layer
        
        # Adding the output layer Dense  
        # 출력층 : 최종 은닉층에 가중치를 곱하고 결과를 얻은 노드로 구성된 층
        # soft max : 가능한 n개의 서로 다른 출력들의 확률 분포
        '''
        # dense = 13인 이유 ? yyy(고장구간)이 0~12 즉 13개라서
        '''
        classifier.add(Dense(13, kernel_initializer='he_normal', activation = 'softmax'))

        # summary : 요약 함수
        classifier.summary()

        # compile : 최종적으로 ANN 묶어줌
        # optimizer : Neural Network를 구성하는 알고리즘 최적화  
        # adam : 세밀하게 이전 맥락 확인 + 계산 적절히
        # loss : 예측값과 실제값의 차이
        # categorical_crossentropy : 클래스가 여러개인 다중 분류 문제, 라벨이 원핫코딩일때 사용
        # metrics : 실제 화면으로 출력되는 아웃풋 (정확성)
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

        # Fitting the ANN to the Training set
        # fit 함수의 validation_split = 0.2 로 하면, training dataset 은 800개로 하여, training 시키고, 나머지 200개는 test dataset 으로 사용하여, 모델을 평가하게 된다. 
        # 예) batch_size : 몇개의 관측치에 예측을 하고 비교할 것인지
        results=classifier.fit(X, yyy, batch_size = 10, epochs = 150, validation_split = 0.2)

        
        # 저장을 해주는 코딩
        for i in range(150):

            if results.history['val_accuracy'][i] > results.history['val_accuracy'][i-1]:
                from keras.models import load_model
                classifier.save('C:/Users/jhr96/Desktop/PYTHON/excel/HR')
            else:
                pass

        
        # SJ 서치 어디까지 됐는지 편하게 확인
        print('dense1 : %.1d / dense2 : %.1d Complete!!' %(dense1, dense2))

        # 엑셀 파일에 'test_accuracy'에 저장
        loss_and_metrics = classifier.evaluate(X_te, yyy_te, batch_size=1)

        # 엑셀에 내용을 저장하기 위한 준비
        conclusion_2 = pd.DataFrame({'dense1': [dense1],
                                    'dense2': [dense2],
                                    'accuracy': [max(results.history['accuracy'])],
                                    'val_accuracy':[max(results.history['val_accuracy'])],
                                    'test_accuracy': [loss_and_metrics[1]]})


        # 결과를 모두 합침
        conclusion = pd.concat([conclusion, conclusion_2], ignore_index=True)


        # 마무리 : 엑셀에 저장
conclusion.to_excel('conclusion.xlsx')

# conclusion 이용해서 제일 좋은 성능 나오는 dense1 dense2 를 빼오는 코딩 1줄