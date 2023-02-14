#LDA
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_excel('C:/Users/jhr96/Desktop/PYTHON/excel/fault_feature 2L_1000(9.29).xls') # b,c가 정상데이터로 인식 

X=data.drop(['target','type'],axis=1)
X_ab=data.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VC_am','VC_aph','IC_am','IC_aph','VD_am','VD_aph','ID_am','ID_aph','VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph','VC_bm','VC_bph','IC_bm','IC_bph','VD_bm','VD_bph','ID_bm','ID_bph'])

#X_a=data.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph',] ['VC_am','VC_aph','IC_am','IC_aph','VD_am','VD_aph','ID_am','ID_aph'])
#X_b=data.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph',] ['VC_bm','VC_bph','IC_bm','IC_bph','VD_bm','VD_bph','ID_bm','ID_bph'])
#X_c=data.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph',] ['VC_cm','VC_cph','IC_cm','IC_cph','VD_cm','VD_cph','ID_cm','ID_cph'])

y=data.filter(['target'])
z=data.filter(['type'])


row=y.shape[0]
yy=y.to_numpy()
z=z.to_numpy()



# one-hot     
for i in range(0,row):
    if z[i]==4: # ab상 정상 0
        yy[i]=yy[i]
    elif z[i]==5: # bc상 정상 
        yy[i]=0
    elif z[i]==6: # ca상 정상 0
        yy[i]=0
    else:         # 정상데이터 있을 시에 
        yy[i]=0


yy=pd.DataFrame(yy)
z_1=pd.DataFrame(z)

print('yy')
print(yy)



#tn_lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components=3)
lda.fit(X_ab,y)
#lda.fit(X_b,y)
#lda.fit(X_c,y)

Xab_lda=lda.transform(X_ab)
#Xb_lda=lda.transform(X_b)
#Xc_lda=lda.transform(X_c)

###############################
print( '\nLDA 적용 후 데이터 셋')
lda_columns = ['lda_comp1', 'lda_comp2', 'lda_comp3']


Xab_lda_df = pd.DataFrame(Xab_lda, columns = lda_columns)

Xab_lda_df['target'] = y

print( '\n head(5) 출력 결과 : ')
print(Xab_lda_df.head(5))

# 3차원!!!!
# 3차원 그래프세팅



# LDA 시각화
import matplotlib.pyplot as plt
df = Xab_lda_df
markers = ['o','.','X','s','D','v','^','*','s','x','_','2','1'] 

fig = plt.figure()  #figsize=(9,6)
ax = fig.add_subplot(111, projection='3d')

labels = ['0','1','2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']

for i, mark in enumerate(markers):
    X_i = df[df['target']== i]  #+1
    target_i = labels[i]
    X1 = X_i['lda_comp1']
    X2 = X_i['lda_comp2']
    X3 = X_i['lda_comp3']
    ax.scatter(X1, X2, X3 , marker=mark, label=target_i)

ax.set_xlabel('lda_component1')
ax.set_ylabel('lda_component2')
ax.set_zlabel('lda_component3')
ax.legend(bbox_to_anchor=(1.25, 1))
plt.show()


